import argparse
import ast
import contextlib
import copy
import functools
import importlib
import inspect
import logging
import math
import operator
import os
import re
import shutil
import sys
import uuid
import warnings
from argparse import Namespace, ArgumentParser, ArgumentError
from collections import OrderedDict, Counter
from dataclasses import MISSING, dataclass, _MISSING_TYPE, field
from enum import Enum, EnumMeta
from itertools import accumulate
from multiprocessing import Pool
from typing import Tuple, Optional, Dict, List, Union, Any, Type, NamedTuple, Callable

import torch
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize, compose
from omegaconf import DictConfig, open_dict, OmegaConf, MISSING, II
from torch import Tensor, nn as nn
from torch.nn import functional as F, Parameter
from torch.utils import checkpoint as checkpoint

# from fairseq.modules.layer_norm import  FusedLayerNorm
# from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

logger = logging.getLogger(__name__)


try:
    from amp_C import multi_tensor_l2norm

    multi_tensor_l2norm_available = True
except ImportError:
    multi_tensor_l2norm_available = False

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


MANIFOLD_PATH_SEP = "|"

MODEL_REGISTRY = {}
MODEL_DATACLASS_REGISTRY = {}


def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for
    subsequent quantization with Iterative Product Quantization as
    described in "Training with Quantization Noise for Extreme Model Compression"

    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """

    # if no quantization noise, don't register hook
    if p <= 0:
        return module

    # supported modules
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv = module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert (
            module.weight.size(1) % block_size == 0
        ), "Input features must be a multiple of block sizes"

    # 4D matrix
    else:
        # 1x1 convolutions
        if module.kernel_size == (1, 1):
            assert (
                module.in_channels % block_size == 0
            ), "Input channels must be a multiple of block sizes"
        # regular convolutions
        else:
            k = module.kernel_size[0] * module.kernel_size[1]
            assert k % block_size == 0, "Kernel size must be a multiple of block size"

    def _forward_pre_hook(mod, input):
        # no noise for evaluation
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight = mod.weight
                in_features = weight.size(1)
                out_features = weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks
                mask = torch.zeros(
                    in_features // block_size * out_features, device=weight.device
                )
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)

            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size == (1, 1):
                    mask = torch.zeros(
                        int(in_channels // block_size * out_channels),
                        device=weight.device,
                    )
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(
                        weight.size(0), weight.size(1), device=weight.device
                    )
                    mask.bernoulli_(p)
                    mask = (
                        mask.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
                    )

            # scale weights and apply mask
            mask = mask.to(
                torch.bool
            )  # x.bool() is not currently supported in TorchScript
            s = 1 / (1 - p)
            mod.weight.data = s * weight.masked_fill(mask, 0)

    module.register_forward_pre_hook(_forward_pre_hook)
    return module

apply_quant_noise_ = quant_noise

class StrEnumMeta(EnumMeta):
    # this is workaround for submitit pickling leading to instance checks failing in hydra for StrEnum, see
    # https://github.com/facebookresearch/hydra/issues/1156
    @classmethod
    def __instancecheck__(cls, other):
        return "enum" in str(type(other))


class StrEnum(Enum, metaclass=StrEnumMeta):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value

    def __hash__(self):
        return hash(str(self))


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)


except ImportError:
    has_fused_layernorm = False


LOG_FORMAT_CHOICES = ChoiceEnum(["json", "none", "simple", "tqdm"])
DDP_BACKEND_CHOICES = ChoiceEnum([
    "c10d",  # alias for pytorch_ddp
    "fully_sharded",  # FullyShardedDataParallel from fairscale
    "legacy_ddp",
    "no_c10d",  # alias for legacy_ddp
    "pytorch_ddp",
    "slow_mo",
])
DATASET_IMPL_CHOICES = ChoiceEnum(["raw", "lazy", "cached", "mmap", "fasta"])
GENERATION_CONSTRAINTS_CHOICES = ChoiceEnum(["ordered", "unordered"])
GENERATION_DECODING_FORMAT_CHOICES = ChoiceEnum(
    ["unigram", "ensemble", "vote", "dp", "bs"]
)
ZERO_SHARDING_CHOICES = ChoiceEnum(["none", "os"])
PIPELINE_CHECKPOINT_CHOICES = ChoiceEnum(["always", "never", "except_last"])
PRINT_ALIGNMENT_CHOICES = ChoiceEnum(["hard", "soft"])


class BaseFairseqModel(nn.Module):
    """Base class for fairseq models."""

    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            # do not set defaults so that settings defaults from various architectures still works
            gen_parser_from_dataclass(parser, dc(), delete_default=True)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"]

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            # syntactic sugar for simple models which don't have a decoder
            # (e.g., the classification tutorial)
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """

        if model_cfg is None and args is not None:
            logger.warn("using 'args' is deprecated, please update your code to use dataclass config")
            model_cfg = convert_namespace_to_omegaconf(args).model

        self.upgrade_state_dict(state_dict)

        new_state_dict = prune_state_dict(state_dict, model_cfg)
        return super().load_state_dict(new_state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, "")

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += "."

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, "upgrade_state_dict_named"):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, "upgrade_state_dict"):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

    def prepare_for_inference_(self, cfg: DictConfig):
        """Prepare model for inference."""
        kwargs = {}
        kwargs["beamable_mm_beam_size"] = (
            None
            if getattr(cfg.generation, "no_beamable_mm", False)
            else getattr(cfg.generation, "beam", 5)
        )
        kwargs["need_attn"] = getattr(cfg.generation, "print_alignment", False)
        if getattr(cfg.generation, "retain_dropout", False):
            kwargs["retain_dropout"] = cfg.generation.retain_dropout
            kwargs["retain_dropout_modules"] = cfg.generation.retain_dropout_modules
        self.make_generation_fast_(**kwargs)

    def make_generation_fast_(self, **kwargs):
        """
        Legacy entry point to optimize model for faster generation.
        Prefer prepare_for_inference_.
        """
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except (AttributeError, ValueError):  # this module didn't have weight norm
                return

        self.apply(apply_remove_weight_norm)

        def apply_make_generation_fast_(module, prefix):
            if len(prefix) > 0:
                prefix += "."

            base_func = BaseFairseqModel.make_generation_fast_
            for n, m in module.named_modules():
                if (
                    m != self
                    and hasattr(m, "make_generation_fast_")
                    # don't call this implementation again, e.g., if
                    # children modules also inherit from BaseFairseqModel
                    and m.make_generation_fast_.__func__ is not base_func
                ):
                    name = prefix + n
                    m.make_generation_fast_(name=name, **kwargs)

        apply_make_generation_fast_(self, "")

        def train(mode=True):
            if mode:
                raise RuntimeError("cannot train after make_generation_fast")

        # this model should no longer be used for training
        self.eval()
        self.train = train

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""
        seen = set()

        def apply_prepare_for_onnx_export_(module):
            if (
                module != self
                and hasattr(module, "prepare_for_onnx_export_")
                and module not in seen
            ):
                seen.add(module)
                module.prepare_for_onnx_export_(**kwargs)

        self.apply(apply_prepare_for_onnx_export_)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        **kwargs,
    ):
        """
        Load a :class:`~fairseq.models.FairseqModel` from a pre-trained model
        file. Downloads and caches the pre-trained model file if needed.

        The base implementation returns a
        :class:`~fairseq.hub_utils.GeneratorHubInterface`, which can be used to
        generate translations or sample from language models. The underlying
        :class:`~fairseq.models.FairseqModel` can be accessed via the
        *generator.models* attribute.

        Other models may override this to implement custom hub interfaces.

        Args:
            model_name_or_path (str): either the name of a pre-trained model to
                load or a path/URL to a pre-trained model state dict
            checkpoint_file (str, optional): colon-separated list of checkpoint
                files in the model archive to ensemble (default: 'model.pt')
            data_name_or_path (str, optional): point args.data to the archive
                at the given path/URL. Can start with '.' or './' to reuse the
                model archive path.
        """
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            **kwargs,
        )
        logger.info(x["args"])
        return hub_utils.GeneratorHubInterface(x["args"], x["task"], x["models"])

    @classmethod
    def hub_models(cls):
        return {}


class FairseqEncoderDecoderModel(BaseFairseqModel):
    """Base class for encoder-decoder models.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        check_type(self.encoder, FairseqEncoder)
        check_type(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


def register_model(name, dataclass=None):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        if not issubclass(cls, BaseFairseqModel):
            raise ValueError(
                "Model ({}: {}) must extend BaseFairseqModel".format(name, cls.__name__)
            )
        MODEL_REGISTRY[name] = cls
        if dataclass is not None and not issubclass(dataclass, FairseqDataclass):
            raise ValueError(
                "Dataclass {} must extend FairseqDataclass".format(dataclass)
            )

        cls.__dataclass = dataclass
        if dataclass is not None:
            MODEL_DATACLASS_REGISTRY[name] = dataclass

            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="model", node=node, provider="fairseq")

            @register_model_architecture(name, name)
            def noop(_):
                pass

        return cls

    return register_model_cls


@register_model("convtransformer")
class ConvTransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer-based Speech translation model from ESPNet-ST
    https://arxiv.org/abs/2004.10234
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    # @staticmethod
    # def add_args(parser):
    #     """Add model-specific arguments to the parser."""
    #     parser.add_argument(
    #         "--input-feat-per-channel",
    #         type=int,
    #         metavar="N",
    #         help="encoder input dime/home/nastia/fairseq/fairseq/utils.pynsion per input channel",
    #     )
    #     parser.add_argument(
    #         "--activation-fn",
    #         choices=get_available_activation_fns(),
    #         help="activation function to use",
    #     )
    #     parser.add_argument(
    #         "--dropout", type=float, metavar="D", help="dropout probability"
    #     )
    #     parser.add_argument(
    #         "--attention-dropout",
    #         type=float,
    #         metavar="D",
    #         help="dropout probability for attention weights",
    #     )
    #     parser.add_argument(
    #         "--activation-dropout",
    #         "--relu-dropout",
    #         type=float,
    #         metavar="D",
    #         help="dropout probability after activation in FFN.",
    #     )
    #     parser.add_argument(
    #         "--encoder-embed-dim",
    #         type=int,
    #         metavar="N",
    #         help="encoder embedding dimension",
    #     )
    #     parser.add_argument(
    #         "--encoder-ffn-embed-dim",
    #         type=int,
    #         metavar="N",
    #         help="encoder embedding dimension for FFN",
    #     )
    #     parser.add_argument(
    #         "--encoder-layers", type=int, metavar="N", help="num encoder layers"
    #     )
    #     parser.add_argument(
    #         "--encoder-attention-heads",
    #         type=int,
    #         metavar="N",
    #         help="num encoder attention heads",
    #     )
    #     parser.add_argument(
    #         "--encoder-normalize-before",
    #         action="store_true",
    #         help="apply layernorm before each encoder block",
    #     )
    #     parser.add_argument(
    #         "--decoder-embed-dim",
    #         type=int,
    #         metavar="N",
    #         help="decoder embedding dimension",
    #     )
    #     parser.add_argument(
    #         "--decoder-ffn-embed-dim",
    #         type=int,
    #         metavar="N",
    #         help="decoder embedding dimension for FFN",
    #     )
    #     parser.add_argument(
    #         "--decoder-layers", type=int, metavar="N", help="num decoder layers"
    #     )
    #     parser.add_argument(
    #         "--decoder-attention-heads",
    #         type=int,
    #         metavar="N",
    #         help="num decoder attention heads",
    #     )
    #     parser.add_argument(
    #         "--decoder-normalize-before",
    #         action="store_true",
    #         help="apply layernorm before each decoder block",
    #     )
    #     parser.add_argument(
    #         "--decoder-output-dim",
    #         type=int,
    #         metavar="N",
    #         help="decoder output dimension (extra linear layer if different from decoder embed dim)",
    #     )
    #     parser.add_argument(
    #         "--share-decoder-input-output-embed",
    #         action="store_true",
    #         help="share decoder input and output embeddings",
    #     )
    #     parser.add_argument(
    #         "--layernorm-embedding",
    #         action="store_true",
    #         help="add layernorm to embedding",
    #     )
    #     parser.add_argument(
    #         "--no-scale-embedding",
    #         action="store_true",
    #         help="if True, dont scale embeddings",
    #     )
    #     parser.add_argument(
    #         "--load-pretrained-encoder-from",
    #         type=str,
    #         metavar="STR",
    #         help="model to take encoder weights from (for initialization)",
    #     )
    #     parser.add_argument(
    #         "--load-pretrained-decoder-from",
    #         type=str,
    #         metavar="STR",
    #         help="model to take decoder weights from (for initialization)",
    #     )
    #     parser.add_argument(
    #         "--conv-out-channels",
    #         type=int,
    #         metavar="INT",
    #         help="the number of output channels of conv layer",
    #     )

    @classmethod
    def build_encoder(cls, args):
        encoder = ConvTransformerEncoder(args)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = TransformerDecoderNoExtra(args, task.target_dictionary, embed_tokens)
        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    @staticmethod
    @torch.jit.unused
    def set_batch_first(lprobs):
        lprobs.batch_first = True

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        if self.training:
            self.set_batch_first(lprobs)
        return lprobs

    def output_layout(self):
        return "BTD"

    """
    The forward method inherited from the base class has a **kwargs argument in
    its input, which is not supported in torchscript. This method overrites the forward
    method definition without **kwargs.
    """

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


@register_model("convtransformer_simul_trans")  # TODO: remove this from here
class SimulConvTransformerModel(ConvTransformerModel):
    """
    Implementation of the paper:

    SimulMT to SimulST: Adapting Simultaneous Text Translation to
    End-to-End Simultaneous Speech Translation

    https://www.aclweb.org/anthology/2020.aacl-main.58.pdf
    """

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        tgt_dict = task.tgt_dict

        decoder = TransformerDecoder(args, tgt_dict, embed_tokens)

        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder


class FairseqEncoder(nn.Module):
    """Base class for encoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, src_tokens, src_lengths=None, **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        """
        raise NotImplementedError

    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            return self.forward(
                src_tokens=net_input["src_tokens"],
                src_lengths=net_input["src_lengths"],
            )
        else:
            return self.forward_non_torchscript(net_input)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        return self.forward(**encoder_input)

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to `new_order`.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            `encoder_out` rearranged according to `new_order`
        """
        raise NotImplementedError

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code."""
        return state_dict

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        def _apply(m):
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

        self.apply(_apply)


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False
        self.adaptive_softmax = None


    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        raise NotImplementedError

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code."""
        return state_dict

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True


def load_pretrained_component_from_model(
    component: Union[FairseqEncoder, FairseqDecoder], checkpoint: str
):
    """
    Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
    provided `component` object. If state_dict fails to load, there may be a
    mismatch in the architecture of the corresponding `component` found in the
    `checkpoint` file.
    """
    if not PathManager.exists(checkpoint):
        raise IOError("Model file not found: {}".format(checkpoint))
    state = load_checkpoint_to_cpu(checkpoint)
    if isinstance(component, FairseqEncoder):
        component_type = "encoder"
    elif isinstance(component, FairseqDecoder):
        component_type = "decoder"
    else:
        raise ValueError(
            "component to load must be either a FairseqEncoder or "
            "FairseqDecoder. Loading other component types are not supported."
        )
    component_state_dict = OrderedDict()
    for key in state["model"].keys():
        if key.startswith(component_type):
            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            component_subkey = key[len(component_type) + 1 :]
            component_state_dict[component_subkey] = state["model"][key]
    component.load_state_dict(component_state_dict, strict=True)
    return component


def prune_state_dict(state_dict, model_cfg: Optional[DictConfig]):
    """Prune the given state_dict if desired for LayerDrop
    (https://arxiv.org/abs/1909.11556).

    Training with LayerDrop allows models to be robust to pruning at inference
    time. This function prunes state_dict to allow smaller models to be loaded
    from a larger model and re-maps the existing state_dict for this to occur.

    It's called by functions that load models from checkpoints and does not
    need to be called directly.
    """
    arch = None
    if model_cfg is not None:
        arch = (
            model_cfg._name
            if isinstance(model_cfg, DictConfig)
            else getattr(model_cfg, "arch", None)
        )

    if not model_cfg or arch is None or arch == "ptt_transformer":
        # args should not be none, but don't crash if it is.
        return state_dict

    encoder_layers_to_keep = getattr(model_cfg, "encoder_layers_to_keep", None)
    decoder_layers_to_keep = getattr(model_cfg, "decoder_layers_to_keep", None)

    if not encoder_layers_to_keep and not decoder_layers_to_keep:
        return state_dict

    # apply pruning
    logger.info(
        "Pruning model to specified layer configuration - this works best if the model was trained with LayerDrop"
    )

    def create_pruning_pass(layers_to_keep, layer_name):
        keep_layers = sorted(
            int(layer_string) for layer_string in layers_to_keep.split(",")
        )
        mapping_dict = {}
        for i in range(len(keep_layers)):
            mapping_dict[str(keep_layers[i])] = str(i)

        regex = re.compile(r"^{layer}.*\.layers\.(\d+)".format(layer=layer_name))
        return {"substitution_regex": regex, "mapping_dict": mapping_dict}

    pruning_passes = []
    if encoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(encoder_layers_to_keep, "encoder"))
    if decoder_layers_to_keep:
        pruning_passes.append(create_pruning_pass(decoder_layers_to_keep, "decoder"))

    new_state_dict = {}
    for layer_name in state_dict.keys():
        match = re.search(r"\.layers\.(\d+)\.", layer_name)
        # if layer has no number in it, it is a supporting layer, such as an
        # embedding
        if not match:
            new_state_dict[layer_name] = state_dict[layer_name]
            continue

        # otherwise, layer should be pruned.
        original_layer_number = match.group(1)
        # figure out which mapping dict to replace from
        for pruning_pass in pruning_passes:
            if original_layer_number in pruning_pass["mapping_dict"] and pruning_pass[
                "substitution_regex"
            ].search(layer_name):
                new_layer_number = pruning_pass["mapping_dict"][original_layer_number]
                substitution_match = pruning_pass["substitution_regex"].search(
                    layer_name
                )
                new_state_key = (
                    layer_name[: substitution_match.start(1)]
                    + new_layer_number
                    + layer_name[substitution_match.end(1) :]
                )
                new_state_dict[new_state_key] = state_dict[layer_name]

    # Since layers are now pruned, *_layers_to_keep are no longer needed.
    # This is more of "It would make it work fix" rather than a proper fix.
    if isinstance(model_cfg, DictConfig):
        context = open_dict(model_cfg)
    else:
        context = contextlib.ExitStack()
    with context:
        if hasattr(model_cfg, "encoder_layers_to_keep"):
            model_cfg.encoder_layers_to_keep = None
        if hasattr(model_cfg, "decoder_layers_to_keep"):
            model_cfg.decoder_layers_to_keep = None

    return new_state_dict


def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        "gelu_fast",  # deprecated
        "gelu_accurate",
        "tanh",
        "linear",
    ]





def load_checkpoint_to_cpu(path, arg_overrides=None, load_on_all_ranks=False):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility).

    If doing single-GPU training or if the checkpoint is only being loaded by at
    most one process on each node (current default behavior is for only rank 0
    to read the checkpoint from disk), load_on_all_ranks should be False to
    avoid errors from torch.distributed not having been initialized or
    torch.distributed.barrier() hanging.

    If all processes on each node may be loading the checkpoint
    simultaneously, load_on_all_ranks should be set to True to avoid I/O
    conflicts.

    There's currently no support for > 1 but < all processes loading the
    checkpoint on each node.
    """
    local_path = PathManager.get_local_path(path)
    # The locally cached file returned by get_local_path() may be stale for
    # remote files that are periodically updated/overwritten (ex:
    # checkpoint_last.pt) - so we remove the local copy, sync across processes
    # (if needed), and then download a fresh copy.
    if local_path != path and PathManager.path_requires_pathmanager(path):
        try:
            os.remove(local_path)
        except FileNotFoundError:
            # With potentially multiple processes removing the same file, the
            # file being missing is benign (missing_ok isn't available until
            # Python 3.8).
            pass
        if load_on_all_ranks:
            torch.distributed.barrier()
        local_path = PathManager.get_local_path(path)

    with open(local_path, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))

    if "args" in state and state["args"] is not None and arg_overrides is not None:
        args = state["args"]
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)

    if "cfg" in state and state["cfg"] is not None:

        # hack to be able to set Namespace in dict config. this should be removed when we update to newer
        # omegaconf version that supports object flags, or when we migrate all existing models
        from omegaconf import _utils

        old_primitive = _utils.is_primitive_type
        _utils.is_primitive_type = lambda _: True

        state["cfg"] = OmegaConf.create(state["cfg"])

        _utils.is_primitive_type = old_primitive
        OmegaConf.set_struct(state["cfg"], True)

        if arg_overrides is not None:
            overwrite_args_by_name(state["cfg"], arg_overrides)

    state = _upgrade_state_dict(state)
    return state


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""

    # add optimizer_history
    if "optimizer_history" not in state:
        state["optimizer_history"] = [
            {"criterion_name": "CrossEntropyCriterion", "best_loss": state["best_loss"]}
        ]
        state["last_optimizer_state"] = state["optimizer"]
        del state["optimizer"]
        del state["best_loss"]
    # move extra_state into sub-dictionary
    if "epoch" in state and "extra_state" not in state:
        state["extra_state"] = {
            "epoch": state["epoch"],
            "batch_offset": state["batch_offset"],
            "val_loss": state["val_loss"],
        }
        del state["epoch"]
        del state["batch_offset"]
        del state["val_loss"]
    # reduce optimizer history's memory usage (only keep the last state)
    if "optimizer" in state["optimizer_history"][-1]:
        state["last_optimizer_state"] = state["optimizer_history"][-1]["optimizer"]
        for optim_hist in state["optimizer_history"]:
            del optim_hist["optimizer"]
    # record the optimizer class name
    if "optimizer_name" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["optimizer_name"] = "FairseqNAG"
    # move best_loss into lr_scheduler_state
    if "lr_scheduler_state" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["lr_scheduler_state"] = {
            "best": state["optimizer_history"][-1]["best_loss"]
        }
        del state["optimizer_history"][-1]["best_loss"]
    # keep track of number of updates
    if "num_updates" not in state["optimizer_history"][-1]:
        state["optimizer_history"][-1]["num_updates"] = 0
    # old model checkpoints may not have separate source/target positions
    if "args" in state and hasattr(state["args"], "max_positions") and not hasattr(
        state["args"], "max_source_positions"
    ):
        state["args"].max_source_positions = state["args"].max_positions
        state["args"].max_target_positions = state["args"].max_positions
    # use stateful training data iterator
    if "train_iterator" not in state["extra_state"]:
        state["extra_state"]["train_iterator"] = {
            "epoch": state["extra_state"]["epoch"],
            "iterations_in_epoch": state["extra_state"].get("batch_offset", 0),
        }

    # backward compatibility, cfg updates
    if "args" in state and state["args"] is not None:
        # default to translation task
        if not hasattr(state["args"], "task"):
            state["args"].task = "translation"
        # --raw-text and --lazy-load are deprecated
        if getattr(state["args"], "raw_text", False):
            state["args"].dataset_impl = "raw"
        elif getattr(state["args"], "lazy_load", False):
            state["args"].dataset_impl = "lazy"
        # epochs start at 1
        if state["extra_state"]["train_iterator"] is not None:
            state["extra_state"]["train_iterator"]["epoch"] = max(
                state["extra_state"]["train_iterator"].get("epoch", 1), 1
            )
        # --remove-bpe ==> --postprocess
        if hasattr(state["args"], "remove_bpe"):
            state["args"].post_process = state["args"].remove_bpe
        # --min-lr ==> --stop-min-lr
        if hasattr(state["args"], "min_lr"):
            state["args"].stop_min_lr = state["args"].min_lr
            del state["args"].min_lr
        # binary_cross_entropy => wav2vec criterion
        if (
            hasattr(state["args"], "criterion")
            and state["args"].criterion == "binary_cross_entropy"
        ):
            state["args"].criterion = "wav2vec"
        # speech_pretraining => audio pretraining
        if (
            hasattr(state["args"], "task")
            and state["args"].task == "speech_pretraining"
        ):
            state["args"].task = "audio_pretraining"
        # audio_cpc => wav2vec
        if hasattr(state["args"], "arch") and state["args"].arch == "audio_cpc":
            state["args"].arch = "wav2vec"
        # convert legacy float learning rate to List[float]
        if hasattr(state["args"], "lr") and isinstance(state["args"].lr, float):
            state["args"].lr = [state["args"].lr]
        # convert task data arg to a string instead of List[string]
        if (
            hasattr(state["args"], "data")
            and isinstance(state["args"].data, list)
            and len(state["args"].data) > 0
        ):
            state["args"].data = state["args"].data[0]

        state["cfg"] = convert_namespace_to_omegaconf(state["args"])

    if "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
        with open_dict(cfg):
            # any upgrades for Hydra-based configs
            if (
                "task" in cfg
                and "eval_wer_config" in cfg.task
                and isinstance(cfg.task.eval_wer_config.print_alignment, bool)
            ):
                cfg.task.eval_wer_config.print_alignment = "hard"
            if "generation" in cfg and isinstance(cfg.generation.print_alignment, bool):
                cfg.generation.print_alignment = "hard"
            if (
                "model" in cfg
                and "w2v_args" in cfg.model
                and cfg.model.w2v_args is not None
                and (
                    hasattr(cfg.model.w2v_args, "task") or "task" in cfg.model.w2v_args
                )
                and isinstance(
                    cfg.model.w2v_args.task.eval_wer_config.print_alignment, bool
                )
            ):
                cfg.model.w2v_args.task.eval_wer_config.print_alignment = "hard"

    return state


@dataclass
class FairseqDataclass:
    """fairseq base dataclass that supported fetching attributes and metas"""

    model: Any = MISSING
    _name: Optional[str] = None

    @staticmethod
    def name():
        return None

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(
        self, attribute_name: str, meta: str, default: Optional[Any] = None
    ) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith(
                "${"
            ):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif (
                getattr(self, attribute_name)
                != self.__dataclass_fields__[attribute_name].default
            ):
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")

    def _get_argparse_const(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_const")

    def _get_argparse_alias(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_alias")

    def _get_choices(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "choices")


def gen_parser_from_dataclass(
    parser: ArgumentParser,
    dataclass_instance: FairseqDataclass,
    delete_default: bool = False,
) -> None:
    """convert a dataclass instance to tailing parser arguments"""

    def argparse_name(name: str):
        if name == "data":
            # normally data is positional args
            return name
        if name == "_name":
            # private member, skip
            return None
        return "--" + name.replace("_", "-")

    def get_kwargs_from_dc(
        dataclass_instance: FairseqDataclass, k: str
    ) -> Dict[str, Any]:
        """k: dataclass attributes"""

        kwargs = {}

        field_type = dataclass_instance._get_type(k)
        inter_type = interpret_dc_type(field_type)

        field_default = dataclass_instance._get_default(k)

        if isinstance(inter_type, type) and issubclass(inter_type, Enum):
            field_choices = [t.value for t in list(inter_type)]
        else:
            field_choices = None

        field_help = dataclass_instance._get_help(k)
        field_const = dataclass_instance._get_argparse_const(k)

        if isinstance(field_default, str) and field_default.startswith("${"):
            kwargs["default"] = field_default
        else:
            if field_default is MISSING:
                kwargs["required"] = True
            if field_choices is not None:
                kwargs["choices"] = field_choices
            if (
                isinstance(inter_type, type)
                and (issubclass(inter_type, List) or issubclass(inter_type, Tuple))
            ) or ("List" in str(inter_type) or "Tuple" in str(inter_type)):
                if "int" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, int)
                elif "float" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, float)
                elif "str" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, str)
                else:
                    raise NotImplementedError(
                        "parsing of type " + str(inter_type) + " is not implemented"
                    )
                if field_default is not MISSING:
                    kwargs["default"] = (
                        ",".join(map(str, field_default))
                        if field_default is not None
                        else None
                    )
            elif (
                isinstance(inter_type, type) and issubclass(inter_type, Enum)
            ) or "Enum" in str(inter_type):
                kwargs["type"] = str
                if field_default is not MISSING:
                    if isinstance(field_default, Enum):
                        kwargs["default"] = field_default.value
                    else:
                        kwargs["default"] = field_default
            elif inter_type is bool:
                kwargs["action"] = (
                    "store_false" if field_default is True else "store_true"
                )
                kwargs["default"] = field_default
            else:
                kwargs["type"] = inter_type
                if field_default is not MISSING:
                    kwargs["default"] = field_default

        kwargs["help"] = field_help
        if field_const is not None:
            kwargs["const"] = field_const
            kwargs["nargs"] = "?"

        return kwargs

    for k in dataclass_instance._get_all_attributes():
        field_name = argparse_name(dataclass_instance._get_name(k))
        field_type = dataclass_instance._get_type(k)
        if field_name is None:
            continue
        elif inspect.isclass(field_type) and issubclass(field_type, FairseqDataclass):
            gen_parser_from_dataclass(parser, field_type(), delete_default)
            continue

        kwargs = get_kwargs_from_dc(dataclass_instance, k)

        field_args = [field_name]
        alias = dataclass_instance._get_argparse_alias(k)
        if alias is not None:
            field_args.append(alias)

        if "default" in kwargs:
            if isinstance(kwargs["default"], str) and kwargs["default"].startswith(
                "${"
            ):
                if kwargs["help"] is None:
                    # this is a field with a name that will be added elsewhere
                    continue
                else:
                    del kwargs["default"]
            if delete_default and "default" in kwargs:
                del kwargs["default"]
        try:
            parser.add_argument(*field_args, **kwargs)
        except ArgumentError:
            pass


def convert_namespace_to_omegaconf(args: Namespace) -> DictConfig:
    """Convert a flat argparse.Namespace to a structured DictConfig."""

    # Here we are using field values provided in args to override counterparts inside config object
    overrides, deletes = override_module_args(args)

    # configs will be in fairseq/config after installation
    config_path = os.path.join("..", "config")

    GlobalHydra.instance().clear()

    with initialize(config_path=config_path):
        try:
            composed_cfg = compose("config", overrides=overrides, strict=False)
        except:
            logger.error("Error when composing. Overrides: " + str(overrides))
            raise

        for k in deletes:
            composed_cfg[k] = None

    cfg = OmegaConf.create(
        OmegaConf.to_container(composed_cfg, resolve=True, enum_to_str=True)
    )

    # hack to be able to set Namespace in dict config. this should be removed when we update to newer
    # omegaconf version that supports object flags, or when we migrate all existing models
    from omegaconf import _utils

    old_primitive = _utils.is_primitive_type
    _utils.is_primitive_type = lambda _: True

    if cfg.task is None and getattr(args, "task", None):
        cfg.task = Namespace(**vars(args))
        from fairseq.tasks import TASK_REGISTRY

        _set_legacy_defaults(cfg.task, TASK_REGISTRY[args.task])
        cfg.task._name = args.task
    if cfg.model is None and getattr(args, "arch", None):
        cfg.model = Namespace(**vars(args))

        _set_legacy_defaults(cfg.model, ARCH_MODEL_REGISTRY[args.arch])
        cfg.model._name = args.arch
    if cfg.optimizer is None and getattr(args, "optimizer", None):
        cfg.optimizer = Namespace(**vars(args))
        from fairseq.optim import OPTIMIZER_REGISTRY

        _set_legacy_defaults(cfg.optimizer, OPTIMIZER_REGISTRY[args.optimizer])
        cfg.optimizer._name = args.optimizer
    if cfg.lr_scheduler is None and getattr(args, "lr_scheduler", None):
        cfg.lr_scheduler = Namespace(**vars(args))
        from fairseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY

        _set_legacy_defaults(cfg.lr_scheduler, LR_SCHEDULER_REGISTRY[args.lr_scheduler])
        cfg.lr_scheduler._name = args.lr_scheduler
    if cfg.criterion is None and getattr(args, "criterion", None):
        cfg.criterion = Namespace(**vars(args))
        from fairseq.criterions import CRITERION_REGISTRY

        _set_legacy_defaults(cfg.criterion, CRITERION_REGISTRY[args.criterion])
        cfg.criterion._name = args.criterion

    _utils.is_primitive_type = old_primitive
    OmegaConf.set_struct(cfg, True)
    return cfg


def overwrite_args_by_name(cfg: DictConfig, overrides: Dict[str, any]):
    # this will be deprecated when we get rid of argparse and model_overrides logic

    from fairseq.registry import REGISTRIES

    with open_dict(cfg):
        for k in cfg.keys():
            # "k in cfg" will return false if its a "mandatory value (e.g. ???)"
            if k in cfg and isinstance(cfg[k], DictConfig):
                if k in overrides and isinstance(overrides[k], dict):
                    for ok, ov in overrides[k].items():
                        if isinstance(ov, dict) and cfg[k][ok] is not None:
                            overwrite_args_by_name(cfg[k][ok], ov)
                        else:
                            cfg[k][ok] = ov
                else:
                    overwrite_args_by_name(cfg[k], overrides)
            elif k in cfg and isinstance(cfg[k], Namespace):
                for override_key, val in overrides.items():
                    setattr(cfg[k], override_key, val)
            elif k in overrides:
                if (
                    k in REGISTRIES
                    and overrides[k] in REGISTRIES[k]["dataclass_registry"]
                ):
                    cfg[k] = DictConfig(
                        REGISTRIES[k]["dataclass_registry"][overrides[k]]
                    )
                    overwrite_args_by_name(cfg[k], overrides)
                    cfg[k]._name = overrides[k]
                else:
                    cfg[k] = overrides[k]


def eval_str_list(x, x_type=float):
    if x is None:
        return None
    if isinstance(x, str):
        if len(x) == 0:
            return []
        x = ast.literal_eval(x)
    try:
        return list(map(x_type, x))
    except TypeError:
        return [x_type(x)]


def interpret_dc_type(field_type):
    if isinstance(field_type, str):
        raise RuntimeError("field should be a type")

    if field_type == Any:
        return str

    typestring = str(field_type)
    if re.match(
        r"(typing.|^)Union\[(.*), NoneType\]$", typestring
    ) or typestring.startswith("typing.Optional"):
        return field_type.__args__[0]
    return field_type


def _set_legacy_defaults(args, cls):
    """Helper to set default arguments based on *add_args*."""
    if not hasattr(cls, "add_args"):
        return

    import argparse

    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS, allow_abbrev=False
    )
    cls.add_args(parser)
    # copied from argparse.py:
    defaults = argparse.Namespace()
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if not hasattr(defaults, action.dest):
                if action.default is not argparse.SUPPRESS:
                    setattr(defaults, action.dest, action.default)
    for key, default_value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, default_value)


def override_module_args(args: Namespace) -> Tuple[List[str], List[str]]:
    """use the field in args to overrides those in cfg"""
    overrides = []
    deletes = []

    for k in FairseqConfig.__dataclass_fields__.keys():
        overrides.extend(
            _override_attr(k, FairseqConfig.__dataclass_fields__[k].type, args)
        )

    if args is not None:
        if hasattr(args, "task"):
            from fairseq.tasks import TASK_DATACLASS_REGISTRY

            migrate_registry(
                "task", args.task, TASK_DATACLASS_REGISTRY, args, overrides, deletes
            )
        else:
            deletes.append("task")

        # these options will be set to "None" if they have not yet been migrated
        # so we can populate them with the entire flat args
        CORE_REGISTRIES = {"criterion", "optimizer", "lr_scheduler"}

        from fairseq.registry import REGISTRIES

        for k, v in REGISTRIES.items():
            if hasattr(args, k):
                migrate_registry(
                    k,
                    getattr(args, k),
                    v["dataclass_registry"],
                    args,
                    overrides,
                    deletes,
                    use_name_as_val=k not in CORE_REGISTRIES,
                )
            else:
                deletes.append(k)

        no_dc = True
        if hasattr(args, "arch"):

            if args.arch in ARCH_MODEL_REGISTRY:
                m_cls = ARCH_MODEL_REGISTRY[args.arch]
                dc = getattr(m_cls, "__dataclass", None)
                if dc is not None:
                    m_name = ARCH_MODEL_NAME_REGISTRY[args.arch]
                    overrides.append("model={}".format(m_name))
                    overrides.append("model._name={}".format(args.arch))
                    # override model params with those exist in args
                    overrides.extend(_override_attr("model", dc, args))
                    no_dc = False
        if no_dc:
            deletes.append("model")

    return overrides, deletes


def _override_attr(
    sub_node: str, data_class: Type[FairseqDataclass], args: Namespace
) -> List[str]:
    overrides = []

    if not inspect.isclass(data_class) or not issubclass(data_class, FairseqDataclass):
        return overrides

    def get_default(f):
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    for k, v in data_class.__dataclass_fields__.items():
        if k.startswith("_"):
            # private member, skip
            continue

        val = get_default(v) if not hasattr(args, k) else getattr(args, k)

        field_type = interpret_dc_type(v.type)
        if (
            isinstance(val, str)
            and not val.startswith("${")  # not interpolation
            and field_type != str
            and (
                not inspect.isclass(field_type) or not issubclass(field_type, Enum)
            )  # not choices enum
        ):
            # upgrade old models that stored complex parameters as string
            val = ast.literal_eval(val)

        if isinstance(val, tuple):
            val = list(val)

        v_type = getattr(v.type, "__origin__", None)
        if (
            (v_type is List or v_type is list or v_type is Optional)
            # skip interpolation
            and not (isinstance(val, str) and val.startswith("${"))
        ):
            # if type is int but val is float, then we will crash later - try to convert here
            if hasattr(v.type, "__args__"):
                t_args = v.type.__args__
                if len(t_args) == 1 and (t_args[0] is float or t_args[0] is int):
                    val = list(map(t_args[0], val))
        elif val is not None and (
            field_type is int or field_type is bool or field_type is float
        ):
            try:
                val = field_type(val)
            except:
                pass  # ignore errors here, they are often from interpolation args

        if val is None:
            overrides.append("{}.{}=null".format(sub_node, k))
        elif val == "":
            overrides.append("{}.{}=''".format(sub_node, k))
        elif isinstance(val, str):
            val = val.replace("'", r"\'")
            overrides.append("{}.{}='{}'".format(sub_node, k, val))
        elif isinstance(val, FairseqDataclass):
            overrides += _override_attr(f"{sub_node}.{k}", type(val), args)
        elif isinstance(val, Namespace):
            sub_overrides, _ = override_module_args(val)
            for so in sub_overrides:
                overrides.append(f"{sub_node}.{k}.{so}")
        else:
            overrides.append("{}.{}={}".format(sub_node, k, val))

    return overrides


def migrate_registry(
    name, value, registry, args, overrides, deletes, use_name_as_val=False
):
    if value in registry:
        overrides.append("{}={}".format(name, value))
        overrides.append("{}._name={}".format(name, value))
        overrides.extend(_override_attr(name, registry[value], args))
    elif use_name_as_val and value is not None:
        overrides.append("{}={}".format(name, value))
    else:
        deletes.append(name)




@dataclass
class CommonConfig(FairseqDataclass):
    # This is the core dataclass including common parameters shared by all different jobs. Please append your params to other dataclasses if they were
    # used for a particular purpose or task, such as those dedicated for `distributed training`, `optimization`, etc.
    no_progress_bar: bool = field(
        default=False, metadata={"help": "disable progress bar"}
    )
    log_interval: int = field(
        default=100,
        metadata={
            "help": "log progress every N batches (when progress bar is disabled)"
        },
    )
    log_format: Optional[LOG_FORMAT_CHOICES] = field(
        default=None, metadata={"help": "log format to use"}
    )
    tensorboard_logdir: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to save logs for tensorboard, should match --logdir "
            "of running tensorboard (default: no tensorboard logging)"
        },
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights and Biases project name to use for logging"},
    )
    azureml_logging: Optional[bool] = field(
        default=False, metadata={"help": "Log scalars to AzureML context"},
    )
    seed: int = field(
        default=1, metadata={"help": "pseudo random number generator seed"}
    )
    cpu: bool = field(default=False, metadata={"help": "use CPU instead of CUDA"})
    tpu: bool = field(default=False, metadata={"help": "use TPU instead of CUDA"})
    bf16: bool = field(default=False, metadata={"help": "use bfloat16; implies --tpu"})
    memory_efficient_bf16: bool = field(
        default=False,
        metadata={
            "help": "use a memory-efficient version of BF16 training; implies --bf16"
        },
    )
    fp16: bool = field(default=False, metadata={"help": "use FP16"})
    memory_efficient_fp16: bool = field(
        default=False,
        metadata={
            "help": "use a memory-efficient version of FP16 training; implies --fp16"
        },
    )
    fp16_no_flatten_grads: bool = field(
        default=False, metadata={"help": "don't flatten FP16 grads tensor"}
    )
    fp16_init_scale: int = field(
        default=2 ** 7, metadata={"help": "default FP16 loss scale"}
    )
    fp16_scale_window: Optional[int] = field(
        default=None,
        metadata={"help": "number of updates before increasing loss scale"},
    )
    fp16_scale_tolerance: float = field(
        default=0.0,
        metadata={
            "help": "pct of updates that can overflow before decreasing the loss scale"
        },
    )
    min_loss_scale: float = field(
        default=1e-4,
        metadata={"help": "minimum FP16 loss scale, after which training is stopped"},
    )
    threshold_loss_scale: Optional[float] = field(
        default=None, metadata={"help": "threshold FP16 loss scale from below"}
    )
    user_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to a python module containing custom extensions (tasks and/or architectures)"
        },
    )
    empty_cache_freq: int = field(
        default=0,
        metadata={"help": "how often to clear the PyTorch CUDA cache (0 to disable)"},
    )
    all_gather_list_size: int = field(
        default=16384,
        metadata={"help": "number of bytes reserved for gathering stats from workers"},
    )
    model_parallel_size: int = field(
        default=1, metadata={"help": "total number of GPUs to parallelize model over"}
    )
    quantization_config_path: Optional[str] = field(
        default=None, metadata={"help": "path to quantization config file"}
    )
    profile: bool = field(
        default=False, metadata={"help": "enable autograd profiler emit_nvtx"}
    )
    reset_logging: bool = field(
        default=False,
        metadata={
            "help": "when using Hydra, reset the logging at the beginning of training"
        },
    )
    suppress_crashes: bool = field(
        default=False,
        metadata={
            "help": "suppress crashes when training with the hydra_train entry point so that the "
                    "main method can return a value (useful for sweeps)"
        },
    )
    use_plasma_view: bool = field(
        default=False, metadata={"help": "Store indices and sizes in shared memory"}
    )
    plasma_path: Optional[str] = field(
        default="/tmp/plasma",
        metadata={
            "help": "path to run plasma_store, defaults to /tmp/plasma. Paths outside /tmp tend to fail."
        },
    )


@dataclass
class DistributedTrainingConfig(FairseqDataclass):
    distributed_world_size: int = field(
        default=max(1, torch.cuda.device_count()),
        metadata={
            "help": "total number of GPUs across all nodes (default: all visible GPUs)"
        },
    )
    distributed_rank: Optional[int] = field(
        default=0, metadata={"help": "rank of the current worker"}
    )
    distributed_backend: str = field(
        default="nccl", metadata={"help": "distributed backend"}
    )
    distributed_init_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "typically tcp://hostname:port that will be used to "
            "establish initial connetion"
        },
    )
    distributed_port: int = field(
        default=-1,
        metadata={
            "help": "port number (not required if using --distributed-init-method)"
        },
    )
    device_id: int = field(
        default=0,
        metadata={
            "help": "which GPU to use (usually configured automatically)",
            "argparse_alias": "--local_rank",
        },
    )
    distributed_no_spawn: bool = field(
        default=False,
        metadata={
            "help": "do not spawn multiple processes even if multiple GPUs are visible"
        },
    )
    ddp_backend: DDP_BACKEND_CHOICES = field(
        default="pytorch_ddp", metadata={"help": "DistributedDataParallel backend"}
    )
    bucket_cap_mb: int = field(
        default=25, metadata={"help": "bucket size for reduction"}
    )
    fix_batches_to_gpus: bool = field(
        default=False,
        metadata={
            "help": "don't shuffle batches between GPUs; this reduces overall "
            "randomness and may affect precision but avoids the cost of re-reading the data"
        },
    )
    find_unused_parameters: bool = field(
        default=False,
        metadata={
            "help": "disable unused parameter detection (not applicable to "
            "--ddp-backend=legacy_ddp)"
        },
    )
    fast_stat_sync: bool = field(
        default=False,
        metadata={"help": "[deprecated] this is now defined per Criterion"},
    )
    heartbeat_timeout: int = field(
        default=-1,
        metadata={
            "help": "kill the job if no progress is made in N seconds; "
            "set to -1 to disable"
        },
    )
    broadcast_buffers: bool = field(
        default=False,
        metadata={
            "help": "Copy non-trainable parameters between GPUs, such as "
            "batchnorm population statistics"
        },
    )
    slowmo_momentum: Optional[float] = field(
        default=None,
        metadata={
            "help": "SlowMo momentum term; by default use 0.0 for 16 GPUs, "
            "0.2 for 32 GPUs; 0.5 for 64 GPUs, 0.6 for > 64 GPUs"
        },
    )
    slowmo_algorithm: str = field(
        default="LocalSGD", metadata={"help": "whether to use LocalSGD or SGP"}
    )
    localsgd_frequency: int = field(
        default=3, metadata={"help": "Local SGD allreduce frequency"}
    )
    nprocs_per_node: int = field(
        default=max(1, torch.cuda.device_count()),
        metadata={
            "help": "number of GPUs in each node. An allreduce operation across GPUs in "
            "a node is very fast. Hence, we do allreduce across GPUs in a node, "
            "and gossip across different nodes"
        },
    )
    pipeline_model_parallel: bool = field(
        default=False,
        metadata={"help": "if set, use pipeline model parallelism across GPUs"},
    )
    pipeline_balance: Optional[str] = field(
        default=None,
        metadata={
            "help": "partition the model into N_K pieces, where each piece "
            "contains N_i layers. The sum(args.pipeline_balance) "
            "should equal the total number of layers in the model"
        },
    )
    pipeline_devices: Optional[str] = field(
        default=None,
        metadata={
            "help": "a list of device indices indicating which device to place "
            "each of the N_K partitions. The length of this list should "
            "equal the length of the --pipeline-balance argument"
        },
    )
    pipeline_chunks: Optional[int] = field(
        default=0, metadata={"help": "microbatch count for pipeline model parallelism"}
    )
    pipeline_encoder_balance: Optional[str] = field(
        default=None,
        metadata={
            "help": "partition the pipeline parallel encoder into N_K pieces, where each piece "
            "contains N_i layers. The sum(args.pipeline_encoder_balance) "
            "should equal the total number of encoder layers in the model"
        },
    )
    pipeline_encoder_devices: Optional[str] = field(
        default=None,
        metadata={
            "help": "a list of device indices indicating which device to place "
            "each of the N_K partitions. The length of this list should "
            "equal the length of the --pipeline-encoder-balance argument"
        },
    )
    pipeline_decoder_balance: Optional[str] = field(
        default=None,
        metadata={
            "help": "partition the pipeline parallel decoder into N_K pieces, where each piece "
            "contains N_i layers. The sum(args.pipeline_decoder_balance) "
            "should equal the total number of decoder layers in the model"
        },
    )
    pipeline_decoder_devices: Optional[str] = field(
        default=None,
        metadata={
            "help": "a list of device indices indicating which device to place "
            "each of the N_K partitions. The length of this list should "
            "equal the length of the --pipeline-decoder-balance argument"
        },
    )
    pipeline_checkpoint: PIPELINE_CHECKPOINT_CHOICES = field(
        default="never",
        metadata={"help": "checkpointing mode for pipeline model parallelism"},
    )
    zero_sharding: ZERO_SHARDING_CHOICES = field(
        default="none", metadata={"help": "ZeRO sharding"}
    )
    fp16: bool = II("common.fp16")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    tpu: bool = II("common.tpu")
    # configuration for --ddp-backend=fully_sharded
    no_reshard_after_forward: bool = field(
        default=False, metadata={"help": "don't reshard parameters after forward pass"},
    )
    fp32_reduce_scatter: bool = field(
        default=False, metadata={"help": "reduce-scatter grads in FP32"},
    )
    cpu_offload: bool = field(
        default=False, metadata={"help": "offload FP32 params to CPU"}
    )


@dataclass
class DatasetConfig(FairseqDataclass):
    num_workers: int = field(
        default=1, metadata={"help": "how many subprocesses to use for data loading"}
    )
    skip_invalid_size_inputs_valid_test: bool = field(
        default=False,
        metadata={"help": "ignore too long or too short lines in valid and test set"},
    )
    max_tokens: Optional[int] = field(
        default=None, metadata={"help": "maximum number of tokens in a batch"}
    )
    batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "number of examples in a batch",
            "argparse_alias": "--max-sentences",
        },
    )
    required_batch_size_multiple: int = field(
        default=8, metadata={"help": "batch size will be a multiplier of this value"}
    )
    required_seq_len_multiple: int = field(
        default=1,
        metadata={
            "help": "maximum sequence length in batch will be a multiplier of this value"
        },
    )
    dataset_impl: Optional[DATASET_IMPL_CHOICES] = field(
        default=None, metadata={"help": "output dataset implementation"}
    )
    data_buffer_size: int = field(
        default=10, metadata={"help": "Number of batches to preload"}
    )
    train_subset: str = field(
        default="train",
        metadata={"help": "data subset to use for training (e.g. train, valid, test)"},
    )
    valid_subset: str = field(
        default="valid",
        metadata={
            "help": "comma separated list of data subsets to use for validation"
            " (e.g. train, valid, test)"
        },
    )
    validate_interval: int = field(
        default=1, metadata={"help": "validate every N epochs"}
    )
    validate_interval_updates: int = field(
        default=0, metadata={"help": "validate every N updates"}
    )
    validate_after_updates: int = field(
        default=0, metadata={"help": "dont validate until reaching this many updates"}
    )
    fixed_validation_seed: Optional[int] = field(
        default=None, metadata={"help": "specified random seed for validation"}
    )
    disable_validation: bool = field(
        default=False, metadata={"help": "disable validation"}
    )
    max_tokens_valid: Optional[int] = field(
        default=II("dataset.max_tokens"),
        metadata={
            "help": "maximum number of tokens in a validation batch"
            " (defaults to --max-tokens)"
        },
    )
    batch_size_valid: Optional[int] = field(
        default=II("dataset.batch_size"),
        metadata={
            "help": "batch size of the validation batch (defaults to --batch-size)",
            "argparse_alias": "--max-sentences-valid",
        },
    )
    max_valid_steps: Optional[int] = field(default=None, metadata={'help': 'How many batches to evaluate',
                                                                   "argparse_alias": "--nval"})
    curriculum: int = field(
        default=0, metadata={"help": "don't shuffle batches for first N epochs"}
    )
    gen_subset: str = field(
        default="test",
        metadata={"help": "data subset to generate (train, valid, test)"},
    )
    num_shards: int = field(
        default=1, metadata={"help": "shard generation over N shards"}
    )
    shard_id: int = field(
        default=0, metadata={"help": "id of the shard to generate (id < num_shards)"}
    )


@dataclass
class OptimizationConfig(FairseqDataclass):
    max_epoch: int = field(
        default=0, metadata={"help": "force stop training at specified epoch"}
    )
    max_update: int = field(
        default=0, metadata={"help": "force stop training at specified update"}
    )
    stop_time_hours: float = field(
        default=0,
        metadata={
            "help": "force stop training after specified cumulative time (if >0)"
        },
    )
    clip_norm: float = field(
        default=0.0, metadata={"help": "clip threshold of gradients"}
    )
    sentence_avg: bool = field(
        default=False,
        metadata={
            "help": "normalize gradients by the number of sentences in a batch"
            " (default is to normalize by number of tokens)"
        },
    )
    update_freq: List[int] = field(
        default_factory=lambda: [1],
        metadata={"help": "update parameters every N_i batches, when in epoch i"},
    )
    lr: List[float] = field(
        default_factory=lambda: [0.25],
        metadata={
            "help": "learning rate for the first N epochs; all epochs >N using LR_N"
            " (note: this may be interpreted differently depending on --lr-scheduler)"
        },
    )
    stop_min_lr: float = field(
        default=-1.0,
        metadata={"help": "stop training when the learning rate reaches this minimum"},
    )
    use_bmuf: bool = field(
        default=False,
        metadata={
            "help": "specify global optimizer for syncing models on different GPUs/shards"
        },
    )


@dataclass
class CheckpointConfig(FairseqDataclass):
    save_dir: str = field(
        default="checkpoints", metadata={"help": "path to save checkpoints"}
    )
    restore_file: str = field(
        default="checkpoint_last.pt",
        metadata={
            "help": "filename from which to load checkpoint "
            "(default: <save-dir>/checkpoint_last.pt"
        },
    )
    finetune_from_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "finetune from a pretrained model; note that meters and lr scheduler will be reset"
        },
    )
    reset_dataloader: bool = field(
        default=False,
        metadata={
            "help": "if set, does not reload dataloader state from the checkpoint"
        },
    )
    reset_lr_scheduler: bool = field(
        default=False,
        metadata={
            "help": "if set, does not load lr scheduler state from the checkpoint"
        },
    )
    reset_meters: bool = field(
        default=False,
        metadata={"help": "if set, does not load meters from the checkpoint"},
    )
    reset_optimizer: bool = field(
        default=False,
        metadata={"help": "if set, does not load optimizer state from the checkpoint"},
    )
    optimizer_overrides: str = field(
        default="{}",
        metadata={
            "help": "a dictionary used to override optimizer args when loading a checkpoint"
        },
    )
    save_interval: int = field(
        default=1, metadata={"help": "save a checkpoint every N epochs"}
    )
    save_interval_updates: int = field(
        default=0, metadata={"help": "save a checkpoint (and validate) every N updates"}
    )
    keep_interval_updates: int = field(
        default=-1,
        metadata={
            "help": "keep the last N checkpoints saved with --save-interval-updates"
        },
    )
    keep_interval_updates_pattern: int = field(
        default=-1,
        metadata={
            "help": "when used with --keep-interval-updates, skips deleting "
                    "any checkpoints with update X where "
                    "X % keep_interval_updates_pattern == 0"
        },
    )
    keep_last_epochs: int = field(
        default=-1, metadata={"help": "keep last N epoch checkpoints"}
    )
    keep_best_checkpoints: int = field(
        default=-1, metadata={"help": "keep best N checkpoints based on scores"}
    )
    no_save: bool = field(
        default=False, metadata={"help": "don't save models or checkpoints"}
    )
    no_epoch_checkpoints: bool = field(
        default=False, metadata={"help": "only store last and best checkpoints"}
    )
    no_last_checkpoints: bool = field(
        default=False, metadata={"help": "don't store last checkpoints"}
    )
    no_save_optimizer_state: bool = field(
        default=False,
        metadata={"help": "don't save optimizer-state as part of checkpoint"},
    )
    best_checkpoint_metric: str = field(
        default="loss", metadata={"help": 'metric to use for saving "best" checkpoints'}
    )
    maximize_best_checkpoint_metric: bool = field(
        default=False,
        metadata={
            "help": 'select the largest metric value for saving "best" checkpoints'
        },
    )
    patience: int = field(
        default=-1,
        metadata={
            "help": (
                "early stop training if valid performance doesn't "
                "improve for N consecutive validation runs; note "
                "that this is influenced by --validate-interval"
            )
        },
    )
    checkpoint_suffix: str = field(
        default="", metadata={"help": "suffix to add to the checkpoint file name"}
    )
    checkpoint_shard_count: int = field(
        default=1,
        metadata={
            "help": "Number of shards containing the checkpoint - "
            "if the checkpoint is over 300GB, it is preferable "
            "to split it into shards to prevent OOM on CPU while loading "
            "the checkpoint"
        },
    )
    load_checkpoint_on_all_dp_ranks: bool = field(
        default=False,
        metadata={
            "help": "load checkpoints on all data parallel devices "
            "(default: only load on rank 0 and broadcast to other devices)"
        },
    )
    write_checkpoints_asynchronously: bool = field(
        default=False,
        metadata={
            "help": (
                "Write checkpoints asynchronously in a separate "
                "thread. NOTE: This feature is currently being tested."
            ),
            "argparse_alias": "--save-async",
        },
    )
    model_parallel_size: int = II("common.model_parallel_size")


@dataclass
class FairseqBMUFConfig(FairseqDataclass):
    block_lr: float = field(
        default=1, metadata={"help": "block learning rate for bmuf"}
    )
    block_momentum: float = field(
        default=0.875, metadata={"help": "block momentum for bmuf"}
    )
    global_sync_iter: int = field(
        default=50, metadata={"help": "Iteration for syncing global model"}
    )
    warmup_iterations: int = field(
        default=500, metadata={"help": "warmup iterations for model to broadcast"}
    )
    use_nbm: bool = field(
        default=False,
        metadata={"help": "Specify whether you want to use classical BM / Nesterov BM"},
    )
    average_sync: bool = field(
        default=False,
        metadata={
            "help": "Specify whether you want to average the local momentum after each sync"
        },
    )
    distributed_world_size: int = II("distributed_training.distributed_world_size")


@dataclass
class GenerationConfig(FairseqDataclass):
    beam: int = field(
        default=5, metadata={"help": "beam size"},
    )
    nbest: int = field(
        default=1, metadata={"help": "number of hypotheses to output"},
    )
    max_len_a: float = field(
        default=0,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length"
        },
    )
    max_len_b: int = field(
        default=200,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length"
        },
    )
    min_len: int = field(
        default=1, metadata={"help": "minimum generation length"},
    )
    match_source_len: bool = field(
        default=False, metadata={"help": "generations should match the source length"},
    )
    unnormalized: bool = field(
        default=False, metadata={"help": "compare unnormalized hypothesis scores"},
    )
    no_early_stop: bool = field(
        default=False, metadata={"help": "deprecated"},
    )
    no_beamable_mm: bool = field(
        default=False, metadata={"help": "don't use BeamableMM in attention layers"},
    )
    lenpen: float = field(
        default=1,
        metadata={
            "help": "length penalty: <1.0 favors shorter, >1.0 favors longer sentences"
        },
    )
    unkpen: float = field(
        default=0,
        metadata={
            "help": "unknown word penalty: <0 produces more unks, >0 produces fewer"
        },
    )
    replace_unk: Optional[str] = field(
        default=None,
        metadata={
            "help": "perform unknown replacement (optionally with alignment dictionary)",
            "argparse_const": "@@ ",
        },
    )
    sacrebleu: bool = field(
        default=False, metadata={"help": "score with sacrebleu"},
    )
    score_reference: bool = field(
        default=False, metadata={"help": "just score the reference translation"},
    )
    prefix_size: int = field(
        default=0,
        metadata={"help": "initialize generation by target prefix of given length"},
    )
    no_repeat_ngram_size: int = field(
        default=0,
        metadata={
            "help": "ngram blocking such that this size ngram cannot be repeated in the generation"
        },
    )
    sampling: bool = field(
        default=False,
        metadata={"help": "sample hypotheses instead of using beam search"},
    )
    sampling_topk: int = field(
        default=-1,
        metadata={"help": "sample from top K likely next words instead of all words"},
    )
    sampling_topp: float = field(
        default=-1.0,
        metadata={
            "help": "sample from the smallest set whose cumulative probability mass exceeds p for next words"
        },
    )
    constraints: Optional[GENERATION_CONSTRAINTS_CHOICES] = field(
        default=None,
        metadata={
            "help": "enables lexically constrained decoding",
            "argparse_const": "ordered",
        },
    )
    temperature: float = field(
        default=1.0, metadata={"help": "temperature for generation"},
    )
    diverse_beam_groups: int = field(
        default=-1, metadata={"help": "number of groups for Diverse Beam Search"},
    )
    diverse_beam_strength: float = field(
        default=0.5,
        metadata={"help": "strength of diversity penalty for Diverse Beam Search"},
    )
    diversity_rate: float = field(
        default=-1.0,
        metadata={"help": "strength of diversity penalty for Diverse Siblings Search"},
    )
    print_alignment: Optional[PRINT_ALIGNMENT_CHOICES] = field(
        default=None,
        metadata={
            "help": "if set, uses attention feedback to compute and print alignment to source tokens "
            "(valid options are: hard, soft, otherwise treated as hard alignment)",
            "argparse_const": "hard",
        },
    )
    print_step: bool = field(
        default=False, metadata={"help": "print steps"},
    )
    lm_path: Optional[str] = field(
        default=None, metadata={"help": "path to lm checkpoint for lm fusion"},
    )
    lm_weight: float = field(
        default=0.0, metadata={"help": "weight for lm probs for lm fusion"},
    )

    # arguments for iterative refinement generator
    iter_decode_eos_penalty: float = field(
        default=0.0,
        metadata={"help": "if > 0.0, it penalized early-stopping in decoding."},
    )
    iter_decode_max_iter: int = field(
        default=10, metadata={"help": "maximum iterations for iterative refinement."},
    )
    iter_decode_force_max_iter: bool = field(
        default=False,
        metadata={
            "help": "if set, run exact the maximum number of iterations without early stop"
        },
    )
    iter_decode_with_beam: int = field(
        default=1,
        metadata={
            "help": "if > 1, model will generate translations varying by the lengths."
        },
    )
    iter_decode_with_external_reranker: bool = field(
        default=False,
        metadata={
            "help": "if set, the last checkpoint are assumed to be a reranker to rescore the translations"
        },
    )
    retain_iter_history: bool = field(
        default=False,
        metadata={
            "help": "if set, decoding returns the whole history of iterative refinement"
        },
    )
    retain_dropout: bool = field(
        default=False, metadata={"help": "Use dropout at inference time"},
    )
    # temporarily set to Any until https://github.com/facebookresearch/hydra/issues/1117 is fixed
    # retain_dropout_modules: Optional[List[str]] = field(
    retain_dropout_modules: Any = field(
        default=None,
        metadata={
            "help": "if set, only retain dropout for the specified modules; "
            "if not set, then dropout will be retained for all modules"
        },
    )
    # special decoding format for advanced decoding.
    decoding_format: Optional[GENERATION_DECODING_FORMAT_CHOICES] = field(
        default=None,
        metadata={"help": "special decoding format for advanced decoding."},
    )
    no_seed_provided: bool = field(
        default=False,
        metadata={"help": "if set, dont use seed for initializing random generators"},
    )


@dataclass
class CommonEvalConfig(FairseqDataclass):
    path: Optional[str] = field(
        default=None, metadata={"help": "path(s) to model file(s), colon separated"},
    )
    post_process: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "post-process text by removing BPE, letter segmentation, etc. "
                "Valid options can be found in fairseq.data.utils.post_process."
            ),
            "argparse_const": "subword_nmt",
            "argparse_alias": "--remove-bpe",
        },
    )
    quiet: bool = field(default=False, metadata={"help": "only print final scores"})
    model_overrides: str = field(
        default="{}",
        metadata={
            "help": "a dictionary used to override model args at generation that were used during model training"
        },
    )
    results_path: Optional[str] = field(
        default=None, metadata={"help": "path to save eval results (optional)"}
    )


@dataclass
class EvalLMConfig(FairseqDataclass):
    output_word_probs: bool = field(
        default=False,
        metadata={
            "help": "if set, outputs words and their predicted log probabilities to standard output"
        },
    )
    output_word_stats: bool = field(
        default=False,
        metadata={
            "help": "if set, outputs word statistics such as word count, average probability, etc"
        },
    )
    context_window: int = field(
        default=0,
        metadata={
            "help": "ensures that every evaluated token has access to a context of at least this size, if possible"
        },
    )
    softmax_batch: int = field(
        default=sys.maxsize,
        metadata={
            "help": "if BxT is more than this, will batch the softmax over vocab to this amount of tokens, in order to fit into GPU memory"
        },
    )


@dataclass
class InteractiveConfig(FairseqDataclass):
    buffer_size: int = field(
        default=0,
        metadata={
            "help": "read this many sentences into a buffer before processing them"
        },
    )
    input: str = field(
        default="-", metadata={"help": "file to read from; use - for stdin"},
    )



IOPathManager = None


class PathManager:
    """
    Wrapper for insulating OSS I/O (using Python builtin operations) from
    iopath's PathManager abstraction (for transparently handling various
    internal backends).
    """

    @staticmethod
    def open(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        if IOPathManager:
            return IOPathManager.open(
                path=path,
                mode=mode,
                buffering=buffering,
                encoding=encoding,
                errors=errors,
                newline=newline,
            )
        return open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    @staticmethod
    def copy(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        if IOPathManager:
            return IOPathManager.copy(
                src_path=src_path, dst_path=dst_path, overwrite=overwrite
            )
        return shutil.copyfile(src_path, dst_path)

    @staticmethod
    def get_local_path(path: str, **kwargs) -> str:
        if IOPathManager:
            return IOPathManager.get_local_path(path, **kwargs)
        return path

    @staticmethod
    def exists(path: str) -> bool:
        if IOPathManager:
            return IOPathManager.exists(path)
        return os.path.exists(path)

    @staticmethod
    def isfile(path: str) -> bool:
        if IOPathManager:
            return IOPathManager.isfile(path)
        return os.path.isfile(path)

    @staticmethod
    def ls(path: str) -> List[str]:
        if IOPathManager:
            return IOPathManager.ls(path)
        return os.listdir(path)

    @staticmethod
    def mkdirs(path: str) -> None:
        if IOPathManager:
            return IOPathManager.mkdirs(path)
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rm(path: str) -> None:
        if IOPathManager:
            return IOPathManager.rm(path)
        os.remove(path)

    @staticmethod
    def chmod(path: str, mode: int) -> None:
        if not PathManager.path_requires_pathmanager(path):
            os.chmod(path, mode)

    @staticmethod
    def register_handler(handler) -> None:
        if IOPathManager:
            return IOPathManager.register_handler(handler=handler)

    @staticmethod
    def copy_from_local(
        local_path: str, dst_path: str, overwrite: bool = False, **kwargs
    ) -> None:
        if IOPathManager:
            return IOPathManager.copy_from_local(
                local_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs
            )
        return shutil.copyfile(local_path, dst_path)

    @staticmethod
    def path_requires_pathmanager(path: str) -> bool:
        """Do we require PathManager to access given path?"""
        if IOPathManager:
            for p in IOPathManager._path_handlers.keys():
                if path.startswith(p):
                    return True
        return False

    @staticmethod
    def supports_rename(path: str) -> bool:
        # PathManager doesn't yet support renames
        return not PathManager.path_requires_pathmanager(path)

    @staticmethod
    def rename(src: str, dst: str):
        os.rename(src, dst)

    """
    ioPath async PathManager methods:
    """
    @staticmethod
    def opena(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        """
        Return file descriptor with asynchronous write operations.
        """
        global IOPathManager
        if not IOPathManager:
            logging.info("ioPath is initializing PathManager.")
            try:

                IOPathManager = PathManager()
            except Exception:
                logging.exception("Failed to initialize ioPath PathManager object.")
        return IOPathManager.opena(
            path=path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    @staticmethod
    def async_close() -> bool:
        """
        Wait for files to be written and clean up asynchronous PathManager.
        NOTE: `PathManager.async_close()` must be called at the end of any
        script that uses `PathManager.opena(...)`.
        """
        global IOPathManager
        if IOPathManager:
            return IOPathManager.async_close()
        return False



def register_model_architecture(model_name, arch_name):
    """
    New model architectures can be added to fairseq with the
    :func:`register_model_architecture` function decorator. After registration,
    model architectures can be selected with the ``--arch`` command-line
    argument.

    For example::

        @register_model_architecture('lstm', 'lstm_luong_wmt_en_de')
        def lstm_luong_wmt_en_de(cfg):
            args.encoder_embed_dim = getattr(cfg.model, 'encoder_embed_dim', 1000)
            (...)

    The decorated function should take a single argument *cfg*, which is a
    :class:`omegaconf.DictConfig`. The decorated function should modify these
    arguments in-place to match the desired architecture.

    Args:
        model_name (str): the name of the Model (Model must already be
            registered)
        arch_name (str): the name of the model architecture (``--arch``)
    """

    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                "Cannot register model architecture for unknown model type ({})".format(
                    model_name
                )
            )
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError(
                "Cannot register duplicate model architecture ({})".format(arch_name)
            )
        if not callable(fn):
            raise ValueError(
                "Model architecture must be callable ({})".format(arch_name)
            )
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_MODEL_NAME_REGISTRY[arch_name] = model_name
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name, []).append(arch_name)
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn

    return register_model_arch_fn


ARCH_MODEL_REGISTRY = {}
ARCH_MODEL_NAME_REGISTRY = {}
ARCH_MODEL_INV_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}
EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
    ],
)


def check_type(module, expected_type):
    if hasattr(module, "unwrapped_module"):
        assert isinstance(module.unwrapped_module, expected_type), \
            f"{type(module.unwrapped_module)} != {expected_type}"
    else:
        assert isinstance(module, expected_type), f"{type(module)} != {expected_type}"


@dataclass
class FairseqConfig(FairseqDataclass):
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    bmuf: FairseqBMUFConfig = FairseqBMUFConfig()
    generation: GenerationConfig = GenerationConfig()
    eval_lm: EvalLMConfig = EvalLMConfig()
    interactive: InteractiveConfig = InteractiveConfig()
    task: Any = None
    criterion: Any = None
    optimizer: Any = None
    lr_scheduler: Any = None
    scoring: Any = None
    bpe: Any = None
    tokenizer: Any = None


class ConvTransformerEncoder(FairseqEncoder):
    """Conv + Transformer encoder"""

    def __init__(self, args):
        """Construct an Encoder object."""
        super().__init__(None)

        self.dropout = args.dropout
        self.embed_scale = (
            1.0 if args.no_scale_embedding else math.sqrt(args.encoder_embed_dim)
        )
        self.padding_idx = 1
        self.in_channels = 1
        self.input_dim = args.input_feat_per_channel
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, args.conv_out_channels, 3, stride=2, padding=3 // 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                args.conv_out_channels,
                args.conv_out_channels,
                3,
                stride=2,
                padding=3 // 2,
            ),
            torch.nn.ReLU(),
        )
        transformer_input_dim = self.infer_conv_output_dim(
            self.in_channels, self.input_dim, args.conv_out_channels
        )
        self.out = torch.nn.Linear(transformer_input_dim, args.encoder_embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions,
            args.encoder_embed_dim,
            self.padding_idx,
            learned=False,
        )

        self.transformer_layers = nn.ModuleList([])
        self.transformer_layers.extend(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def pooling_ratio(self):
        return 4

    def infer_conv_output_dim(self, in_channels, input_dim, out_channels):
        sample_seq_len = 200
        sample_bsz = 10
        x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
        x = torch.nn.Conv2d(1, out_channels, 3, stride=2, padding=3 // 2)(x)
        x = torch.nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=3 // 2)(x)
        x = x.transpose(1, 2)
        mb, seq = x.size()[:2]
        return x.contiguous().view(mb, seq, -1).size(-1)

    def forward(self, src_tokens, src_lengths):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        bsz, max_seq_len, _ = src_tokens.size()
        x = (
            src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
            .transpose(1, 2)
            .contiguous()
        )
        x = self.conv(x)
        bsz, _, output_seq_len, _ = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
        x = self.out(x)
        x = self.embed_scale * x

        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
        input_len_0 = (src_lengths.float() / subsampling_factor).ceil().long()
        input_len_1 = x.size(0) * torch.ones([src_lengths.size(0)]).long().to(input_len_0.device)
        input_lengths = torch.min(
            input_len_0, input_len_1
        )

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)

        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)

        if not encoder_padding_mask.any():
            maybe_encoder_padding_mask = None
        else:
            maybe_encoder_padding_mask = encoder_padding_mask


        return {
            "encoder_out": [x],
            "encoder_padding_mask": [maybe_encoder_padding_mask]
            if maybe_encoder_padding_mask is not None
            else [],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                (encoder_out["encoder_padding_mask"][0]).index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                (encoder_out["encoder_embedding"][0]).index_select(0, new_order)
            ]
        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,
            "encoder_padding_mask": new_encoder_padding_mask,
            "encoder_embedding": new_encoder_embedding,
            "encoder_states": encoder_states,
            "src_tokens": [],
            "src_lengths": [],
        }


class FairseqIncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_incremental_state()

    def init_incremental_state(self):
        self._incremental_state_id = str(uuid.uuid4())

    def _get_full_incremental_state_key(self, key: str) -> str:
        return "{}.{}".format(self._incremental_state_id, key)

    def get_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
    ) -> Optional[Dict[str, Optional[Tensor]]]:
        """Helper for getting incremental state for an nn.Module."""
        full_key = self._get_full_incremental_state_key(key)
        if incremental_state is None or full_key not in incremental_state:
            return None
        return incremental_state[full_key]

    def set_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        key: str,
        value: Dict[str, Optional[Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
        """Helper for setting incremental state for an nn.Module."""
        if incremental_state is not None:
            full_key = self._get_full_incremental_state_key(key)
            incremental_state[full_key] = value
        return incremental_state


DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


def with_incremental_state(cls):
    cls.__bases__ = (FairseqIncrementalState,) + tuple(
        b for b in cls.__bases__ if b != FairseqIncrementalState
    )
    return cls


@with_incremental_state
class FairseqIncrementalDecoder(FairseqDecoder):
    """Base class for incremental decoders.

    Incremental decoding is a special mode at inference time where the Model
    only receives a single timestep of input corresponding to the previous
    output token (for teacher forcing) and must produce the next output
    *incrementally*. Thus the model must cache any long-term state that is
    needed about the sequence, e.g., hidden states, convolutional states, etc.

    Compared to the standard :class:`FairseqDecoder` interface, the incremental
    decoder interface allows :func:`forward` functions to take an extra keyword
    argument (*incremental_state*) that can be used to cache state across
    time-steps.

    The :class:`FairseqIncrementalDecoder` interface also defines the
    :func:`reorder_incremental_state` method, which is used during beam search
    to select and reorder the incremental state based on the selection of beams.

    To learn more about how incremental decoding works, refer to `this blog
    <http://www.telesens.co/2019/04/21/understanding-incremental-decoding-in-fairseq/>`_.
    """

    def __init__(self, dictionary):
        super().__init__(dictionary)

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs
    ):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs
    ):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder incremental state.

        This will be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        pass

    def reorder_incremental_state_scripting(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Main entry point for reordering the incremental state.

        Due to limitations in TorchScript, we call this function in
        :class:`fairseq.sequence_generator.SequenceGenerator` instead of
        calling :func:`reorder_incremental_state` directly.
        """
        for module in self.modules():
            if hasattr(module, "reorder_incremental_state"):
                result = module.reorder_incremental_state(incremental_state, new_order)
                if result is not None:
                    incremental_state = result

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, "_beam_size", -1) != beam_size:
            seen = set()

            def apply_set_beam_size(module):
                if (
                    module != self
                    and hasattr(module, "set_beam_size")
                    and module not in seen
                ):
                    seen.add(module)
                    module.set_beam_size(beam_size)

            self.apply(apply_set_beam_size)
            self._beam_size = beam_size


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(args, dictionary, embed_tokens)

    def build_output_projection(self, args, dictionary, embed_tokens):
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                eval_str_list_secons(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        num_base_layers = getattr(args, "base_layers", 0)
        for i in range(num_base_layers):
            self.layers.insert(((i+1) * args.decoder_layers) // (num_base_layers + 1), BaseLayer(args))


    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayer(args, no_encoder_attn)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint else 0
        )
        # layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap) NOTE: remove

        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


class TransformerDecoderNoExtra(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None


@register_model_architecture(model_name="convtransformer", arch_name="convtransformer")
def base_architecture(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.max_source_positions = getattr(args, "max_source_positions", 3000)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.conv_out_channels = getattr(args, "conv_out_channels", args.encoder_embed_dim)


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


class AugmentedMemoryConvTransformerEncoder(ConvTransformerEncoder):
    def __init__(self, args):
        super().__init__(args)

        args.encoder_stride = self.stride()

        self.left_context = args.left_context // args.encoder_stride

        self.right_context = args.right_context // args.encoder_stride

        self.left_context_after_stride = args.left_context // args.encoder_stride
        self.right_context_after_stride = args.right_context // args.encoder_stride

        self.transformer_layers = nn.ModuleList([])
        self.transformer_layers.extend(
            [
                AugmentedMemoryTransformerEncoderLayer(args)
                for i in range(args.encoder_layers)
            ]
        )

    def stride(self):
        # Hard coded here. Should infer from convs in future
        stride = 4
        return stride

    def forward(self, src_tokens, src_lengths, states=None):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        bsz, max_seq_len, _ = src_tokens.size()
        x = (
            src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
            .transpose(1, 2)
            .contiguous()
        )
        x = self.conv(x)
        bsz, _, output_seq_len, _ = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
        x = self.out(x)
        x = self.embed_scale * x

        subsampling_factor = 1.0 * max_seq_len / output_seq_len
        # input_lengths = torch.max(
        #     (src_lengths.float() / subsampling_factor).ceil().long(),
        #     x.size(0) * src_lengths.new_ones([src_lengths.size(0)]).long(),
        # )
        input_lengths = x.size(0) * src_lengths.new_ones([src_lengths.size(0)]).long()

        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            input_lengths, batch_first=True
        )

        # TODO: fix positional embedding
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)

        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # State to store memory banks etc.
        if states is None:
            states = [
                {"memory_banks": None, "encoder_states": None}
                for i in range(len(self.transformer_layers))
            ]

        for i, layer in enumerate(self.transformer_layers):
            # x size:
            # (self.left_size + self.segment_size + self.right_size)
            # / self.stride, num_heads, dim
            # TODO: Consider mask here
            x = layer(x, states[i])
            states[i]["encoder_states"] = x[
                self.left_context_after_stride : -self.right_context_after_stride
            ]

        lengths = (
            (
                ~encoder_padding_mask[
                    :, self.left_context_after_stride : -self.right_context_after_stride
                ]
            )
            .sum(dim=1, keepdim=True)
            .long()
        )

        return states[-1]["encoder_states"], lengths, states


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class AugmentedMemoryTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

        self.left_context = args.left_context // args.encoder_stride
        self.right_context = args.right_context // args.encoder_stride

    def forward(self, x, state):

        length, batch_size, x_dim = x.size()

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # init_state
        if state.get("memory_banks", None) is None:
            state["memory_banks"] = []

        # TODO reseach new sum_query method
        seg_start = self.left_context
        seg_end = length - self.right_context
        if seg_start < seg_end:
            summarization_query = torch.mean(x[seg_start:seg_end], keepdim=True, dim=0)
        else:
            summarization_query = x.new_zeros(1, batch_size, x_dim)

        x = torch.cat([x, summarization_query], dim=0)

        x = self.self_attn(input_and_summary=x, state=state)

        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x

    def build_self_attention(self, embed_dim, args):
        return AugmentedMemoryMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            tanh_on_mem=True,
            max_memory_size=args.max_memory_size,
        )


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if (
            not self.onnx_trace
            and not is_tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
        ):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


class AugmentedMemoryMultiheadAttention(MultiheadAttention):
    """
    Augmented Memory Attention from
    Streaming Transformer-based Acoustic Models
    Using Self-attention with Augmented Memory
    https://arxiv.org/abs/2005.08042
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        tanh_on_mem=False,
        memory_dim=None,
        std_scale=0.5,  # 0.5 based on https://arxiv.org/abs/2005.09137
        max_memory_size=-1,
        disable_mem_on_mem_attn=True,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            self_attention,
            encoder_decoder_attention,
            q_noise,
            qn_block_size,
        )

        self.memory_dim = memory_dim if memory_dim is not None else embed_dim
        self.std_scale = std_scale
        self.disable_mem_on_mem_attn = disable_mem_on_mem_attn

        # This Operator was used for factorization in PySpeech
        self.v2e = lambda x: x

        if tanh_on_mem:
            self.squash_mem = torch.tanh
            self.nonlinear_squash_mem = True
        else:
            self.squash_mem = lambda x: x
            self.nonlinear_squash_mem = False

        self.max_memory_size = max_memory_size

    def forward(self, input_and_summary, state):
        """
        input: Encoder states of current segment with left or right context,
            plus one summarization query

        """

        length, batch_size, _ = input_and_summary.shape
        length = length - 1  # not include sum_query, last index

        memory = state["memory_banks"]
        # TODO: positional embedding on memory

        if self.max_memory_size > -1 and len(memory) > self.max_memory_size:
            # TODO: need to fix here
            if self.max_memory_size == 0:
                memory = memory.new_zeros(1, memory.size(1), self.memory_dim)
            else:
                memory = memory[-self.max_memory_size :]

        memory_and_input = torch.cat(memory + [input_and_summary[:-1]], dim=0)
        input_and_sum_query = input_and_summary

        q = self.q_proj(self.v2e(input_and_sum_query))
        k = self.k_proj(self.v2e(memory_and_input))
        v = self.v_proj(self.v2e(memory_and_input))

        q = (
            q.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
            * self.scaling
        )
        k = (
            k.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        v = (
            v.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attention_weights = torch.bmm(q, k.transpose(1, 2))

        if self.disable_mem_on_mem_attn:
            attention_weights = self.suppress_mem_on_mem_attention(
                batch_size, self.num_heads, len(memory), attention_weights
            )

        if self.std_scale is not None:
            attention_weights = attention_suppression(attention_weights, self.std_scale)

        assert list(attention_weights.shape) == [
            batch_size * self.num_heads,
            length + 1,
            length + len(memory),
        ]

        attention_weights = softmax(
            attention_weights.float(), dim=-1
        ).type_as(attention_weights)

        attention_probs = self.dropout_module(attention_weights)

        # [T, T, B, n_head] + [T, B, n_head, d_head] -> [T, B, n_head, d_head]
        attention = torch.bmm(attention_probs, v)

        assert list(attention.shape) == [
            batch_size * self.num_heads,
            length + 1,
            self.head_dim,
        ]

        attention = (
            attention.transpose(0, 1)
            .contiguous()
            .view(length + 1, batch_size, self.embed_dim)
        )

        output_and_memory = self.out_proj(attention)

        next_m = output_and_memory[-1:]
        next_m = self.squash_mem(next_m)
        output = output_and_memory[:-1]

        state["memory_banks"].append(next_m)

        return output

    def suppress_mem_on_mem_attention(
        self, B: int, num_heads: int, mem_size: int, attention_weight: Tensor
    ):
        """
        Arguments:
            - B: batch size
            - num_heads: number of attention heads
            - mem_size: size of memory bank
            - attention_weight: a [B*num_heads, T + 1, T + mem_size] vector

        Return:
            modified attention_weight with [B*num_heads, -1, :mem_size] = -inf
        """
        attention_weight[:, -1, :mem_size] = float("-inf")
        return attention_weight


class SequenceEncoder(FairseqEncoder):
    """
    SequenceEncoder encodes sequences.

    More specifically, `src_tokens` and `src_lengths` in `forward()` should
    describe a batch of "complete" sequences rather than segments.

    Segment-by-segment inference can be triggered by `segment_size`:
    1) `segment_size` is None:
        SequenceEncoder treats the input sequence as one single segment.
    2) `segment_size` is not None (some int instead):
        SequenceEncoder does the following:
            1. breaks the input sequence into several segments
            2. inference on each segment and collect the outputs
            3. concatanete segment outputs into the output sequence.
    Note that `segment_size` here shouldn't include additional left/right
    contexts needed, for example if we wish to infer with LC-BLSTM where the
    middle chunk size is 100 and right context is 20, `segment_size` should be
    100.
    """

    def __init__(self, args, module):
        super().__init__(None)

        self.module = module
        self.input_time_axis = 1
        self.output_time_axis = 0
        self.segment_size = args.segment_size
        self.left_context = args.left_context
        self.right_context = args.right_context

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        states=None,
    ):

        seg_src_tokens_lengths = sequence_to_segments(
            sequence=src_tokens,
            time_axis=self.input_time_axis,
            lengths=src_lengths,
            segment_size=self.segment_size,
            extra_left_context=self.left_context,
            extra_right_context=self.right_context,
        )

        seg_encoder_states_lengths: List[Tuple[Tensor, Tensor]] = []

        for seg_src_tokens, seg_src_lengths in seg_src_tokens_lengths:
            (seg_encoder_states, seg_enc_lengths, states) = self.module(
                seg_src_tokens,
                seg_src_lengths,
                states=states,
            )

            seg_encoder_states_lengths.append((seg_encoder_states, seg_enc_lengths))

        encoder_out, enc_lengths = segments_to_sequence(
            segments=seg_encoder_states_lengths, time_axis=self.output_time_axis
        )

        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            enc_lengths, batch_first=True
        )

        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        return {
            "encoder_out": [encoder_out],
            "encoder_padding_mask": [encoder_padding_mask],
            "encoder_embedding": [],
            "encoder_states": [states],
            "src_tokens": [],
            "src_lengths": [],
        }

    def incremental_encode(
        self,
        seg_src_tokens: Tensor,
        seg_src_lengths: Tensor,
        states=None,
    ):
        """
        Different from forward function, this function takes segmented speech
        as input, and append encoder states to previous states
        """
        (seg_encoder_states, seg_enc_lengths, states) = self.module(
            seg_src_tokens,
            seg_src_lengths,
            states=states,
        )
        return seg_encoder_states, seg_enc_lengths, states


@torch.jit.export
def sequence_to_segments(
    sequence: Tensor,
    time_axis: int,
    lengths: Tensor,
    segment_size: Optional[int] = None,
    extra_left_context: int = 0,
    extra_right_context: int = 0,
) -> List[Tuple[Tensor, Tensor]]:
    """Breaks sequence into segments."""

    sequence = pad_sequence(
        sequence=sequence,
        time_axis=time_axis,
        extra_left_context=extra_left_context,
        extra_right_context=extra_right_context,
    )

    lengths = lengths + extra_left_context + extra_right_context

    segments: List[Tuple[Tensor, Tensor]] = []

    if segment_size is None:
        segments.append((sequence, lengths))
        return segments

    offset = 0
    end = sequence.shape[time_axis]
    step = segment_size
    size = extra_left_context + segment_size + extra_right_context

    while offset + extra_left_context + extra_right_context < end:
        clamped_size = min(size, end - offset)
        segment_lengths = torch.clamp(lengths - offset, min=0, max=clamped_size)
        indices = torch.arange(
            start=offset,
            end=(offset + clamped_size),
            step=1,
            dtype=torch.long,
            device=sequence.device,
        )
        segment_tensor = torch.index_select(sequence, time_axis, indices)
        segments.append((segment_tensor, segment_lengths))
        offset = offset + step

    return segments


@torch.jit.export
def segments_to_sequence(
    segments: List[Tuple[Tensor, Tensor]], time_axis: int
) -> Tuple[Tensor, Tensor]:
    """Concatenate segments into a full sequence."""
    if len(segments) == 1:
        return segments[0]

    tensors_to_concat: List[Tensor] = []
    lengths_to_stack: List[Tensor] = []

    for tensor, lengths in segments:
        tensors_to_concat.append(tensor)
        lengths_to_stack.append(lengths)

    sequence = torch.cat(tensors_to_concat, dim=time_axis)
    lengths = torch.stack(lengths_to_stack, dim=0)
    lengths = torch.sum(lengths, dim=0)

    return sequence, lengths


def lengths_to_encoder_padding_mask(lengths, batch_first: bool = False):
    """
    convert lengths (a 1-D Long/Int tensor) to 2-D binary tensor

    Args:
        lengths: a (B, )-shaped tensor
        batch_first: whether to return a (B, T) tensor

    Return:
        max_length: maximum length of B sequences
        encoder_padding_mask: a (max_length, B) binary mask, where
        [t, b] = False for t < lengths[b] and True otherwise

    TODO:
        kernelize this function if benchmarking shows this function is slow
    """
    max_lengths = torch.max(lengths).item()
    bsz = lengths.size(0)
    encoder_padding_mask = torch.arange(
        max_lengths
    ).to(  # a (T, ) tensor with [0, ..., T-1]
        lengths.device
    ).view(  # move to the right device
        1, max_lengths
    ).expand(  # reshape to (1, T)-shaped tensor
        bsz, -1
    ) > lengths.view(  # expand to (B, T)-shaped tensor
        bsz, 1
    ).expand(
        -1, max_lengths
    )
    if not batch_first:
        return encoder_padding_mask.t(), max_lengths
    else:
        return encoder_padding_mask, max_lengths


def attention_suppression(attention_weights: Tensor, scale: float):
    # B, H, qlen, klen -> B, H, qlen, 1
    attention_prob = softmax(attention_weights.float(), dim=-1)
    attention_nozeros = attention_prob.to(torch.bool)
    nozeros_sum = torch.sum(attention_nozeros.to(torch.float), dim=-1, keepdim=True)

    # For very sparse situation, we need get round about 0s
    key_sum = torch.sum(attention_prob, dim=-1, keepdim=True)

    # nozeros_sum should > 1
    key_mean = key_sum / (nozeros_sum + 1e-8)

    # std calculation
    dis = (attention_prob - key_mean) * (attention_prob - key_mean)

    # if attention_prob[i] < threshold, then dis_masked[i] = 0; for all i
    dis_masked = torch.where(
        attention_nozeros, dis, attention_prob.new_zeros(attention_prob.size())
    )

    key_var = torch.sum(dis_masked, dim=-1, keepdim=True)
    key_var = key_var / (nozeros_sum - 1.0 + 1e-8)
    key_std = torch.sqrt(key_var)
    key_thread = key_mean - scale * key_std

    # if attention_prob[i] >= key_thread, then attention_prob[i]
    # , otherwise "-inf"
    inf_tensor = attention_prob.new_zeros(attention_prob.size()).detach()
    inf_tensor[:] = float("-inf")
    attention_weights_float = torch.where(
        attention_prob < key_thread,
        inf_tensor,
        attention_weights.float(),
    )

    return attention_weights_float.type_as(attention_weights)


@torch.jit.export
def pad_sequence(
    sequence: Tensor,
    time_axis: int,
    extra_left_context: int = 0,
    extra_right_context: int = 0,
) -> Tensor:
    """Pad extra left/right contexts to the sequence."""

    if extra_left_context == 0 and extra_right_context == 0:
        return sequence

    tensors_to_concat = []

    if extra_left_context:
        size = (extra_left_context,)
        fill_value = 0
        indices = torch.full(
            size=size,
            fill_value=fill_value,
            dtype=torch.long,
            device=sequence.device,
        )
        left_padding = torch.index_select(sequence, time_axis, indices)
        tensors_to_concat.append(left_padding)

    tensors_to_concat.append(sequence)

    # NOTE(cfyeh): for efficiency reason we pad 0 instead of the last frame for
    #              extra right contexts.
    if extra_right_context:
        size = list(sequence.shape)
        size[time_axis] = extra_right_context
        right_padding = torch.zeros(size, dtype=sequence.dtype, device=sequence.device)
        tensors_to_concat.append(right_padding)

    padded_sequence = torch.cat(tensors_to_concat, dim=time_axis)
    return padded_sequence


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m



def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    learned: bool = False,
):
    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim,
            padding_idx,
            init_size=num_embeddings + padding_idx + 1,
        )
    return m


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class FairseqDropout(nn.Module):
    def __init__(self, p, module_name=None):
        super().__init__()
        self.p = p
        self.module_name = module_name
        self.apply_during_inference = False

    def forward(self, x, inplace: bool = False):
        if self.p > 0 and (self.training or self.apply_during_inference):
            return F.dropout(x, p=self.p, training=True, inplace=inplace)
        else:
            return x

    def make_generation_fast_(
        self,
        name: str,
        retain_dropout: bool = False,
        retain_dropout_modules: Optional[List[str]] = None,
        **kwargs
    ):
        if retain_dropout:
            if retain_dropout_modules is not None and self.module_name is None:
                logger.warning(
                    "Cannot enable dropout during inference for module {} "
                    "because module_name was not set".format(name)
                )
            elif (
                retain_dropout_modules is None  # if None, apply to all modules
                or self.module_name in retain_dropout_modules
            ):
                logger.info(
                    "Enabling dropout during inference for module: {}".format(name)
                )
                self.apply_during_inference = True
            else:
                logger.info("Disabling dropout for module: {}".format(name))


class LayerDropModuleList(nn.ModuleList):
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`.

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m


class AdaptiveSoftmax(nn.Module):
    """
    This is an implementation of the efficient softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax
    approximation for GPUs" (http://arxiv.org/abs/1609.04309).
    """

    def __init__(
        self,
        vocab_size,
        input_dim,
        cutoff,
        dropout,
        factor=4.0,
        adaptive_inputs=None,
        tie_proj=False,
        q_noise=0,
        qn_block_size=8,
    ):
        super().__init__()

        if vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        else:
            assert (
                vocab_size == cutoff[-1]
            ), "cannot specify cutoff larger than vocab size"

        output_dim = cutoff[0] + len(cutoff) - 1

        self.vocab_size = vocab_size
        self.cutoff = cutoff
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.input_dim = input_dim
        self.factor = factor
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size

        self.lsm = nn.LogSoftmax(dim=1)

        if adaptive_inputs is not None:
            self.head = TiedHeadModule(
                adaptive_inputs.weights_for_band(0),
                input_dim,
                len(cutoff) - 1,
                self.q_noise,
                self.qn_block_size,
            )
        else:
            self.head = quant_noise(
                nn.Linear(input_dim, output_dim, bias=False),
                self.q_noise,
                self.qn_block_size,
            )

        self._make_tail(adaptive_inputs, tie_proj)

        def init_weights(m):
            if (
                hasattr(m, "weight")
                and not isinstance(m, TiedLinear)
                and not isinstance(m, TiedHeadModule)
            ):
                nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)

        self.register_buffer("version", torch.LongTensor([1]))

    def _make_tail(self, adaptive_inputs=None, tie_proj=False):
        self.tail = nn.ModuleList()
        for i in range(len(self.cutoff) - 1):
            dim = int(self.input_dim // self.factor ** (i + 1))

            tied_emb, tied_proj = (
                adaptive_inputs.weights_for_band(i + 1)
                if adaptive_inputs is not None
                else (None, None)
            )

            if tied_proj is not None:
                if tie_proj:
                    proj = quant_noise(
                        TiedLinear(tied_proj, transpose=True),
                        self.q_noise,
                        self.qn_block_size,
                    )
                else:
                    proj = quant_noise(
                        nn.Linear(tied_proj.size(0), tied_proj.size(1), bias=False),
                        self.q_noise,
                        self.qn_block_size,
                    )
            else:
                proj = quant_noise(
                    nn.Linear(self.input_dim, dim, bias=False),
                    self.q_noise,
                    self.qn_block_size,
                )

            if tied_emb is None:
                out_proj = nn.Linear(
                    dim, self.cutoff[i + 1] - self.cutoff[i], bias=False
                )
            else:
                out_proj = TiedLinear(tied_emb, transpose=False)

            m = nn.Sequential(
                proj,
                nn.Dropout(self.dropout_module.p),
                quant_noise(out_proj, self.q_noise, self.qn_block_size),
            )

            self.tail.append(m)

    def upgrade_state_dict_named(self, state_dict, name):
        version_name = name + ".version"
        if version_name not in state_dict:
            raise Exception("This version of the model is no longer supported")

    def adapt_target(self, target):
        """
        In order to be efficient, the AdaptiveSoftMax does not compute the
        scores for all the word of the vocabulary for all the examples. It is
        thus necessary to call the method adapt_target of the AdaptiveSoftMax
        layer inside each forward pass.
        """

        target = target.view(-1)
        new_target = [target.clone()]
        target_idxs = []

        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            new_target[0][mask] = self.cutoff[0] + i

            if mask.any():
                target_idxs.append(mask.nonzero(as_tuple=False).squeeze(1))
                new_target.append(target[mask].add(-self.cutoff[i]))
            else:
                target_idxs.append(None)
                new_target.append(None)

        return new_target, target_idxs

    def forward(self, input, target):
        """
        Args:
            input: (b x t x d)
            target: (b x t)
        Returns:
            2 lists: output for each cutoff section and new targets by cut off
        """

        input = input.contiguous().view(-1, input.size(-1))
        input = self.dropout_module(input)

        new_target, target_idxs = self.adapt_target(target)
        output = [self.head(input)]

        for i in range(len(target_idxs)):
            if target_idxs[i] is not None:
                output.append(self.tail[i](input.index_select(0, target_idxs[i])))
            else:
                output.append(None)

        return output, new_target

    def get_log_prob(self, input, target):
        """
        Computes the log probabilities for all the words of the vocabulary,
        given a 2D tensor of hidden vectors.
        """

        bsz, length, dim = input.size()
        input = input.contiguous().view(-1, dim)

        if target is not None:
            _, target_idxs = self.adapt_target(target)
        else:
            target_idxs = None

        head_y = self.head(input)
        log_probs = head_y.new_zeros(input.size(0), self.vocab_size)

        head_sz = self.cutoff[0] + len(self.tail)
        log_probs[:, :head_sz] = self.lsm(head_y)
        tail_priors = log_probs[:, self.cutoff[0] : head_sz].clone()

        for i in range(len(self.tail)):
            start = self.cutoff[i]
            end = self.cutoff[i + 1]

            if target_idxs is None:
                tail_out = log_probs[:, start:end]
                tail_out.copy_(self.tail[i](input))
                log_probs[:, start:end] = self.lsm(tail_out).add_(
                    tail_priors[:, i, None]
                )
            elif target_idxs[i] is not None:
                idxs = target_idxs[i]
                tail_out = log_probs[idxs, start:end]
                tail_out.copy_(self.tail[i](input[idxs]))
                log_probs[idxs, start:end] = self.lsm(tail_out).add_(
                    tail_priors[idxs, i, None]
                )

        log_probs = log_probs.view(bsz, length, -1)
        return log_probs


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        # bspair = torch.onnx.operators.shape_as_tensor(input)
        bspair = torch._shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False
        if self.padding_idx is not None:
            self.max_positions = self.num_embeddings - self.padding_idx - 1
        else:
            self.max_positions = self.num_embeddings

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        assert (positions is None) or (
            self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                # positions is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                positions = torch.zeros(
                    (1, 1), device=input.device, dtype=input.dtype
                ).fill_(int(self.padding_idx + input.size(1)))
            else:
                positions = make_positions(
                    input, self.padding_idx, onnx_trace=self.onnx_trace
                )
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class BaseLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.num_workers = 1  # distributed_utils.get_data_parallel_world_size()
        expert_centroids = torch.empty(self.num_workers, args.decoder_embed_dim)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        self.register_parameter("expert_centroids", torch.nn.Parameter(expert_centroids))
        self.expert_network = nn.Sequential(*([BaseSublayer(args) for _ in range(args.base_sublayers)]))
        self.expert_id = 0  # distributed_utils.get_data_parallel_rank()
        self.shuffle = args.base_shuffle
        self.cpp = self.load_assignment()

        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        for param in self.expert_network.parameters():
            param.expert = True

    def forward(self, input_features, *args, **kwargs):
        features = input_features.reshape(-1, input_features.size(-1))
        is_training = input_features.requires_grad

        if self.shuffle and is_training:
            # Send each token to a random worker, to break correlations within the batch
            shuffle_sort = torch.randperm(features.size(0), device=features.device)
            features = All2All.apply(features[shuffle_sort])

        with torch.no_grad():
            # Compute similarity of each token to each expert, for routing
            token_expert_affinities = features.matmul(self.expert_centroids.transpose(0, 1))

        # Compute which token goes to which expert
        sort_by_expert, input_splits, output_splits = self.balanced_assignment(token_expert_affinities) \
            if is_training else self.greedy_assignment(token_expert_affinities)
        # Swap these tokens for the right ones for our expert
        routed_features = All2All.apply(features[sort_by_expert], output_splits, input_splits)

        if routed_features.size(0) > 0:
            # Mix in the expert network based on how appropriate it is for these tokens
            alpha = torch.sigmoid(routed_features.mv(self.expert_centroids[self.expert_id])).unsqueeze(1)
            routed_features = alpha * self.expert_network(routed_features) + (1 - alpha) * routed_features
        # Return to original worker and ordering
        result = All2All.apply(routed_features, input_splits, output_splits)[self.inverse_sort(sort_by_expert)]

        if self.shuffle and is_training:
            # Undo shuffling
            result = All2All.apply(result)[self.inverse_sort(shuffle_sort)]

        # Return additional Nones for compatibility with TransformerDecoderLayer
        return result.view(input_features.size()), None, None

    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

    def balanced_assignment(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return self.cpp.balanced_assignment(scores), None, None

    # Assigns each token to the top k experts
    def greedy_assignment(self, scores, k=1):
        token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
        token_to_workers, sort_ordering = torch.sort(token_to_workers)
        worker2token = sort_ordering // k

        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        output_splits = torch.zeros((self.num_workers,), dtype=torch.long, device=scores.device)
        workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        output_splits[workers] = counts
        # Tell other workers how many tokens to expect from us
        input_splits = All2All.apply(output_splits)
        return worker2token, input_splits.tolist(), output_splits.tolist()

    def load_assignment(self):
        try:
            from fairseq import libbase

            return libbase

        except ImportError as e:
            sys.stderr.write(
                "ERROR: missing libbase. run `python setup.py build_ext --inplace`\n"
            )
            raise e


class TiedLinear(nn.Module):
    def __init__(self, weight, transpose):
        super().__init__()
        self.weight = weight
        self.transpose = transpose

    def forward(self, input):
        return F.linear(input, self.weight.t() if self.transpose else self.weight)


class TiedHeadModule(nn.Module):
    def __init__(self, weights, input_dim, num_classes, q_noise, qn_block_size):
        super().__init__()
        tied_emb, _ = weights
        self.num_words, emb_dim = tied_emb.size()

        self.word_proj = quant_noise(
            TiedLinear(tied_emb, transpose=False), q_noise, qn_block_size
        )
        if input_dim != emb_dim:
            self.word_proj = nn.Sequential(
                quant_noise(
                    nn.Linear(input_dim, emb_dim, bias=False), q_noise, qn_block_size
                ),
                self.word_proj,
            )

        self.class_proj = quant_noise(
            nn.Linear(input_dim, num_classes, bias=False), q_noise, qn_block_size
        )
        self.out_dim = self.num_words + num_classes

        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    def forward(self, input):
        inp_sz = functools.reduce(operator.mul, input.shape[:-1], 1)
        out = self._float_tensor.new(inp_sz, self.out_dim)
        out[:, : self.num_words] = self.word_proj(input.view(inp_sz, -1))
        out[:, self.num_words :] = self.class_proj(input.view(inp_sz, -1))
        return out


class BaseSublayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.activation_fn = get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        self.norm = LayerNorm(args.decoder_embed_dim, export=False)
        self.ff1 = torch.nn.Linear(args.decoder_embed_dim, args.decoder_ffn_embed_dim)
        self.ff2 = torch.nn.Linear(args.decoder_ffn_embed_dim, args.decoder_embed_dim)
        self.ff2.weight.data.zero_()

    def forward(self, xs):
        return xs + self.ff2(self.activation_fn(self.ff1(self.norm(xs))))


class All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, input_splits=None, output_splits=None):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        ys = torch.empty_like(xs) if output_splits is None else \
            xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        torch.distributed.all_to_all_single(ys, xs, output_split_sizes=output_splits, input_split_sizes=input_splits)
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        result = torch.empty_like(grad_output) if ctx.input_splits is None else \
            grad_output.new_empty(size=[sum(ctx.input_splits)] + list(grad_output.size()[1:]))
        torch.distributed.all_to_all_single(result, grad_output,
                                            output_split_sizes=ctx.input_splits, input_split_sizes=ctx.output_splits)
        return result, None, None


def checkpoint_wrapper(m, offload_to_cpu=False):
    """
    A friendlier wrapper for performing activation checkpointing.

    Compared to the PyTorch version, this version:
    - wraps an nn.Module, so that all subsequent calls will use checkpointing
    - handles keyword arguments in the forward
    - handles non-Tensor outputs from the forward

    Usage::

        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))
    """
    # should I check whether original_forward has already been set?
    assert not hasattr(
        m, "precheckpoint_forward"
    ), "checkpoint function has already been applied?"
    m.precheckpoint_forward = m.forward
    m.forward = functools.partial(
        _checkpointed_forward,
        m.precheckpoint_forward,  # original_forward
        offload_to_cpu,
    )
    return m


def _checkpointed_forward(original_forward, offload_to_cpu, *args, **kwargs):
    # Autograd Functions in PyTorch work best with positional args, since
    # the backward must return gradients (or None) for every input argument.
    # We can flatten keyword arguments to make this easier.
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    parent_ctx_dict = {"offload": offload_to_cpu}
    output = CheckpointFunction.apply(
        original_forward, parent_ctx_dict, kwarg_keys, *flat_args
    )
    if isinstance(output, torch.Tensor):
        return output
    else:
        packed_non_tensor_outputs = parent_ctx_dict["packed_non_tensor_outputs"]
        if packed_non_tensor_outputs:
            output = unpack_non_tensors(output, packed_non_tensor_outputs)
        return output


def pack_kwargs(*args, **kwargs) -> Tuple[List[str], List[Any]]:
    """
    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == [1, 2]
        assert kwargs == {"a": 3, "b": 4}
    """
    kwarg_keys = []
    flat_args = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)
    return kwarg_keys, flat_args


def unpack_non_tensors(
    tensors: Tuple[torch.Tensor],
    packed_non_tensors: Dict[str, List[Any]],
) -> Tuple[Any]:
    if packed_non_tensors is None:
        return tensors
    assert isinstance(packed_non_tensors, dict)
    mixed = []
    is_tensor_list = packed_non_tensors["is_tensor"]
    objects = packed_non_tensors["objects"]
    assert len(tensors) + len(objects) == len(is_tensor_list)
    obj_i = tnsr_i = 0
    for is_tensor in is_tensor_list:
        if is_tensor:
            mixed.append(tensors[tnsr_i])
            tnsr_i += 1
        else:
            mixed.append(objects[obj_i])
            obj_i += 1
    return tuple(mixed)


class CheckpointFunction(torch.autograd.Function):
    """Similar to the torch version, but support non-Tensor outputs.

    The caller is expected to provide a dict (*parent_ctx_dict*) that will hold
    the non-Tensor outputs. These should be combined with the Tensor *outputs*
    by calling ``unpack_non_tensors``.
    """

    @staticmethod
    def forward(ctx, run_function, parent_ctx_dict, kwarg_keys, *args):
        if torch.is_grad_enabled():  # grad may be disabled, e.g., during validation
            checkpoint.check_backward_validity(args)

        ctx.run_function = run_function
        ctx.kwarg_keys = kwarg_keys
        ctx.fwd_rng_state = get_rng_state()

        tensor_inputs, packed_non_tensor_inputs = split_non_tensors(args)
        if parent_ctx_dict["offload"]:
            ctx.fwd_device = tuple(x.device for x in tensor_inputs)
            ctx.grad_requirements = tuple(x.requires_grad for x in tensor_inputs)
            tensor_inputs = tuple(x.cpu() for x in tensor_inputs)

        else:
            ctx.fwd_device, ctx.grad_requirements = None, None

        ctx.save_for_backward(*tensor_inputs)
        ctx.packed_non_tensor_inputs = packed_non_tensor_inputs

        with torch.no_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(kwarg_keys, args)
            outputs = run_function(*unpacked_args, **unpacked_kwargs)

        if isinstance(outputs, torch.Tensor):
            return outputs
        else:
            # Autograd Functions don't like non-Tensor outputs. We can split the
            # non-Tensor and Tensor outputs, returning the former by reference
            # through *parent_ctx_dict* and returning the latter directly.
            outputs, packed_non_tensor_outputs = split_non_tensors(outputs)
            parent_ctx_dict["packed_non_tensor_outputs"] = packed_non_tensor_outputs
            return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )

        tensor_inputs: Tuple = ctx.saved_tensors
        tensor_inputs = checkpoint.detach_variable(tensor_inputs)
        if ctx.fwd_device is not None:
            tensor_inputs = [
                t.to(ctx.fwd_device[i]) for i, t in enumerate(tensor_inputs)
            ]
            for i, need_grad in enumerate(ctx.grad_requirements):
                tensor_inputs[i].requires_grad = need_grad
        inputs = unpack_non_tensors(tensor_inputs, ctx.packed_non_tensor_inputs)

        # Store the current states.
        bwd_rng_state = get_rng_state()

        # Set the states to what it used to be before the forward pass.
        set_rng_state(ctx.fwd_rng_state)

        with torch.enable_grad():
            unpacked_args, unpacked_kwargs = unpack_kwargs(ctx.kwarg_keys, inputs)
            outputs = ctx.run_function(*unpacked_args, **unpacked_kwargs)
            tensor_outputs, _ = split_non_tensors(outputs)
        # Set the states back to what it was at the start of this function.
        set_rng_state(bwd_rng_state)

        # Run backward() with only Tensors that require grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(tensor_outputs)):
            if tensor_outputs[i].requires_grad:
                outputs_with_grad.append(tensor_outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "None of the outputs have requires_grad=True, "
                "this checkpoint() is not necessary"
            )

        torch.autograd.backward(outputs_with_grad, args_with_grad)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs
        )
        return (None, None, None) + grads


def unpack_kwargs(
    kwarg_keys: List[str], flat_args: List[Any]
) -> Tuple[List[Any], Dict[str, Any]]:
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = {k: v for k, v in zip(kwarg_keys, flat_args[-len(kwarg_keys) :])}
    return args, kwargs


def split_non_tensors(
    mixed: Union[torch.Tensor, Tuple[Any]]
) -> Tuple[Tuple[torch.Tensor], Dict[str, List[Any]]]:
    """
    Usage::

        x = torch.Tensor([1])
        y = torch.Tensor([2])
        tensors, packed_non_tensors = split_non_tensors((x, y, None, 3))
        recon = unpack_non_tensors(tensors, packed_non_tensors)
        assert recon == (x, y, None, 3)
    """
    if isinstance(mixed, torch.Tensor):
        return (mixed,), None
    tensors = []
    packed_non_tensors = {"is_tensor": [], "objects": []}
    for o in mixed:
        if isinstance(o, torch.Tensor):
            packed_non_tensors["is_tensor"].append(True)
            tensors.append(o)
        else:
            packed_non_tensors["is_tensor"].append(False)
            packed_non_tensors["objects"].append(o)
    return tuple(tensors), packed_non_tensors


class FileContentsAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(FileContentsAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from streaming_transformer.all import PathManager

        if PathManager.isfile(values):
            with PathManager.open(values) as f:
                argument = f.read().strip()
        else:
            argument = values
        setattr(namespace, self.dest, argument)


def split_paths(paths: str) -> List[str]:
    return (
        paths.split(os.pathsep)
        if "://" not in paths
        else paths.split(MANIFOLD_PATH_SEP)
    )


def load_ensemble_for_inference(filenames, task, model_arg_overrides=None):
    from fairseq import checkpoint_utils

    deprecation_warning(
        "utils.load_ensemble_for_inference is deprecated. "
        "Please use checkpoint_utils.load_model_ensemble instead."
    )
    return checkpoint_utils.load_model_ensemble(
        filenames, arg_overrides=model_arg_overrides, task=task
    )


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)


def move_to_cpu(sample):

    def _move_to_cpu(tensor):
        # PyTorch has poor support for half tensors (float16) on CPU.
        # Move any such tensors to float32.
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)



def get_incremental_state(
    module: MultiheadAttention,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    key: str,
) -> Optional[Dict[str, Optional[Tensor]]]:
    """Helper for getting incremental state for an nn.Module."""
    return module.get_incremental_state(incremental_state, key)


def set_incremental_state(
    module: MultiheadAttention,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    key: str,
    value: Dict[str, Optional[Tensor]],
) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        result = module.set_incremental_state(incremental_state, key, value)
        if result is not None:
            incremental_state = result
    return incremental_state


def load_align_dict(replace_unk):
    if replace_unk is None:
        align_dict = None
    elif isinstance(replace_unk, str) and len(replace_unk) > 0:
        # Load alignment dictionary for unknown word replacement if it was passed as an argument.
        align_dict = {}
        with open(replace_unk, "r") as f:
            for line in f:
                cols = line.split()
                align_dict[cols[0]] = cols[1]
    else:
        # No alignment dictionary provided but we still want to perform unknown word replacement by copying the
        # original source word.
        align_dict = {}
    return align_dict


def print_embed_overlap(embed_dict, vocab_dict):
    embed_keys = set(embed_dict.keys())
    vocab_keys = set(vocab_dict.symbols)
    overlap = len(embed_keys & vocab_keys)
    logger.info("found {}/{} types in embedding file".format(overlap, len(vocab_dict)))


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]]
            )
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding


def replace_unk(hypo_str, src_str, alignment, align_dict, unk):
    from fairseq import tokenizer

    # Tokens are strings here
    hypo_tokens = tokenize_line(hypo_str)
    # TODO: Very rare cases where the replacement is '<eos>' should be handled gracefully
    src_tokens = tokenize_line(src_str) + ["<eos>"]
    for i, ht in enumerate(hypo_tokens):
        if ht == unk:
            src_token = src_tokens[alignment[i]]
            # Either take the corresponding value in the aligned dictionary or just copy the original value.
            hypo_tokens[i] = align_dict.get(src_token, src_token)
    return " ".join(hypo_tokens)


def post_process_prediction(
    hypo_tokens,
    src_str,
    alignment,
    align_dict,
    tgt_dict,
    remove_bpe=None,
    extra_symbols_to_ignore=None,
):
    hypo_str = tgt_dict.string(
        hypo_tokens, remove_bpe, extra_symbols_to_ignore=extra_symbols_to_ignore
    )
    if align_dict is not None:
        hypo_str = replace_unk(
            hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string()
        )
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
    return hypo_tokens, hypo_str, alignment


def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def convert_padding_direction(
    src_tokens, padding_idx, right_to_left: bool = False, left_to_right: bool = False
):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    buffered = torch.empty(0).long()
    if max_len > 0:
        torch.arange(max_len, out=buffered)
    range = buffered.type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


def item(tensor):
    # tpu-comment: making this a no-op for xla devices.
    if torch.is_tensor(tensor) and tensor.device.type == 'xla':
        return tensor.detach()
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor


def multi_tensor_total_norm(grads, chunk_size=2048 * 32) -> torch.Tensor:
    per_device_grads = {}
    norms = []
    for grad in grads:
        device = grad.device
        cur_device_grads = per_device_grads.get(device)
        if cur_device_grads is None:
            cur_device_grads = []
            per_device_grads[device] = cur_device_grads
        cur_device_grads.append(grad)
    for device in per_device_grads.keys():
        cur_device_grads = per_device_grads[device]
        if device.type == "cuda":
            # TODO(msb) return has_inf
            has_inf = torch.zeros((1, 1), dtype=torch.int, device=device)
            with torch.cuda.device(device):
                norm = multi_tensor_l2norm(
                    chunk_size, has_inf, [cur_device_grads], False
                )
            norms.append(norm[0].to(torch.cuda.current_device()))
        else:
            norms += [torch.norm(g, p=2, dtype=torch.float32) for g in cur_device_grads]
    total_norm = torch.norm(torch.stack(norms))
    return total_norm


@torch.no_grad()
def clip_grad_norm_(params, max_norm, aggregate_norm_fn=None) -> torch.Tensor:
    def grad_exists(p):
        return p is not None and getattr(p, "grad", None) is not None
    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    grads = [p.grad.detach() for p in params if grad_exists(p) and not hasattr(p, 'expert')]
    expert_grads = [p.grad.detach() for p in params if grad_exists(p) and hasattr(p, 'expert')]

    if len(grads) == 0:
        if len(params) > 0:
            return params[0].new_tensor(0.0)
        else:
            return torch.tensor(0.0)

    if len(grads) == 1:
        total_norm = torch.norm(grads[0], p=2, dtype=torch.float32)
    else:
        if multi_tensor_l2norm_available:
            total_norm = multi_tensor_total_norm(grads)
        else:
            if torch.cuda.is_available():
                warnings.warn(
                    "amp_C fused kernels unavailable, disabling multi_tensor_l2norm; "
                    "you may get better performance by installing NVIDIA's apex library"
                )
                device = torch.cuda.current_device()
            elif grads[0].device.type == "xla":
                device = grads[0].device
            else:
                device = torch.device("cpu")
            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(g, p=2, dtype=torch.float32).to(device) for g in grads]
                )
            )

    if aggregate_norm_fn is not None:
        total_norm = aggregate_norm_fn(total_norm)

    if max_norm > 0:
        max_norm = float(max_norm)
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp_(max=1)
        for g in grads + expert_grads:
            g.mul_(clip_coef)
    return total_norm


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _match_types(arg1, arg2):
    """Convert the numerical argument to the same type as the other argument"""

    def upgrade(arg_number, arg_structure):
        if isinstance(arg_structure, tuple):
            return tuple([arg_number] * len(arg_structure))
        elif isinstance(arg_structure, dict):
            arg = copy.deepcopy(arg_structure)
            for k in arg:
                arg[k] = upgrade(arg_number, arg_structure[k])
            return arg
        else:
            return arg_number

    if isinstance(arg1, float) or isinstance(arg1, int):
        return upgrade(arg1, arg2), arg2
    elif isinstance(arg2, float) or isinstance(arg2, int):
        return arg1, upgrade(arg2, arg1)

    return arg1, arg2


def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def map_value_update(d1, d2):
        updated_value = copy.deepcopy(d1)
        for key in d2:
            if key not in updated_value:
                updated_value[key] = d2[key]
            else:
                updated_value[key] = min(d1[key], d2[key])
        return updated_value

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            max_positions, arg = _match_types(max_positions, arg)
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            elif isinstance(arg, dict):
                max_positions = map_value_update(max_positions, arg)
            else:
                max_positions = tuple(map(nullsafe_min, zip(max_positions, arg)))

    return max_positions


def import_user_module(args):
    module_path = getattr(args, "user_dir", None)
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        if not os.path.exists(module_path) and not os.path.isfile(os.path.dirname(module_path)):
            fairseq_rel_path = os.path.join(os.path.dirname(__file__), args.user_dir)
            if os.path.exists(fairseq_rel_path):
                module_path = fairseq_rel_path
            else:
                fairseq_rel_path = os.path.join(
                    os.path.dirname(__file__), "..", args.user_dir
                )
                if os.path.exists(fairseq_rel_path):
                    module_path = fairseq_rel_path
                else:
                    raise FileNotFoundError(module_path)

        # ensure that user modules are only imported once
        import_user_module.memo = getattr(import_user_module, "memo", set())
        if module_path not in import_user_module.memo:
            import_user_module.memo.add(module_path)

            module_parent, module_name = os.path.split(module_path)
            if module_name not in sys.modules:
                sys.path.insert(0, module_parent)
                importlib.import_module(module_name)
            else:
                raise ImportError(
                    "Failed to import --user-dir={} because the corresponding module name "
                    "({}) is not globally unique. Please rename the directory to "
                    "something unique and try again.".format(module_path, module_name)
                )


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def log_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)


def get_perplexity(loss, round=2, base=2):
    from fairseq.logging.meters import safe_round

    if loss is None:
        return 0.0
    try:
        return safe_round(base ** loss, round)
    except OverflowError:
        return float("inf")


def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        deprecation_warning(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


@contextlib.contextmanager
def model_eval(model):
    is_training = model.training
    model.eval()
    yield
    model.train(is_training)


def has_parameters(module):
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if xm is not None:
        state["xla_rng_state"] = xm.get_rng_state()
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if xm is not None:
        xm.set_rng_state(state["xla_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


class set_torch_seed(object):
    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = get_rng_state()

        torch.manual_seed(seed)
        if xm is not None:
            xm.set_rng_state(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        set_rng_state(self.rng_state)


def parse_alignment(line):
    """
    Parses a single line from the alingment file.

    Args:
        line (str): String containing the alignment of the format:
            <src_idx_1>-<tgt_idx_1> <src_idx_2>-<tgt_idx_2> ..
            <src_idx_m>-<tgt_idx_m>. All indices are 0 indexed.

    Returns:
        torch.IntTensor: packed alignments of shape (2 * m).
    """
    alignments = line.strip().split()
    parsed_alignment = torch.IntTensor(2 * len(alignments))
    for idx, alignment in enumerate(alignments):
        src_idx, tgt_idx = alignment.split("-")
        parsed_alignment[2 * idx] = int(src_idx)
        parsed_alignment[2 * idx + 1] = int(tgt_idx)
    return parsed_alignment


def get_token_to_word_mapping(tokens, exclude_list):
    n = len(tokens)
    word_start = [int(token not in exclude_list) for token in tokens]
    word_idx = list(accumulate(word_start))
    token_to_word = {i: word_idx[i] for i in range(n)}
    return token_to_word


def extract_hard_alignment(attn, src_sent, tgt_sent, pad, eos):
    tgt_valid = (
        ((tgt_sent != pad) & (tgt_sent != eos)).nonzero(as_tuple=False).squeeze(dim=-1)
    )
    src_invalid = (
        ((src_sent == pad) | (src_sent == eos)).nonzero(as_tuple=False).squeeze(dim=-1)
    )
    src_token_to_word = get_token_to_word_mapping(src_sent, [eos, pad])
    tgt_token_to_word = get_token_to_word_mapping(tgt_sent, [eos, pad])
    alignment = []
    if len(tgt_valid) != 0 and len(src_invalid) < len(src_sent):
        attn_valid = attn[tgt_valid]
        attn_valid[:, src_invalid] = float("-inf")
        _, src_indices = attn_valid.max(dim=1)
        for tgt_idx, src_idx in zip(tgt_valid, src_indices):
            alignment.append(
                (
                    src_token_to_word[src_idx.item()] - 1,
                    tgt_token_to_word[tgt_idx.item()] - 1,
                )
            )
    return alignment


def extract_soft_alignment(attn, src_sent, tgt_sent, pad, eos):
    tgt_valid = (
        ((tgt_sent != pad)).nonzero(as_tuple=False)
    )
    src_valid = (
        ((src_sent != pad)).nonzero(as_tuple=False).squeeze(dim=-1)
    )
    alignment = []
    if len(tgt_valid) != 0 and len(src_valid) != 0:
        attn_valid = attn[tgt_valid, src_valid]
        alignment = [
            ["{:.6f}".format(p) for p in src_probs.tolist()]
            for src_probs in attn_valid
        ]
    return alignment


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def get_tpu_device():
    return xm.xla_device()




def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == 'xla'


def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor


class CudaEnvironment(object):
    def __init__(self):
        cur_device = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties("cuda:{}".format(cur_device))
        self.name = prop.name
        self.major = prop.major
        self.minor = prop.minor
        self.total_memory_in_GB = prop.total_memory / 1024 / 1024 / 1024

    @staticmethod
    def pretty_print_cuda_env_list(cuda_env_list):
        """
        Given a list of CudaEnviorments, pretty print them
        """
        num_workers = len(cuda_env_list)
        center = "CUDA enviroments for all {} workers".format(num_workers)
        banner_len = 40 - len(center) // 2
        first_line = "*" * banner_len + center + "*" * banner_len
        logger.info(first_line)
        for r, env in enumerate(cuda_env_list):
            logger.info(
                "rank {:3d}: ".format(r)
                + "capabilities = {:2d}.{:<2d} ; ".format(env.major, env.minor)
                + "total memory = {:.3f} GB ; ".format(env.total_memory_in_GB)
                + "name = {:40s}".format(env.name)
            )
        logger.info(first_line)


def csv_str_list(x):
    return x.split(",")


def eval_str_list_secons(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class Dictionary:
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        *,  # begin keyword-only arguments
        bos="<s>",
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        extra_special_symbols=None,
    ):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(
        self,
        tensor,
        bpe_symbol=None,
        escape_unk=False,
        extra_symbols_to_ignore=None,
        unk_string=None,
        include_eos=False,
    ):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(t, bpe_symbol, escape_unk, extra_symbols_to_ignore, include_eos=include_eos)
                for t in tensor
            )

        extra_symbols_to_ignore = set(extra_symbols_to_ignore or [])
        extra_symbols_to_ignore.add(self.eos())

        def token_string(i):
            if i == self.unk():
                if unk_string is not None:
                    return unk_string
                else:
                    return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, "bos_index"):
            extra_symbols_to_ignore.add(self.bos())

        sent = " ".join(
            token_string(i)
            for i in tensor
            if item(i) not in extra_symbols_to_ignore
        )

        return post_process(sent, bpe_symbol)

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1, overwrite=False):
        """Adds a word to the dictionary"""
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = Counter(
            dict(
                sorted(zip(self.symbols[self.nspecial :], self.count[self.nspecial :]))
            )
        )
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

        self.pad_to_multiple_(padding_factor)

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        if padding_factor > 1:
            i = 0
            while len(self) % padding_factor != 0:
                symbol = "madeupword{:04d}".format(i)
                self.add_symbol(symbol, n=0)
                i += 1

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(PathManager.get_local_path(f), "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        lines = f.readlines()
        indices_start_line = self._load_meta(lines)

        for line in lines[indices_start_line:]:
            try:
                line, field = line.rstrip().rsplit(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        "Duplicate word found when loading Dictionary: '{}'. "
                        "Duplicate words can overwrite earlier ones by adding the "
                        "#fairseq:overwrite flag at the end of the corresponding row "
                        "in the dictionary file. If using the Camembert model, please "
                        "download an updated copy of the model file.".format(word)
                    )
                self.add_symbol(word, n=count, overwrite=overwrite)
            except ValueError:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt> [flags]'"
                )

    def _save(self, f, kv_iterator):
        if isinstance(f, str):
            PathManager.mkdirs(os.path.dirname(f))
            with PathManager.open(f, "w", encoding="utf-8") as fd:
                return self.save(fd)
        for k, v in kv_iterator:
            print("{} {}".format(k, v), file=f)

    def _get_meta(self):
        return [], []

    def _load_meta(self, lines):
        return 0

    def save(self, f):
        """Stores dictionary into a text file"""
        ex_keys, ex_vals = self._get_meta()
        self._save(
            f,
            zip(
                ex_keys + self.symbols[self.nspecial :],
                ex_vals + self.count[self.nspecial :],
            ),
        )

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ) -> torch.IntTensor:
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    @staticmethod
    def _add_file_to_dictionary_single_worker(
        filename, tokenize, eos_word, worker_id=0, num_workers=1
    ):
        counter = Counter()
        with open(PathManager.get_local_path(filename), "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                # f.tell() returns only an opaque number which can
                # return to the position in the file via f.seek()
                # and does not necessarily represent a byte position
                # in the file. However, f.tell() is faithful to the
                # byte position _most of the time_. Thus we can just
                # check against the file size to prevent early exit.
                if f.tell() > end and f.tell() < size:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            for w, c in sorted(counter.items()):
                dict.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(
                    pool.apply_async(
                        Dictionary._add_file_to_dictionary_single_worker,
                        (filename, tokenize, dict.eos_word, worker_id, num_workers),
                    )
                )
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                Dictionary._add_file_to_dictionary_single_worker(
                    filename, tokenize, dict.eos_word
                )
            )

def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol in {"subword_nmt", "@@ ", "@@"}:
        if symbol == "subword_nmt":
            symbol = "@@ "
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    elif symbol == "none":
        pass
    elif symbol is not None:
        raise NotImplementedError(f"Unknown post_process option: {symbol}")
    return sentence


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)