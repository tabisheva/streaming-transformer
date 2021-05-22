from argparse import Namespace
from typing import Tuple, Optional, Dict, List

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch import Tensor, nn as nn
from torch.autograd import profiler as profiler
from torch.nn import functional as F

from src.modules.decoder import TransformerDecoder, TransformerDecoderNoExtra, FairseqDecoder
from src.modules.encoder import SequenceEncoder, AugmentedMemoryConvTransformerEncoder, ConvTransformerEncoder, \
    FairseqEncoder
from src.modules.layers import Embedding
from src.modules.utils import base_architecture, check_type, \
    gen_parser_from_dataclass, \
    logger, convert_namespace_to_omegaconf, prune_state_dict, MODEL_REGISTRY, FairseqDataclass, \
    MODEL_DATACLASS_REGISTRY, register_model_architecture
from src.modules.utils import (
    load_pretrained_component_from_model,
)


def augmented_memory(klass):
    class StreamSeq2SeqModel(klass):
        @staticmethod
        def add_args(parser):
            super(StreamSeq2SeqModel, StreamSeq2SeqModel).add_args(parser)
            parser.add_argument(
                "--segment-size", type=int, required=True, help="Length of the segment."
            )
            parser.add_argument(
                "--left-context",
                type=int,
                default=0,
                help="Left context for the segment.",
            )
            parser.add_argument(
                "--right-context",
                type=int,
                default=0,
                help="Right context for the segment.",
            )
            parser.add_argument(
                "--max-memory-size",
                type=int,
                default=-1,
                help="Right context for the segment.",
            )

    StreamSeq2SeqModel.__name__ = klass.__name__
    return StreamSeq2SeqModel


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
            logger.warn(
                "using 'args' is deprecated, please update your code to use dataclass config"
            )
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
        encoder (src.encoder.FairseqEncoder): the encoder
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
        with profiler.record_function("Encoder_out"):
            encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)

        with profiler.record_function("Decoder_out"):
            decoder_out = self.decoder(
                prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
            )
        return decoder_out


@register_model("convtransformer_simul_trans")
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


@register_model("convtransformer_augmented_memory")
@augmented_memory
class AugmentedMemoryConvTransformerModel(SimulConvTransformerModel):
    @classmethod
    def build_encoder(cls, args):
        encoder = SequenceEncoder(args, AugmentedMemoryConvTransformerEncoder(args))

        if getattr(args, "load_pretrained_encoder_from", None) is not None:
            encoder = load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )

        return encoder
