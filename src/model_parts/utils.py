import torch
from typing import Callable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from typing import Dict, Optional, Tuple, List
import logging
import functools
import operator

import math
import warnings

logger = logging.getLogger(__name__)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def item(tensor):
    # tpu-comment: making this a no-op for xla devices.
    if torch.is_tensor(tensor) and tensor.device.type == 'xla':
        return tensor.detach()
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


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


class FeatureMap(nn.Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.
        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        return

    def forward(self, x):
        return self.activation_function(x)


elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)


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


def lengths_to_padding_mask(lengths: Tensor) -> Tensor:
    """Convert lengths of shape (B, ) to padding mask."""
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(  # [0, ..., T-1]
        max_length, device=lengths.device, dtype=lengths.dtype
    ).expand(batch_size, max_length) >= lengths.unsqueeze(1)

    return padding_mask


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def attention_suppression(attention_weights: Tensor, scale: float):
    # B, H, qlen, klen -> B, H, qlen, 1
    attention_prob = torch.nn.functional.softmax(attention_weights.float(), dim=-1)
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


def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a = math.sqrt(2 / math.pi)
    return (
        0.5 * x * (1 + torch.tanh(gelu_accurate._a * (x + 0.044715 * torch.pow(x, 3))))
    )


def gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float()).type_as(x)


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
