import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Dict, Tuple, List, NamedTuple, Optional
from layers import (AugmentedMemoryTransformerEncoderLayer,
                    TransformerEncoderLayer,
                    LayerNorm,
                    PositionalEmbedding,
                    MultiheadAttention)
from utils import (lengths_to_padding_mask,
                   sequence_to_segments,
                   segments_to_sequence,
                   lengths_to_encoder_padding_mask,
                   elu_feature_map)


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
                                          self.left_context_after_stride: -self.right_context_after_stride
                                          ]

        lengths = (
            (
                ~encoder_padding_mask[
                 :, self.left_context_after_stride: -self.right_context_after_stride
                 ]
            )
                .sum(dim=1, keepdim=True)
                .long()
        )

        return states[-1]["encoder_states"], lengths, states


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
    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0 or \
                encoder_out["encoder_padding_mask"][0] is None:
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
        encoder_states = encoder_out["encoder_states"][0]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = {"memory_banks": [s.index_select(1, new_order) for s in state["memory_banks"]],
                                       "encoder_states": state["encoder_states"].index_select(1, new_order)}

        return {
            "encoder_out": new_encoder_out,
            "encoder_padding_mask": new_encoder_padding_mask,
            "encoder_embedding": new_encoder_embedding,
            "encoder_states": [encoder_states],
            "src_tokens": [],
            "src_lengths": [],
        }