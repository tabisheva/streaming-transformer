import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any
from torch import Tensor
import torch.nn.functional as F
import math
import sys

from utils import (LayerNorm,
                   FairseqDropout,
                   get_activation_fn,
                   quant_noise,
                   MultiheadAttention,
                   attention_suppression,
                   make_positions,
                   elu_feature_map)


class BaseLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.num_workers = distributed_utils.get_data_parallel_world_size()
        expert_centroids = torch.empty(self.num_workers, args.decoder_embed_dim)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        self.register_parameter("expert_centroids", torch.nn.Parameter(expert_centroids))
        self.expert_network = nn.Sequential(*([BaseSublayer(args) for _ in range(args.base_sublayers)]))
        self.expert_id = distributed_utils.get_data_parallel_rank()
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
        module = AugmentedMemoryMultiheadLinearAttention if args.linear else AugmentedMemoryMultiheadAttention
        return module(
            embed_dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            tanh_on_mem=True,
            max_memory_size=args.max_memory_size,
        )


# ------------------------------------------------------------------------------
#   AugmentedMemoryMultiheadAttention
# ------------------------------------------------------------------------------
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

        attention_weights = torch.nn.functional.softmax(
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


# ------------------------------------------------------------------------------
#   AugmentedMemoryMultiheadLinearAttention
# ------------------------------------------------------------------------------
class AugmentedMemoryMultiheadLinearAttention(MultiheadAttention):
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

        self.feature_map = (
            elu_feature_map(embed_dim)
        )

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
                memory = memory[-self.max_memory_size:]

        memory_and_input = torch.cat(memory + [input_and_summary[:-1]], dim=0)
        input_and_sum_query = input_and_summary

        Q_segment = self.feature_map.forward_queries(input_and_summary[:-1])
        Q_summary = self.feature_map.forward_queries(input_and_summary[-1])

        K_full = self.feature_map.forward_keys(memory_and_input)
        K_segment = self.feature_map.forward_keys(input_and_summary[:-1])
        #        K = K * key_lengths.float_matrix[:, :, None, None]

        KV_segment = torch.einsum("nshd,nshm->nhmd", K_segment, input_and_summary[:-1])
        KV_full = torch.einsum("nshd,nshm->nhmd", K_full, memory_and_input)

        # Compute the normalizer
        Z_full = 1 / (torch.einsum("nlhd,nhd->nlh", Q_segment, KV_full.sum(dim=1)) + self.eps)
        Z_memory = 1 / (torch.einsum("nlhd,nhd->nlh", Q_summary, KV_segment.sum(dim=1)) + self.eps)

        # Finally compute and return the new values
        attention = torch.einsum("nlhd,nhmd,nlh->nlhm", Q_segment, KV_full, Z_full)
        attention_memory = torch.einsum("nlhd,nhmd,nlh->nlhm", Q_summary, KV_segment, Z_memory)

        # return V.contiguous()

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
        bspair = torch.onnx.operators.shape_as_tensor(input)
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