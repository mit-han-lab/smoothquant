import torch
from torch import nn
from transformers.models.opt.modeling_opt import (
    OPTConfig,
    OPTForCausalLM,
    OPTModel,
    OPTPreTrainedModel,
    OPTLearnedPositionalEmbedding,
    OPTAttention,
    OPTDecoderLayer,
    OPTDecoder,
    BaseModelOutputWithPast,
)
from typing import Optional, Tuple, List
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from torch_int.nn.fused import LayerNormQ
from transformers.utils import logging
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T

logger = logging.get_logger(__name__)


class Int8OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )

        self.attention_weight_scale = 1.0

        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        self.k_proj = W8A8B8O8Linear(embed_dim, embed_dim)
        self.v_proj = W8A8B8O8Linear(embed_dim, embed_dim)
        self.q_proj = W8A8B8O8Linear(embed_dim, embed_dim)
        self.out_proj = W8A8BFP32OFP32Linear(embed_dim, embed_dim)

    @staticmethod
    @torch.no_grad()
    def from_float(
        module: OPTAttention,
        input_scale: float,
        q_output_scale: float,
        k_output_scale: float,
        v_output_scale: float,
        out_input_scale: float,
    ):
        int8_module = Int8OPTAttention(module.embed_dim, module.num_heads)
        # Fuse the scaling into the q_proj output scale
        q_output_scale = q_output_scale * module.scaling
        module.q_proj.weight *= module.scaling
        module.q_proj.bias *= module.scaling
        int8_module.q_proj = W8A8B8O8Linear.from_float(
            module.q_proj, input_scale, q_output_scale
        )
        int8_module.k_proj = W8A8B8O8Linear.from_float(
            module.k_proj, input_scale, k_output_scale
        )
        int8_module.v_proj = W8A8B8O8Linear.from_float(
            module.v_proj, input_scale, v_output_scale
        )
        int8_module.out_proj = W8A8BFP32OFP32Linear.from_float(
            module.out_proj, out_input_scale
        )
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(q_output_scale, k_output_scale)

        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127, v_output_scale, out_input_scale
        )
        return int8_module

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = self.qk_bmm(query_states, key_states)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_probs = layer_head_mask.view(1, -1, 1, 1) * attn_probs.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_probs = attn_probs.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_probs_reshaped = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_probs_reshaped = None

        # (A_row V_row)_row = (A_row V_col ^T)_row
        attn_probs.mul_(127).round_()
        attn_probs = attn_probs.to(torch.int8)

        value_states = value_states.transpose(1, 2).contiguous()
        attn_output = self.pv_bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_probs_reshaped, past_key_value


class Int8OPTDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_attention_heads, ffn_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = Int8OPTAttention(
            embed_dim=self.embed_dim, num_heads=num_attention_heads
        )

        self.self_attn_layer_norm = LayerNormQ(self.embed_dim)
        self.fc1 = W8A8B8O8LinearReLU(self.embed_dim, ffn_dim)
        self.fc2 = W8A8BFP32OFP32Linear(ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNormQ(self.embed_dim)

    @staticmethod
    def from_float(
        module: OPTDecoderLayer,
        attn_input_scale: float,
        q_output_scale: float,
        k_output_scale: float,
        v_output_scale: float,
        out_input_scale: float,
        fc1_input_scale: float,
        fc2_input_scale: float,
    ):
        int8_module = Int8OPTDecoderLayer(
            module.embed_dim, module.self_attn.num_heads, module.fc1.out_features
        )
        int8_module.self_attn_layer_norm = LayerNormQ.from_float(
            module.self_attn_layer_norm, attn_input_scale
        )
        int8_module.self_attn = Int8OPTAttention.from_float(
            module.self_attn,
            attn_input_scale,
            q_output_scale,
            k_output_scale,
            v_output_scale,
            out_input_scale,
        )
        int8_module.final_layer_norm = LayerNormQ.from_float(
            module.final_layer_norm, fc1_input_scale
        )
        int8_module.fc1 = W8A8B8O8LinearReLU.from_float(
            module.fc1, fc1_input_scale, fc2_input_scale
        )
        int8_module.fc2 = W8A8BFP32OFP32Linear.from_float(module.fc2, fc2_input_scale)
        return int8_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.Int8Tensor`): the output of previous layer's layernorm in INT8
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # Self Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        residual.add_(hidden_states.to(residual.dtype))

        hidden_states = self.final_layer_norm(residual)

        hidden_states = self.fc1(hidden_states)

        hidden_states = self.fc2(hidden_states)

        residual.add_(hidden_states.to(residual.dtype))

        outputs = (residual,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Int8OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Int8OPTDecoderLayer`]

    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.word_embed_proj_dim, self.padding_idx
        )
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(
                config.hidden_size, config.word_embed_proj_dim, bias=False
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(
                config.word_embed_proj_dim, config.hidden_size, bias=False
            )
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList(
            [
                Int8OPTDecoderLayer(
                    config.hidden_size, config.num_attention_heads, config.ffn_dim
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = OPTDecoder.get_input_embeddings
    set_input_embeddings = OPTDecoder.set_input_embeddings
    _prepare_decoder_attention_mask = OPTDecoder._prepare_decoder_attention_mask
    old_forward = OPTDecoder.forward

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8OPTDecoder(module.config)
        int8_module.embed_tokens = module.embed_tokens
        int8_module.embed_positions = module.embed_positions
        int8_module.project_out = module.project_out
        int8_module.final_layer_norm = module.final_layer_norm
        for i, layer in enumerate(module.layers):
            int8_module.layers[i] = Int8OPTDecoderLayer.from_float(
                layer, **decoder_layer_scales[i]
            )
        return int8_module

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        # pad the input to the multiple of 16
        input_len = input_ids.shape[1]
        from torch.nn.functional import pad

        if input_len % 16 != 0:
            # <pad> is 1
            padding_len = 16 - input_len % 16
            input_ids = pad(input_ids, (0, padding_len), value=1)
            if attention_mask is not None:
                attention_mask = pad(attention_mask, (0, padding_len), value=0)
        output = self.old_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # slice the output to the original length
        if input_len % 16 != 0:
            output.last_hidden_state = output.last_hidden_state[:, :input_len, :]
        return output


class Int8OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = Int8OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    get_input_embeddings = OPTModel.get_input_embeddings
    set_input_embeddings = OPTModel.set_input_embeddings
    get_decoder = OPTModel.get_decoder
    forward = OPTModel.forward

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8OPTModel(module.config)
        int8_module.decoder = Int8OPTDecoder.from_float(
            module.decoder, decoder_layer_scales
        )
        return int8_module


class Int8OPTForCausalLM(OPTPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Int8OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8OPTForCausalLM(module.config)
        int8_module.model = Int8OPTModel.from_float(module.model, decoder_layer_scales)
        int8_module.lm_head = module.lm_head
        return int8_module

    get_input_embeddings = OPTForCausalLM.get_input_embeddings
    set_input_embeddings = OPTForCausalLM.set_input_embeddings
    get_output_embeddings = OPTForCausalLM.get_output_embeddings
    set_output_embeddings = OPTForCausalLM.set_output_embeddings
    set_decoder = OPTForCausalLM.set_decoder
    get_decoder = OPTForCausalLM.get_decoder
    forward = OPTForCausalLM.forward
    prepare_inputs_for_generation = OPTForCausalLM.prepare_inputs_for_generation
    _reorder_cache = OPTForCausalLM._reorder_cache
