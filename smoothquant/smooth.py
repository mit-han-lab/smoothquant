import torch
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from datasets import load_dataset
import functools
from tqdm import tqdm
from smoothquant.fake_quant import *
import numpy as np


@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat([fc.weight.abs().max(
        dim=0, keepdim=True)[0] for fc in fcs], dim=0)
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)
              ).clamp(min=1e-5).to(device).to(dtype)

    ln.weight.div_(scales)
    ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            layer_name = name + '.self_attn.q_proj'
            qkv_input_scales = scales[layer_name]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha[layer_name] if isinstance(alpha, dict) else alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            layer_name = name + '.fc1'
            fc1_input_scales = scales[layer_name]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha[layer_name] if isinstance(alpha, dict) else alpha)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            layer_name = name + '.self_attention.query_key_value'
            qkv_input_scales = scales[layer_name]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha[layer_name] if isinstance(alpha, dict) else alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            layer_name = name + '.mlp.dense_h_to_4h'
            fc1_input_scales = scales[layer_name]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha[layer_name] if isinstance(alpha, dict) else alpha)


def get_smooth_layer_keys(model):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            return ['self_attn.q_proj', 'fc1']
        elif isinstance(module, BloomBlock):
            return ['self_attention.query_key_value', 'mlp.dense_h_to_4h']


@torch.no_grad()
def auto_smooth_lm(model, tokenizer, act_maxes, dataset_path, num_samples=10, seq_len=512, act_quant="per_token", weight_quant="per_channel"):
    model.eval()
    device = next(model.parameters()).device

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    layer_alphas = {}

    # adapted from llm-awq
    def search_alpha_hook(m, x, y, name):
        ori_y = y
        device = m.weight.device
        dtype = m.weight.dtype

        act_scales = act_maxes[name].to(device).to(dtype)
        weight_scales = m.weight.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5)

        if isinstance(y, tuple):
            ori_y = y[0]
        if isinstance(x, tuple):
            x = x[0]
        
        best_alpha = -1
        best_scales = None
        best_loss = float('inf')
        loss_history = []

        steps = 20
        for s in range(1, steps):
            alpha = s / steps
            scales = (act_scales.pow(alpha) / weight_scales.pow(1-alpha)).clamp(min=1e-5).to(device).to(dtype)
            xs = x / scales
            ws = m.weight * scales.view(1, -1)

            if act_quant == "per_token":
                xfq = quantize_activation_per_token_absmax(xs)
            elif act_quant == "per_tensor":
                xfq = quantize_activation_per_tensor_absmax(xs)
            else:
                raise ValueError("do not support act quant method: " + act_quant)
            
            if weight_quant == "per_channel":
                wfq = quantize_weight_per_channel_absmax(ws)
            elif weight_quant == "per_tensor":
                wfq = quantize_weight_per_tensor_absmax(ws)
            else:
                raise ValueError("do not support weight quant method: " + weight_quant)

            yfq = torch.matmul(xfq, wfq.t_())
            if m.bias is not None:
                yfq = yfq + m.bias

            loss = (ori_y - yfq).float().pow(2).mean().item()
            loss_history.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_scales = scales
                best_alpha = alpha

        # print(name, best_alpha, loss_history)
        if name not in layer_alphas.keys():
            layer_alphas[name] = [best_alpha,]
        else:
            layer_alphas[name].append(best_alpha)
    
    smooth_layer_keys = get_smooth_layer_keys(model) # only search alphas for layers that need smoothing
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and any(c in name for c in smooth_layer_keys):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(search_alpha_hook, name=name))
            )

    for i in tqdm(range(num_samples), "search smooth alpha..."):
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                            max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    for k, v in layer_alphas.items():
        layer_alphas[k] = np.mean(v)

    smooth_lm(model, act_maxes, layer_alphas)

