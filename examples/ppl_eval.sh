# Llama-2-7B
CUDA_VISIBLE_DEVICES=0 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path meta-llama/Llama-2-7b-hf \
    --act_scales_path act_scales/llama-2-7b.pt \
    --smooth \
    --quantize

# Llama-2-13B
CUDA_VISIBLE_DEVICES=0 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path meta-llama/Llama-2-13b-hf \
    --act_scales_path act_scales/llama-2-13b.pt \
    --smooth \
    --quantize

# Llama-2-70B
CUDA_VISIBLE_DEVICES=0,1,2 python smoothquant/ppl_eval.py \
    --alpha 0.9 \
    --model_path meta-llama/Llama-2-70b-hf \
    --act_scales_path act_scales/llama-2-70b.pt \
    --smooth \
    --quantize

# Mistral-7B
CUDA_VISIBLE_DEVICES=0 python smoothquant/ppl_eval.py \
    --alpha 0.8 \
    --model_path mistralai/Mistral-7B-v0.1 \
    --act_scales_path act_scales/Mistral-7B-v0.1.pt \
    --smooth \
    --quantize

# Mixtral-8x7B
CUDA_VISIBLE_DEVICES=0,1 python smoothquant/ppl_eval.py \
    --alpha 0.8 \
    --model_path mistralai/Mixtral-8x7B-v0.1 \
    --act_scales_path act_scales/Mixtral-8x7B-v0.1.pt \
    --smooth \
    --quantize

# Falcon-7B
CUDA_VISIBLE_DEVICES=0 python smoothquant/ppl_eval.py \
    --alpha 0.6 \
    --model_path tiiuae/falcon-7b \
    --act_scales_path act_scales/falcon-7b.pt \
    --smooth \
    --quantize

# Falcon-40B
CUDA_VISIBLE_DEVICES=0,1 python smoothquant/ppl_eval.py \
    --alpha 0.7 \
    --model_path tiiuae/falcon-40b \
    --act_scales_path act_scales/falcon-40b.pt \
    --smooth \
    --quantize

# Meta-Llama-3-8B
CUDA_VISIBLE_DEVICES=0 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path meta-llama/Meta-Llama-3-8B \
    --act_scales_path act_scales/Meta-Llama-3-8B.pt \
    --smooth \
    --quantize

# Meta-Llama-3-70B
CUDA_VISIBLE_DEVICES=0,1 python smoothquant/ppl_eval.py \
    --alpha 0.85 \
    --model_path meta-llama/Meta-Llama-3-70B \
    --act_scales_path act_scales/Meta-Llama-3-70B.pt \
    --smooth \
    --quantize
