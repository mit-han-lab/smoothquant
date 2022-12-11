# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models [[paper](https://arxiv.org/abs/2211.10438)]

**If you are interested in getting updates, please sign up [here](https://forms.gle/YjYQQas5Hbqge1LH9) to get notified!**

![intuition](figures/intuition.png)

## Abstract

Large language models (LLMs) show excellent performance but are compute- and memory-intensive. Quantization can reduce memory and accelerate inference. However, for LLMs beyond 100 billion parameters, existing methods cannot maintain accuracy or have to rely on techniques that do not run efficiently on hardware. We propose SmoothQuant, a training-free, lightweight, and general-purpose post-training quantization (PTQ) solution to enable lossless 8-bit weight, 8-bit activation (W8A8) quantization for LLMs that can be implemented efficiently. We observe that systematic outliers appear at fixed activation channels. Based on the fact that weights are easy to quantize while activations are not, SmoothQuant smooths the activation outliers by migrating the quantization difficulty from activations to weights with a mathematically equivalent transformation. SmoothQuant enables an INT8 quantization of both weights and activations for all the GEMMs in LLMs, including OPT-175B, BLOOM-176B, and GLM-130B. SmoothQuant has better hardware efficiency than existing techniques using mixed-precision activation quantization or weight-only quantization. We demonstrate up to 1.56x speedup and 2x memory reduction for LLMs with negligible loss in accuracy. Thanks to the hardware-friendly design, we integrate SmoothQuant into FasterTransformer, a state-of-the-art LLM serving framework, and achieve faster inference speed with half the number of GPUs compared to FP16. Our work offers a turn-key solution that reduces hardware costs and democratizes LLMs.


## Installation

```bash
conda create -n smoothquant python=3.8
conda activate smoothquant
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers accelerate datasets

python setup.py install
```

## Usage

### SmoothQuant INT8 Inference for PyTorch

We implement SmoothQuant INT8 inference for PyTorch with [CUTLASS](https://github.com/NVIDIA/cutlass) INT8 GEMM kernels, which are wrapped as PyTorch modules in [torch-int](https://github.com/Guangxuan-Xiao/torch-int). Please install [torch-int](https://github.com/Guangxuan-Xiao/torch-int) before running the SmoothQuant PyTorch INT8 inference.

We implement the quantized OPT model class in [smoothquant/opt.py](smoothquant/opt.py), which uses INT8 linear layers and bundles quantization scales. We provide the already smoothed and quantized OPT model at [https://huggingface.co/mit-han-lab/opt-[MODEL-SIZE]-smoothquant](https://huggingface.co/mit-han-lab/opt-[MODEL-SIZE]-smoothquant), where `[MODEL-SIZE]` can be `125m`, `1.3B`, `2.7B`, `6.7B`, `13B`, `30b`, and `66b`. You can load the INT8 model with the following code:

```python
from smoothquant.opt import Int8OPTForCausalLM
model = Int8OPTForCausalLM.from_pretrained("mit-han-lab/opt-30b-smoothquant")
```

You can also check [generate_act_scales.py](examples/generate_act_scales.py) and [export_int8_model.py](examples/export_int8_model.py) to see how we smooth, quantize and export INT8 models.

In [examples/smoothquant_opt_real_int8_demo.ipynb](examples/smoothquant_opt_real_int8_demo.ipynb), we use OPT-30B model to demonstrate the latency and memory advantages of SmoothQuant. We demonstrate on OPT-30B because it is the largest model we can run both the FP16 and INT8 inference on a single A100 GPU. For larger models requiring multiple GPUs, we recommend using the [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) implementation of SmoothQuant.

### Activation Channel Scales and Calibration

We provide the activation channel scales for OPT and BLOOM models in [act_scales/](act_scales/). We get those scales with 512 random sentences in the Pile validation set. You can use [examples/smoothquant_opt_demo.ipynb](examples/smoothquant_opt_demo.ipynb) to test smoothing and quantizing those models.

We also provide the script to get the activation channel scales for your models. Please refer to [examples/generate_act_scales.py](examples/generate_act_scales.py). You can use the following command to get the scales for your models:

```bash
python examples/generate_act_scales.py \
    --model-name <model_name_or_path> \
    --output-path <output_act_scales_file_path> \
    --num-samples <num_samples> \
    --seq-len <sequence_length> \
    --dataset-path <path_to_the_calibration_dataset>
```

### Demo on OPT-13B with W8A8 Fake Quantization

In [examples/smoothquant_opt_demo.ipynb](examples/smoothquant_opt_demo.ipynb), we use OPT-13B as an example to demonstrate SmoothQuant can match the accuracy of FP16 and INT8 inference, while the naive baseline cannot. We simulate INT8 inference with FP16 ([smoothquant/fake_quant.py](smoothquant/fake_quant.py)), i.e., fake quantization.

## Open Source Roadmap

The following table shows the open-source roadmap of SmoothQuant. We will gradually release the code in two weeks. Stay tuned!

- [x] Code for SmoothQuant transformation
- [x] Activation scales of OPT and BLOOM models
- [x] Demo for OPT-13B
- [x] Code for SmoothQuant smoothing factor calibration
- [x] SmoothQuant real-INT8 inference for PyTorch
- [x] SmoothQuant real-INT8 inference for FasterTransformer
- [ ] Integration with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to reproduce the results in the paper.

## Results

- SmoothQuant migrates **part of** the quantization difficulties from activation to weights, which smooths out the systematic outliers in activation, making both weights and activations **easy to quantize**. 

![migrate](figures/migrate.jpg)

- SmoothQuant can achieve W8A8 quantization of LLMs (e.g., OPT-175B) **without degrading performance**.

![accuracy](figures/accuracy.png)

- SmoothQuant can achieve **faster inference** compared to FP16 when integrated into PyTorch, while previous work LLM.int8() does not lead to acceleration (usually slower).

![torch_latency_mem](figures/torch_latency_mem.png)

- We also integrate SmoothQuant into the state-of-the-art serving framework [FasterTransformer](https://github.com/NVIDIA/FasterTransformer), achieving **faster** inference speed using only **half the GPU numbers** compared to FP16 (1 instead of 2 for OPT-66B, 4 instead of 8 for OPT-175B).

![ft_latency_mem](figures/ft_latency_mem.png)

## Citation

If you find SmoothQuant useful or relevant to your research, please kindly cite our paper:

```bibtex
@article{xiao2022smoothquant,
  title={SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models},
  author={Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Demouth, Julien and Han, Song},
  journal={arXiv},
  year={2022}
}
```
