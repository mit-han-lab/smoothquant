# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models 

### [[paper](https://arxiv.org/abs/2211.10438)]

**We plan to gradually release the code in ~2 weeks. Stay tuned!**

**If you are interested in getting updates, please sign up [here](https://forms.gle/YjYQQas5Hbqge1LH9) to get notified!**


![intuition](figures/intuition.png)

## Abstract

Large language models (LLMs) show excellent performance but are compute- and memory-intensive. Quantization can reduce memory and accelerate inference. However, for LLMs beyond 100 billion parameters, existing methods cannot maintain accuracy or have to rely on techniques that do not run efficiently on hardware. We propose SmoothQuant, a training-free, lightweight, and general-purpose post-training quantization (PTQ) solution to enable lossless 8-bit weight, 8-bit activation (W8A8) quantization for LLMs that can be implemented efficiently. We observe that systematic outliers appear at fixed activation channels. Based on the fact that weights are easy to quantize while activations are not, SmoothQuant smooths the activation outliers by migrating the quantization difficulty from activations to weights with a mathematically equivalent transformation. SmoothQuant enables an INT8 quantization of both weights and activations for all the GEMMs in LLMs, including OPT-175B, BLOOM-176B, and GLM-130B. SmoothQuant has better hardware efficiency than existing techniques using mixed-precision activation quantization or weight-only quantization. We demonstrate up to 1.56x speedup and 2x memory reduction for LLMs with negligible loss in accuracy. Thanks to the hardware-friendly design, we integrate SmoothQuant into FasterTransformer, a state-of-the-art LLM serving framework, and achieve faster inference speed with half the number of GPUs compared to FP16. Our work offers a turn-key solution that reduces hardware costs and democratizes LLMs.


## Installation

```bash
conda create -n smoothquant python=3.8
conda activate smoothquant
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install huggingface accelerate datasets

python setup.py install
```

## Example

In `examples/smoothquant_opt_demo.ipynb`, we use OPT-13B as an example to demonstrate SmoothQuant can match the accuracy of FP16 and INT8 inference. In the current repo, we simulate INT8 inference with FP16 (`smoothquant/fake_quant.py`), i.e., fake quantization. We have implemented the real 8-bit quantization with INT8 CUTLASS GEMM kernels for both PyTorch and FasterTransformer. Please stay tuned for the release.


## Open Source Roadmap

The following table shows the open source roadmap of SmoothQuant. We will gradually release the code in two weeks. Stay tuned!

- [x] Code for SmoothQuant transformation
- [x] Activation scales of OPT and BLOOM models
- [x] Demo for OPT-13B
- [ ] SmoothQuant real-INT8 inference for PyTorch
- [ ] SmoothQuant real-INT8 inference for FasterTransformer
- [ ] Integration with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to reproduce the results in the paper.

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