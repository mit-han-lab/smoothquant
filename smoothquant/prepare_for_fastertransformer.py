import torch

from argparse import ArgumentParser
from smoothquant.smooth import smooth_lm
from transformers import AutoModelForCausalLM


@torch.no_grad()
def main():
    parser = ArgumentParser()
    parser.add_argument("--model", help="Name or path of non-smoothed model")
    parser.add_argument("--scales", help="Path of activation scales. Can be obtained with the calibrate.py script.")
    parser.add_argument("--output-path", help="Where to save the smoothed model")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map='auto')
    act_scales = torch.load(args.scales)

    smooth_lm(model, act_scales, 0.5)
    model.save_pretrained(getattr(args, "output-path"))


if __name__ == '__main__':
    main()
