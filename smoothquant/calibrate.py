import os
import argparse

import torch
import torch.nn as nn
import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from datasets import load_dataset
import functools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_and_tokenizer(model_name, tokenizer_name=None):
    if tokenizer_name is None:
        tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def calibrate(model, tokenizer, dataset_path, n_samples, seq_len):
    model.eval()
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)
        stat_tensor(name + ".after", y)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    if not os.path.exists(dataset_path):
        print(f'Cannot find the dataset at {dataset_path}')
        print('Please download the Pile dataset and put the validation set at the path')
        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
        raise FileNotFoundError

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm.tqdm(range(n_samples)):
        input_ids = tokenizer(dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='facebook/opt-1.3b', help='model name')
    parser.add_argument('--output-path', type=str,
                        help='where to save the meta data')
    parser.add_argument('--data-path', type=str, default='../dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--num-samples', type=int, default=512)
    parser.add_argument('--seq-len', type=int, default=512)
    parser.add_argument('--tokenizer-name', type=str, default=None,
                        help='name of the tokenizer to use. By default, use the same as model-name')
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(
        args.model_name, args.tokenizer_name)

    act_scales = calibrate(model, tokenizer, args.data_path,
                           args.num_samples, args.seq_len)

    os.makedirs(os.path.dirname(
        os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == '__main__':
    main()
