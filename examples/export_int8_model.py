import torch
import argparse
import os

from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers import AutoTokenizer

from smoothquant.opt import Int8OPTForCausalLM
from smoothquant.smooth import smooth_lm

from smoothquant.calibration import get_static_decoder_layer_scales


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='facebook/opt-13b')
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--act-scales", type=str,
                        default='act_scales/opt-13b.pt')
    parser.add_argument("--output-path", type=str,
                        default='int8_models/opt-13b-smoothquant')
    parser.add_argument('--dataset-path', type=str, default='dataset/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    args = parser.parse_args()
    model = OPTForCausalLM.from_pretrained(
        args.model_name, device_map="sequential", torch_dtype=torch.float16)
    act_scales = torch.load(args.act_scales)
    smooth_lm(model, act_scales, 0.5)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        print('Please download the Pile dataset and put the validation set at the path')
        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
        raise FileNotFoundError

    decoder_layer_scales = get_static_decoder_layer_scales(model,
                                                           tokenizer,
                                                           args.dataset_path,
                                                           num_samples=args.num_samples,
                                                           seq_len=args.seq_len)
    int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)
    int8_model.save_pretrained(args.output_path)
