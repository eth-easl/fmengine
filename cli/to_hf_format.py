import os
import argparse
from tqdm import tqdm
from fmengine.modeling.llama.hf_interface import to_hf_model

def convert_to_hf_format(args):
    print(args)
    if args.step =='all':
        steps = os.listdir(args.in_model_path)
        for step in steps:
            if os.path.isdir(os.path.join(args.in_model_path, step)) and \
            not os.path.exists(os.path.join(args.out_model_path, step)):
                to_hf_model(
                    in_model_path=args.in_model_path,
                    model_family=args.base_model,
                    out_model_path=os.path.join(args.out_model_path, step),
                    step=step,
                    is_lora_tuned=args.is_lora_tuned,
                )
    else:
        to_hf_model(
            in_model_path=args.in_model_path,
            model_family=args.base_model,
            out_model_path=args.out_model_path,
            step=args.step,
            is_lora_tuned=args.is_lora_tuned,
        )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-model-path", type=str, required=True)
    parser.add_argument("--out-model-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--step", type=str, default='all')
    parser.add_argument("--is-lora-tuned", action="store_true", default=False)
    args = parser.parse_args()
    convert_to_hf_format(args)