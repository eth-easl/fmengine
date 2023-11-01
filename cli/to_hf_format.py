import argparse
from fmengine.modeling.llama.hf_interface import to_hf_model

def convert_to_hf_format(args):
    print(args)
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
    parser.add_argument("--step", type=str, required=True)
    parser.add_argument("--is-lora-tuned", action="store_true", default=False)
    args = parser.parse_args()
    convert_to_hf_format(args)