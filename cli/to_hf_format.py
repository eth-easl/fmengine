import argparse
from fmengine.modeling.llama.hf_interface import to_hf_model

def convert_to_hf_format(args):
    to_hf_model(
        in_model_path=args.in_model_path,
        model_family=args.base_model,
        out_model_path=args.out_model_path,
        step=args.step,
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-model-path", type=str, required=True)
    parser.add_argument("--out-model-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--step", type=str, required=True)
    args = parser.parse_args()
    convert_to_hf_format(args)