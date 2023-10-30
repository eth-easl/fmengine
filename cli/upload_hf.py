import os
import shutil
import argparse
from multiprocessing import Pool
from fmengine.utils.hf import upload_hf
from fmengine.modeling.neox.hf_interface import to_hf_model

def upload(args):
    print(args)
    os.makedirs(args.temp_path, exist_ok=True)
    os.makedirs(os.path.join(args.temp_path, 'hf'), exist_ok=True)
    steps = [x for x in os.listdir(args.ckpt_path) if x.startswith("global_step")]

    os.system(f"cd {os.path.join(args.temp_path, 'hf')} && huggingface-cli repo create {args.hf_repo} --organization {args.hf_org} && git lfs install && git clone https://huggingface.co/{args.hf_org}/{args.hf_repo} .")
    for step in steps:
        print(f"Exporting {step}")
        to_hf_model(
            in_model_path = args.ckpt_path,
            model_family=args.base_model,
            out_model_path = os.path.join(args.temp_path, step),
            step=step
        )
        shutil.copytree(os.path.join(args.temp_path, step), os.path.join(args.temp_path, 'hf', step))
        os.system(f"cd {os.path.join(args.temp_path, 'hf')} && git add . && git commit -m 'add step {step}' && git tag -a global_step{step} -m 'step{step}' && git push git@hf.co/{args.hf_org}/{args.hf_repo} global_step{step}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--temp-path", type=str, required=True, default='.cache/temp')
    parser.add_argument("--hf-org", type=str, required=True)
    parser.add_argument("--hf-repo", type=str, required=True)
    args = parser.parse_args()
    upload(args)
