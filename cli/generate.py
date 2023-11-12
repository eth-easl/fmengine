import json
import argparse
from typing import Optional, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
)
from peft import PeftModel
from loguru import logger

def postprocess(text):
    text = text.strip()
    # logic:
    # if starts with \n, take the remaining
    if text.startswith("\n"):
        text = text.split("\n")[1]
    # if there's \n left, take the first part
    text = text.split("\n")[0]
    return text

def generate(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        torch_dtype=torch.float16,
    )
    model.eval()
    with torch.inference_mode():
        with open(args.input_file, "r") as f:
            data = [json.loads(line) for line in f]
        pipe = TextGenerationPipeline(
            model=model, tokenizer=tokenizer, device="cuda"
        )
        logger.info("Pipeline Ready")
        prompts = [datum[args.input_field] for datum in data]
        outputs = pipe(
            prompts,
            max_new_tokens=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            return_full_text=False,
        )
        results = []
        for datum, output in zip(data, outputs):
            result = datum
            result["prediction"] = [postprocess(o["generated_text"]) for o in output]
            results.append(result)
        with open(args.output_file, "w") as f:
            for datum in data:
                f.write(json.dumps(datum) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", "--model",
                        type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--input-field", type=str, default="input")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    generate(args)
