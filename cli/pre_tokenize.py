import argparse
from transformers import AutoTokenizer
from fmengine.dataloader.jsonl_loader import get_jsonl_dataloader


def pre_tokenize(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.init_ckpt, use_fast=args.fast_tokenizer
    )
    data_loader = get_jsonl_dataloader(
        args.dataset,
        tokenizer=tokenizer,
        args={
            "seq_length": 128,
            "batch_size": 32,
        },
    )
    for it in data_loader:
        print(it)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--field", type=str, default="text")
    parser.add_argument("--fast-tokenizer", action="store_true", default=True)
    args = parser.parse_args()
    pre_tokenize(args)
