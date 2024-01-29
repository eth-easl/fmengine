import torch
import argparse
from transformers import AutoTokenizer
from fmengine.dataloader.datasets.jsonl_loader import get_jsonl_dataloader
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm
import webdataset as wds


def pre_tokenize(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.init_ckpt, use_fast=args.fast_tokenizer
    )
    data_loader = get_jsonl_dataloader(
        args.dataset,
        tokenizer=tokenizer,
        return_repeating_loader=False,
        args={
            "seq_length": args.seq_length,
            "batch_size": 1,
            "return_dict": True,
        },
    )
    sink = wds.TarWriter(args.output)
    logger.info("Pre-tokenizing dataset...")
    for idx, it in enumerate(tqdm(data_loader)):
        data = {
            k + ".pyd": v
            for k, v in it.items()
            if k in ["input_ids", "attention_mask", "labels"]
        }
        data["__key__"] = "sample%06d" % idx
        sink.write(data)
    sink.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--field", type=str, default="text")
    parser.add_argument("--fast-tokenizer", action="store_true", default=True)
    args = parser.parse_args()
    pre_tokenize(args)
