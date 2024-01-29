import deepspeed
from itertools import chain
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader

from fmengine.dataloader.loader import AutoregressiveLanguageModelDataCollator


def get_jsonl_dataset(jsonl_path, args):
    streaming = args.get("streaming", False)
    seed = args.get("seed", 42)
    raw_datasets = load_dataset(
        "json", split="train", data_files=jsonl_path, streaming=streaming
    ).shuffle(seed=seed)

    return raw_datasets
