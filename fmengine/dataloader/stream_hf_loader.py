from datasets import load_dataset
from fmengine.utils import logger_rank0 as logger


def get_stream_dataset(dataset_name_or_path, tokenizer, args):
    split = args.get("split", "train")
    raw_datasets = load_dataset(dataset_name_or_path, split=split, streaming=True)
    return raw_datasets
