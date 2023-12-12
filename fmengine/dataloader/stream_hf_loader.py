from datasets import load_dataset
from fmengine.utils import logger_rank0 as logger


def get_stream_dataset(dataset_name_or_path, args: dict = {}):
    split = args.get("split", "train")
    print(f"Loading dataset {dataset_name_or_path} split {split}")
    raw_datasets = load_dataset(
        dataset_name_or_path,
        "en",
        split=split,
        streaming=True,
    )
    print(raw_datasets)
    return raw_datasets
