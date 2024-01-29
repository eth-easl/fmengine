from loguru import logger
from datasets import load_dataset
from fmengine.utils import logger_rank0 as logger


def get_hf_stream_dataset(dataset_name_or_path, args: dict = {}):
    split = args.get("split", "train")
    partition = args.get("partition", "default")
    logger.info(
        f"Loading dataset {dataset_name_or_path} with split {split} and partition {partition}"
    )
    raw_datasets = load_dataset(
        dataset_name_or_path,
        partition,
        split=split,
        streaming=True,
    )
    return raw_datasets
