from fmengine.dataloader.datasets import get_jsonl_dataset, get_hf_stream_dataset
from fmengine.dataloader.loader import get_dataloader_from_datasets

def get_dataloader(datapath, tokenizer, args):
    if "jsonl" in datapath:
        ds = get_jsonl_dataset(datapath, args)
    else:
        ds = get_hf_stream_dataset(datapath, args)
    return get_dataloader_from_datasets(
        ds,
        tokenizer=tokenizer,
        args={
            "seq_length": args.get("seq_length", 1024),
            "batch_size": args.get("batch_size", 1),
        },
    )
