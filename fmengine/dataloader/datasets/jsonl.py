from datasets import load_dataset


def get_jsonl_dataset(jsonl_path, args):
    streaming = args.get("streaming", False)
    seed = args.get("seed", 3407)
    raw_datasets = load_dataset(
        "json",
        split="train",
        data_files=jsonl_path,
        streaming=streaming,
        trust_remote_code=True,
    ).shuffle(seed=seed)

    return raw_datasets
