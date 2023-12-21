import deepspeed
from itertools import chain
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader

from fmengine.dataloader.loader import AutoregressiveLanguageModelDataCollator


def get_jsonl_dataloader(jsonl_path, tokenizer, return_repeating_loader=True, args={}):
    data_collator = AutoregressiveLanguageModelDataCollator(
        tokenizer, return_dict=args.get("return_dict", False)
    )
    batch_size = args.get("batch_size", 1)
    ctx_length = args.get("seq_length", 1024) + 1  # +1 for shifting

    def tokenize(examples):
        examples = tokenizer(examples["text"], truncation=True, max_length=ctx_length)
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= ctx_length:
            total_length = (total_length // ctx_length) * ctx_length
        result = {
            k: [t[i : i + ctx_length] for i in range(0, total_length, ctx_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    raw_datasets = get_jsonl_dataset(jsonl_path, tokenizer, args)
    raw_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets.column_names
    ).with_format("torch")
    dataloader = DataLoader(
        raw_datasets, shuffle=False, collate_fn=data_collator, batch_size=batch_size
    )
    return iter(deepspeed.utils.RepeatingLoader(dataloader))



def get_jsonl_dataset(jsonl_path, tokenizer, args):
    streaming = args.get("streaming", False)
    seed = args.get("seed", 42)
    batch_size = args.get("batch_size", 1)

    raw_datasets = load_dataset(
        "json", split="train", data_files=jsonl_path, streaming=streaming
    ).shuffle(seed=seed)

    return raw_datasets
