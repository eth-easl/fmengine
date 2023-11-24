import transformers
from fmengine.dataloader.jsonl_loader import get_jsonl_dataloader

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "openlm-research/open_llama_3b_v2",
    model_max_length=2048,
    use_fast=True,
)
tokenizer.pad_token_id = tokenizer.eos_token_id

data_loader = get_jsonl_dataloader(
    ".cache/data/dialogs.jsonl",
    tokenizer=tokenizer,
    args={
        "seq_length": 128,
        "batch_size": 2,
    },
)

for it in data_loader:
    pass
