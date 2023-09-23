import transformers
from fmengine.dataloader.jsonl_loader import get_jsonl_dataloader

tokenizer = transformers.AutoTokenizer.from_pretrained(
        'openlm-research/open_llama_3b_v2',
        model_max_length=2048,
        padding_side="left",
        use_fast=True,
        repo_type="",
    )

data_loader = get_jsonl_dataloader(
    ".cache/data/dialogs.jsonl",
    tokenizer=tokenizer,
    args={
        "seq_length": 2048,
        "batch_size": 1,
    },
)

for it in data_loader:
    print(it)