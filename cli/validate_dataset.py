import transformers
from fmengine.dataloader import get_dataloader

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "openlm-research/open_llama_3b_v2",
    model_max_length=2048,
    use_fast=True,
)
tokenizer.pad_token_id = tokenizer.eos_token_id

# data_loader = get_jsonl_dataloader(
#     "/mnt/scratch/xiayao/cache/datasets/exported_function_calling.jsonl",
#     tokenizer=tokenizer,
#     args={
#         "seq_length": 128,
#         "batch_size": 32,
#     },
# )
data_loader = get_dataloader(
    "c4", tokenizer, {"batch_size": 32, "seq_length": 128, "partition": "en"}
)

for it in data_loader:
    print(it)
    pass
