import transformers
from fmengine.dataloader.jsonl_loader import get_jsonl_dataloader
from fmengine.dataloader.stream_hf_loader import get_stream_dataset
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
data_loader = get_stream_dataset("cerebras/SlimPajama-627B")
for it in data_loader:
    print(it)
    pass
