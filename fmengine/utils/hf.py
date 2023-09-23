import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from huggingface_hub import HfApi


def upload_hf(path_to_hf_model: str, repo_id: str, revision: str):
    print(f"push to {repo_id} revision {revision}")
    model = AutoModel.from_pretrained(
        path_to_hf_model, from_tf=False, use_safetensors=True, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(path_to_hf_model)
    config = AutoConfig.from_pretrained(path_to_hf_model)

    model.push_to_hub(
        repo_id, safe_serialization=True, revision=revision, blocking=False
    )
    tokenizer.push_to_hub(repo_id, revision=revision)
    config.push_to_hub(repo_id, revision=revision)
