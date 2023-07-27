from transformers import AutoModel, AutoConfig, AutoTokenizer
from huggingface_hub import HfApi

def upload_hf(
        path_to_hf_model: str,
        repo_id: str,
        revision: str='main'
    ):
    model = AutoModel.from_pretrained(path_to_hf_model, from_tf=False, use_safetensors=True)
    model.push_to_hub(repo_id, revision=revision,safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(path_to_hf_model)
    tokenizer.push_to_hub(repo_id, revision=revision)
    config = AutoConfig.from_pretrained(path_to_hf_model)
    config.push_to_hub(repo_id, revision=revision)

if __name__=="__main__":
    upload_hf(".cache/exported", 'fmzip/test', revision='main')
