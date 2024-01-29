import os
import json
from typing import Sequence, Dict
from transformers import PreTrainedTokenizer


def write_loader_status(ckpt_path: str, skip: int):
    with open(os.path.join(ckpt_path, "dataloader_status.json"), "w") as f:
        json.dump({"skip": skip}, f)


def tokenize(texts: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    tokenized = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in texts
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizer,
):
    """Preprocess the data by tokenizing."""
    samples_tokenized = tokenize(texts, tokenizer)
