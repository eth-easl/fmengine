import copy
import torch
import deepspeed
import transformers
from dataclasses import dataclass
from datasets import load_dataset
from itertools import chain
from torch.utils.data.dataloader import DataLoader
from typing import Dict, List

from fmengine.utils import logger_rank0 as logger


@dataclass
class AutoregressiveLanguageModelDataCollator(object):
    """
    Collate for autoregressive language models
    """

    tokenizer: transformers.PreTrainedTokenizer
    ignore_index: int = -100

    def get_attn_mask(self, input_ids):
        """
        Get triangular attention mask for a given sequence length / device.
        """
        bs = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        # lower triangular attention mask
        mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
            bs, 1, seq_length, seq_length
        )
        # convert to binary
        return mask < 0.5

    def get_position_ids(self, input_ids):
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def __call__(self, samples: List) -> Dict[str, torch.Tensor]:
        input_ids = [sample["input_ids"] for sample in samples]
        labels = copy.deepcopy(input_ids)
        # shifting input_ids & labels
        # https://d2l.ai/chapter_recurrent-neural-networks/language-model.html#learning-language-models
        input_ids = [input_id[:-1] for input_id in input_ids]
        labels = [label[1:] for label in labels]
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        labels = torch.where(
            labels == self.tokenizer.pad_token_id, self.ignore_index, labels
        )
        return (
            (
                input_ids,
                self.get_position_ids(input_ids),
                self.get_attn_mask(input_ids),
            ),
            labels,
        )


def get_jsonl_dataloader(jsonl_path, tokenizer, args):
    data_collator = AutoregressiveLanguageModelDataCollator(tokenizer)
    ctx_length = args.get("seq_length", 1024) + 1  # +1 for shifting
    streaming = args.get("streaming", False)
    seed = args.get("seed", 3407)
    batch_size = args.get("batch_size", 1)

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

    raw_datasets = load_dataset(
        "json", split="train", data_files=jsonl_path, streaming=streaming
    ).shuffle(seed=seed)

    raw_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets.column_names
    ).with_format("torch")

    dataloader = DataLoader(
        raw_datasets, shuffle=False, collate_fn=data_collator, batch_size=batch_size
    )
    return iter(deepspeed.utils.RepeatingLoader(dataloader))
