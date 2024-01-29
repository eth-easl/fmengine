import copy
import torch
import deepspeed
import transformers
from itertools import chain
from typing import List, Dict
from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader
from fmengine.utils import logger_rank0 as logger
from transformers import DataCollatorForLanguageModeling
from fmengine.dataloader.constants import DEFAULT_IGNORE_INDEX

# deprecated
@dataclass
class AutoregressiveLanguageModelDataCollator(object):
    """
    Collate for autoregressive language models
    """

    tokenizer: transformers.PreTrainedTokenizer
    return_dict: bool
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
        if self.return_dict:
            return {
                "input_ids": input_ids,
                "position_ids": self.get_position_ids(input_ids),
                "attention_mask": self.get_attn_mask(input_ids),
                "labels": labels,
            }
        return (
            (
                input_ids,
                self.get_position_ids(input_ids),
                self.get_attn_mask(input_ids),
            ),
            labels,
        )


def get_dataloader_from_datasets(
    raw_datasets, 
    tokenizer,
    return_repeating_loader=True, 
    shifting_labels = True,
    return_dict = True,
    args={}
):
    batch_size = args.get("batch_size", 1)
    ctx_length = args.get("seq_length", 1024)
    field = args.get("field", "text")
    block_size = args.get("block_size", ctx_length)

    def tokenize(examples):
        return tokenizer(
            examples[field], 
            truncation=True,
            padding=True,
            max_length=ctx_length
        )
        
    def get_position_ids(input_ids):
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)
    
    def get_attn_mask(input_ids):
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
    
    def group_texts(examples):
        # TODO(xiaozhe): we'd better take keys from the dataset, but sometimes dataset doesn't have keys (don't understand why, but I failed to load keys from slimpajama).
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        input_ids = [torch.tensor(input_id) for input_id in result['input_ids']]
        labels = [torch.tensor(label) for label in result['labels']]
        
        if shifting_labels:
            input_ids = [input_id[:-1] for input_id in input_ids]
            labels = [label[1:] for label in labels]
        # convert to torch tensors
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        labels = torch.where(
            labels == tokenizer.pad_token_id,
            DEFAULT_IGNORE_INDEX,
            labels
        )
        if return_dict:
            res = {
                "input_ids": input_ids,
                "position_ids": get_position_ids(input_ids),
                "attention_mask": get_attn_mask(input_ids),
                "labels": labels,
            }
            return res
        res = (
            (
                input_ids,
                get_position_ids(input_ids),
                get_attn_mask(input_ids),
            ),
            labels,
        )
        return res
        
    tokenized_ds = raw_datasets.map(
        tokenize,
        batched=True,
        remove_columns=["text", "timestamp", "url"]
    )
    lm_ds = tokenized_ds.map(
        group_texts,
        batched=True,
    )
    dataloader = DataLoader(
        lm_ds, shuffle=False, batch_size=batch_size
    )
    if return_repeating_loader:
        return iter(deepspeed.utils.RepeatingLoader(dataloader))
    else:
        return dataloader