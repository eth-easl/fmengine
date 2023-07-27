import torch
import copy
import transformers
from typing import Dict, List
from dataclasses import dataclass


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
        print("input_ids", input_ids)
        print("labels", labels)
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
