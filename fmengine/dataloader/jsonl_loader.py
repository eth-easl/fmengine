import copy
import torch
import deepspeed
from itertools import cycle
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset
from fmengine.utils import logger_rank0 as logger
from fmengine.dataloader.collate import AutoregressiveLanguageModelDataCollator

class JSONLDataset(IterableDataset):
    def __init__(self,
                 data,
                 tokenizer: Tokenizer,
                 seq_length: int, 
                 doc_sep=''
                ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length + 1 # additional 1 for shifting
        self.doc_sep = doc_sep
        self.field = 'text'
        self.it = None
        self.iter_count = 0
        self.buffer_tokens = []
        self._len = -1
    
    def state_dict(self):
        return {
            'iter_count': self.iter_count,
            'buffer_tokens': self.buffer_tokens,
        }

    def load_state_dict(self, state_dict):
        self.iter_count = state_dict['iter_count']
        self.buffer_tokens = state_dict['buffer_tokens']
        self.data = self.data.skip(self.iter_count)
    
    def get_sequence(self):
        buffer_tokens = self.buffer_tokens
        while True:
            try:
                for x in self.data:
                    self.iter_count += 1
                    curr_tokens = self.tokenizer(self.doc_sep + x['text'])['input_ids']
                    buffer_tokens += curr_tokens
                    while len(buffer_tokens) >= self.seq_length:
                        tokens = buffer_tokens[:self.seq_length]
                        buffer_tokens = buffer_tokens[self.seq_length:]
                        input_ids = torch.tensor(tokens)
                        self.buffer_tokens = buffer_tokens
                        yield {
                            'input_ids': input_ids,
                        }
            except Exception as e:
                print("next epoch")
                pass

    def get_stream(self):
        return cycle(self.get_sequence())

    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it

def get_jsonl_dataloader(
            path_to_jsonl_file: str,
            tokenizer: Tokenizer,
            num_workers = 0,
            state_dict = None,
            streaming = False,
            args = None
        ):
    seed = args.get('seed', 42)
    seq_length = args.get('seq_length', 1024)
    batch_size = args.get('batch_size', 1)
    data_group_size = args.get('data_group_size', 1)
    shuffle = args.get('shuffle', False)
    data = load_dataset(
            'json',
            split='train',
            data_files=path_to_jsonl_file,
            streaming=streaming
        ).shuffle(seed=seed).with_format('torch')
    
    stream_dataset = JSONLDataset(data, tokenizer, seq_length)
    collator = AutoregressiveLanguageModelDataCollator(tokenizer)
    if state_dict:
        stream_dataset.load_state_dict(state_dict)

    train_data_loader = torch.utils.data.DataLoader(
        stream_dataset,
        batch_size = batch_size * data_group_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = True,
        collate_fn = collator
    )
    return iter(deepspeed.utils.RepeatingLoader(train_data_loader))