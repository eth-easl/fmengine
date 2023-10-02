import os
import json
from multiprocessing import Pool
from transformers import AutoTokenizer


def count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def parallel_count(tokenizer, texts):
    # get cpu count
    n_processor = os.cpu_count()
    with Pool(n_processor) as pool:
        return sum(pool.map(count, [tokenizer] * len(texts), texts))

def count_tokens_from_file(
    filename: str,
    tokenizer_name: str,
    field: str = 'text',
    special_tokens: dict = None, 
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if special_tokens is None:
        special_tokens = {}
    with open(filename, 'r') as f:
        data = [json.loads(x)[field] for x in f.readlines()]
    return sum([count(tokenizer, x) for x in data])