import os
from multiprocessing import Pool
from transformers import AutoTokenizer


def count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def parallel_count(tokenizer_name, texts):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # get cpu count
    n_processor = os.cpu_count()
    with Pool(n_processor) as pool:
        return sum(pool.map(count, [tokenizer] * len(texts), texts))
