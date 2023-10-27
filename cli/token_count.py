import argparse
from transformers import AutoTokenizer
from fmengine.utils.token_count import count_tokens_from_file

def count_token(args):
    token_count = count_tokens_from_file(args.input, args.tokenizer, args.field)
    print(f"Token count: {token_count}")
    epoch_size = token_count // (args.global_batch_size * args.dp_degree * args.seq_length)
    print(f"Epoch size: {epoch_size}")
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--field', type=str, default='text')
    parser.add_argument('--seq-length', type=int, default=2048)
    parser.add_argument('--global-batch-size', type=int, default=32)
    parser.add_argument('--dp-degree', type=int, default=1)
    args = parser.parse_args()
    count_token(args)