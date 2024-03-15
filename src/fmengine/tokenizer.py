def get_tokenizer(trainer_config):
    tokenizer_type = trainer_config.tokenizer.tokenizer_type.lower()
    tokenizer_path = trainer_config.tokenizer.tokenizer_name_or_path
    assert tokenizer_type in ['hf', 'openai'], f"Unknown tokenizer type {tokenizer_type}"
    if tokenizer_type == 'openai':
        import tiktoken
        tokenizer = tiktoken.get_encoding(tokenizer_path)
    elif tokenizer_type == 'hf':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        raise NotImplementedError(f"Tokenizer type {tokenizer_type} not implemented")
    return tokenizer