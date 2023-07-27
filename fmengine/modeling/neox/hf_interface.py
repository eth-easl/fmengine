import os
import torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import save_file
from loguru import logger
def to_hf_model(
        in_model_path: str,
        model_family: str,
        out_model_path: str,
        step='latest',
    ):
    os.makedirs(out_model_path, exist_ok=True)
    config = AutoConfig.from_pretrained(model_family)
    tokenizer = AutoTokenizer.from_pretrained(model_family)
    tensors = {}
    n_layers = config.num_hidden_layers
    tokenizer_size = config.vocab_size
    if step == 'latest':
        with open(os.path.join(in_model_path, 'latest'), 'r') as f:
            step = f.read().strip()
    logger.info("Processing step: {}", step)
    for pt in Path(os.path.join(in_model_path, step)).iterdir():
        logger.info(f"Processing {pt}")
        loaded = torch.load(pt, map_location="cpu")
        if not pt.name.startswith('layer_'):
            continue
        if pt.name == 'layer_00-model_00-model_states.pt':
            tensors['gpt_neox.embed_in.weight'] = loaded['weight'][: tokenizer_size, :]
            continue
        if pt.name == f'layer_{n_layers + 1}-model_00-model_states.pt':
            tensors['gpt_neox.final_layer_norm.weight'] = loaded['weight']
        if pt.name == f'layer_{n_layers + 2}-model_00-model_states.pt':
            tensors['embed_out.weight'] = loaded['weight'][: tokenizer_size, :]
            continue
        layer_i = int(pt.name.split('-')[0].replace('layer_', '')) - 1
        layer_loaded = { f"gpt_neox.layers.{layer_i}.{nm}": weight for nm, weight in loaded.items() }
        tensors.update(layer_loaded)

    save_file(tensors, os.path.join(out_model_path, 'model.safetensors'))
    config.save_pretrained(out_model_path)
    tokenizer.save_pretrained(out_model_path)