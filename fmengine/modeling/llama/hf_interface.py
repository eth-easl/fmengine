import os
import torch
import transformers
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

from fmengine.modeling.llama.optimizations import (
    smart_tokenizer_and_embedding_resize,
)
from fmengine.dataloader.constants import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_UNK_TOKEN,
)

def write_ckpt(outpath: Path, model: torch.nn.Module, model_config: transformers.AutoConfig, mp: int):
    loaded = model.state_dict()

    n_layers = model_config.num_hidden_layers
    # embedding
    sd = {"weight": loaded['model.embed_tokens.weight']}
    torch.save(sd, os.path.join(outpath,  "layer_00-model_00-model_states.pt"))
    # norm
    sd = {f"weight": loaded['model.norm.weight']}
    torch.save(sd, os.path.join(outpath, f"layer_{n_layers + 1}-model_00-model_states.pt"))
    # lm head
    sd = {f"weight": loaded['lm_head.weight']}
    torch.save(sd, os.path.join(outpath, f"layer_{n_layers + 2}-model_00-model_states.pt"))
    # decoder layers
    for layer_i in range(n_layers):
        sd = {nm.replace(f"model.layers.{layer_i}.", f""): weight for nm, weight in loaded.items() if nm.startswith(f"model.layers.{layer_i}.")}
        torch.save(sd, os.path.join(outpath, f"layer_{layer_i + 1:02d}-model_00-model_states.pt"))

    model_state = {
        "dp_world_size": 1,
        "mp_world_size": mp,
        "module": None,
        "optimizer": None,
        "global_steps": 1,
        "skipped_steps": 1,
        "iteration": 1,
    }
    for rank in range(mp):
        torch.save(model_state, os.path.join(outpath, f"mp_rank_{rank:02d}_model_states.pt"))


def from_hf(model_name_or_path: str, outdir: str, mp_size:int):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    model_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
            "pad_token": DEFAULT_PAD_TOKEN,
        }
    )
    outpath = Path(outdir)
    if outpath.exists():
        print(f"Output directory {outpath} already exists. Exiting.")
        exit(0)
    print(f"Writing to {outpath}")
    outpath.mkdir()
    with open(os.path.join(outpath, "latest"), "w") as fout:
        fout.write("global_step001")
    steppath = os.path.join(outpath, "global_step001")
    os.mkdir(steppath)
    write_ckpt(steppath, model, model_config, mp_size)
    tokenizer.save_pretrained(outpath)
    model_config.save_pretrained(outpath)