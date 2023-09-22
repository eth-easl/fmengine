import os
import torch
import accelerate
from pathlib import Path
from loguru import logger
from safetensors.torch import save_model
from transformers import AutoConfig, AutoTokenizer, GPTNeoXForCausalLM


def to_hf_model(
    in_model_path: str,
    model_family: str,
    out_model_path: str,
    step="latest",
    fp16=True,
):
    os.makedirs(out_model_path, exist_ok=True)
    config = AutoConfig.from_pretrained(model_family)
    tokenizer = AutoTokenizer.from_pretrained(model_family)
    tensors = {}
    n_layers = config.num_hidden_layers
    tokenizer_size = config.vocab_size
    logger.info(f"[config]: total layers: {n_layers}, vocab size: {tokenizer_size}")
    if step == "latest":
        with open(os.path.join(in_model_path, "latest"), "r") as f:
            step = f.read().strip()
    logger.info("Processing step: {}", step)
    for pt in Path(os.path.join(in_model_path, step)).iterdir():
        loaded = torch.load(pt, map_location="cpu")

        if not pt.name.startswith("layer_"):
            continue

        if pt.name == "layer_00-model_00-model_states.pt":
            logger.info("Loading embedding layer")
            tensors["gpt_neox.embed_in.weight"] = loaded["weight"][:tokenizer_size, :]
            continue

        if pt.name == f"layer_{n_layers + 1}-model_00-model_states.pt":
            logger.info("Loading final layer norm")
            tensors["gpt_neox.final_layer_norm.weight"] = loaded["weight"]
            tensors["gpt_neox.final_layer_norm.bias"] = loaded["bias"]
            continue

        if pt.name == f"layer_{n_layers + 2}-model_00-model_states.pt":
            logger.info("Loading embedding output layer")
            tensors["embed_out.weight"] = loaded["weight"][:tokenizer_size, :]
            continue

        layer_i = int(pt.name.split("-")[0].replace("layer_", "")) - 1
        logger.info(f"Loading {layer_i}th layer")
        layer_loaded = {
            f"gpt_neox.layers.{layer_i}.{nm}": weight for nm, weight in loaded.items()
        }
        tensors.update(layer_loaded)
    # with accelerate.init_empty_weights():
    model = GPTNeoXForCausalLM(config)
    model.load_state_dict(tensors, strict=True)
    if fp16:
        model.half()
    save_model(
        model,
        os.path.join(out_model_path, "model.safetensors"),
        metadata={"step": step, "format": "pt"},
    )

    config.save_pretrained(out_model_path)
    tokenizer.save_pretrained(out_model_path)


def from_hf_model(model_name_or_path: str, outpath: str, mp: int):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name_or_path)
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    loaded = model.state_dict()
    n_layers = model_config.num_hidden_layers
    # embedding
    sd = {"weight": loaded["gpt_neox.embed_in.weight"]}
    torch.save(sd, outpath / "layer_00-model_00-model_states.pt")
    # norm
    sd = {
        f"weight": loaded["gpt_neox.final_layer_norm.weight"],
        f"bias": loaded["gpt_neox.final_layer_norm.bias"],
    }
    torch.save(sd, outpath / f"layer_{n_layers + 1}-model_00-model_states.pt")
    # lm head
    sd = {f"weight": loaded["embed_out.weight"]}
    torch.save(sd, outpath / f"layer_{n_layers + 2}-model_00-model_states.pt")
    # decoder layers
    for layer_i in range(n_layers):
        sd = {
            nm.replace(f"gpt_neox.layers.{layer_i}.", f""): weight
            for nm, weight in loaded.items()
            if nm.startswith(f"gpt_neox.layers.{layer_i}.")
        }
        torch.save(sd, outpath / f"layer_{layer_i + 1:02d}-model_00-model_states.pt")

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
        torch.save(model_state, outpath / f"mp_rank_{rank:02d}_model_states.pt")
