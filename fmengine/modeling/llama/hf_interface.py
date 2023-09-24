import os
import torch
import transformers
from pathlib import Path
from loguru import logger
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
from safetensors.torch import save_model
from fmengine.modeling.llama.flash_attention import (
    smart_tokenizer_and_embedding_resize,
)
from fmengine.dataloader.constants import (
    DEFAULT_BOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_EOS_TOKEN,
    DEFAULT_UNK_TOKEN,
)


def write_ckpt(
    outpath: Path,
    model: torch.nn.Module,
    model_config: transformers.AutoConfig,
    mp: int,
):
    loaded = model.state_dict()
    n_layers = model_config.num_hidden_layers
    if mp == 1:
        # embedding
        sd = {"weight": loaded["model.embed_tokens.weight"]}
        torch.save(sd, os.path.join(outpath, "layer_00-model_00-model_states.pt"))
        # norm
        sd = {"weight": loaded["model.norm.weight"]}
        torch.save(
            sd,
            os.path.join(outpath, f"layer_{n_layers+1:02d}-model_00-model_states.pt"),
        )
        # lm head
        sd = {"weight": loaded["lm_head.weight"]}
        torch.save(
            sd,
            os.path.join(outpath, f"layer_{n_layers+2:02d}-model_00-model_states.pt"),
        )
        # decoder layers
        for layer_i in range(n_layers):
            sd = {
                nm.replace(f"model.layers.{layer_i}.", f""): weight
                for nm, weight in loaded.items()
                if nm.startswith(f"model.layers.{layer_i}.")
            }
            torch.save(
                sd,
                os.path.join(
                    outpath, f"layer_{layer_i+1:02d}-model_00-model_states.pt"
                ),
            )
    else:
        # embedding
        for i_mp in range(mp):
            vocab_size = loaded["model.embed_tokens.weight"].size(0) // mp
            sd = {
                "weight": loaded["model.embed_tokens.weight"][
                    i_mp * vocab_size : (i_mp + 1) * vocab_size
                ]
            }
            torch.save(
                sd, os.path.join(outpath, f"layer_00-model_{i_mp:02d}-model_states.pt")
            )

            sd = {"weight": loaded["model.norm.weight"]}
            torch.save(
                sd,
                os.path.join(
                    outpath, f"layer_{n_layers+1:02d}-model_{i_mp:02d}-model_states.pt"
                ),
            )

            assert loaded["lm_head.weight"].size(0) // mp == vocab_size
            sd = {
                "weight": loaded["lm_head.weight"][
                    i_mp * vocab_size : (i_mp + 1) * vocab_size
                ]
            }
            torch.save(
                sd,
                os.path.join(
                    outpath, f"layer_{n_layers+2:02d}-model_{i_mp:02d}-model_states.pt"
                ),
            )

            for layer_i in range(n_layers):
                sd = {
                    nm.replace(f"model.layers.{layer_i}.", f""): weight
                    for nm, weight in loaded.items()
                    if nm.startswith(f"model.layers.{layer_i}.")
                }
                for n, p in sd.items():
                    if (
                        "gate_proj" in n
                        or "up_proj" in n
                        or "q_proj" in n
                        or "k_proj" in n
                        or "v_proj" in n
                    ):
                        dim = p.size(0) // mp
                        sd[n] = p[i_mp * dim : (i_mp + 1) * dim]

                    elif "down_proj" in n or "o_proj" in n:
                        dim = p.size(1) // mp
                        sd[n] = p[:, i_mp * dim : (i_mp + 1) * dim]
                torch.save(
                    sd,
                    os.path.join(
                        outpath,
                        f"layer_{layer_i+1:02d}-model_{i_mp:02d}-model_states.pt",
                    ),
                )

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
        torch.save(
            model_state, os.path.join(outpath, f"mp_rank_{rank:02d}_model_states.pt")
        )


def from_hf(model_name_or_path: str, outdir: str, mp_size: int):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    model_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    torch.nn.Linear.reset_parameters = lambda x: None
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
    # if tokenizer.pad_token is None:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN),
    #         tokenizer=tokenizer,
    #         model=model,
    #     )
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
            tensors["model.embed_tokens.weight"] = loaded["weight"][:tokenizer_size, :]
            continue

        if pt.name == f"layer_{n_layers + 1}-model_00-model_states.pt":
            logger.info("Loading final layer norm")
            tensors["model.norm.weight"] = loaded["weight"]
            continue

        if pt.name == f"layer_{n_layers + 2}-model_00-model_states.pt":
            logger.info("Loading embedding output layer")
            tensors["lm_head.weight"] = loaded["weight"][:tokenizer_size, :]
            continue

        layer_i = int(pt.name.split("-")[0].replace("layer_", "")) - 1
        logger.info(f"Loading {layer_i}th layer")

        layer_loaded = {
            f"model.layers.{layer_i}.{nm}": weight for nm, weight in loaded.items()
        }
        tensors.update(layer_loaded)
    # with accelerate.init_empty_weights():
    model = LlamaForCausalLM(config)
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
