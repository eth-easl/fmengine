from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

import torch
import transformers

@dataclass
class Arguments:
    model_name: Optional[str] = field(default="/path/to/llama-7b-hf")
    output_dir: str = field(default="./llama-7B-init-ckpt")
    mp_world_size: int = field(default=1)

def write_ckpt(
        outpath: Path,
        model: torch.nn.Module,
        model_config: transformers.AutoConfig,
        mp: int
    ):
    loaded = model.state_dict()
    n_layers = model_config.num_hidden_layers
    # embedding
    sd = {"weight": loaded['gpt_neox.embed_in.weight']}
    torch.save(sd, outpath / "layer_00-model_00-model_states.pt")
    # norm
    sd = {
        f"weight": loaded['gpt_neox.final_layer_norm.weight'],
        f"bias": loaded['gpt_neox.final_layer_norm.bias'],
    }
    torch.save(sd, outpath / f"layer_{n_layers + 1}-model_00-model_states.pt")
    # lm head
    sd = {f"weight": loaded['embed_out.weight']}
    torch.save(sd, outpath / f"layer_{n_layers + 2}-model_00-model_states.pt")
    # decoder layers
    for layer_i in range(n_layers):
        sd = {nm.replace(f"gpt_neox.layers.{layer_i}.", f""): weight for nm, weight in loaded.items() if nm.startswith(f"gpt_neox.layers.{layer_i}.")}
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

def main():
    parser = transformers.HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    model_config = transformers.AutoConfig.from_pretrained(args.model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)

    outpath = Path(args.output_dir)
    if outpath.exists():
        print(f"{outpath} exists. Do nothing.")
        exit(0)

    print(f"create {outpath}")
    outpath.mkdir()
    steppath = outpath / "global_step001"
    steppath.mkdir()

    write_ckpt(steppath, model, model_config, args.mp_world_size)
    with open(outpath / "latest", "w") as fout:
        fout.write("global_step001")

    tokenizer.save_pretrained(outpath)
    model_config.save_pretrained(outpath)


if __name__ == "__main__":
    main()