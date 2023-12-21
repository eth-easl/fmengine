import os
import torch
import random
import deepspeed
import pathlib
import numpy as np
import transformers
from munch import munchify
from typing import Optional
from dataclasses import dataclass, field, asdict

from fmengine.utils import jload, get_rank
from fmengine.trainer.llm_trainer import LLMTrainer
from fmengine.modeling._common.model import get_model
from fmengine.dataloader.jsonl_loader import get_jsonl_dataloader
from fmengine.dataloader.stream_hf_loader import get_stream_dataset
from fmengine.dataloader.loader import get_dataloader_from_datasets
from fmengine.utils.megatron import initialize_megatron
from fmengine.modeling.llama.patching import patch_llama
from fmengine.modeling.neox.flash_attention import replace_neox_attn_with_flash_attn
from fmengine.callbacks.monitor import speed_monitor, wandb_monitor
from fmengine.modeling.sigma.configuration_sigma import SigmaConfig
def read_ds_config(config_path):
    config = jload(config_path)
    return config


@dataclass
class ModelArguments:
    init_ckpt: str = field(default="llama-7B-init-test-ckpt")
    use_flash_attn: Optional[bool] = field(default=False)
    # fused ops may pose changes to the training process, see warnings while use.
    # by default this is disabled
    use_fused_ops: Optional[bool] = field(default=False)


@dataclass
class DeepspeedArguments:
    use_deepspeed: Optional[bool] = field(default=True)
    rank: int = field(default=None)
    local_rank: int = field(default=None)
    pipe_parallel_size: int = field(default=1)
    model_parallel_size: int = field(default=1)
    world_size: int = field(default=None)
    seed: int = field(default=3407)
    deepspeed_config: Optional[str] = field(default=None)
    # TODO(xiaoyuan): should be 2 number, but now we make the second one be default 0
    # TODO(xiaoyuan): should be in model args, which is not passed into Module init
    window_size: Optional[int] = field(default=0)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    num_workers: int = field(default=1)


@dataclass
class TrainerArguments:
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./output")
    max_seq_len: int = field(default=128)
    train_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    save_steps: int = field(default=100)
    log_steps: int = field(default=1)
    pretrain: bool = field(default=False)
    project_name: str = field(default="fmengine")
    experiment_name: str = field(default="experiment")
    dry_run: bool = field(default=False)  # only for memory information
    res_dir: str = field(default="./output")  # save memory info result, not model checkpoint



if __name__ == "__main__":
    torch.cuda.reset_max_memory_allocated()
    start = torch.cuda.memory_allocated()
    print(f"[rank: {get_rank()}] cuda memory start {start / 2**30} GB")

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainerArguments, DeepspeedArguments)
    )
    model_args, data_args, trainer_args, ds_args = parser.parse_args_into_dataclasses()
    # merge all configs

    # setup deepspeed and other stuff
    assert ds_args.use_deepspeed
    deepspeed.init_distributed(dist_backend="nccl")

    ds_args.world_size = torch.distributed.get_world_size()
    if ds_args.local_rank is None:
        ds_args.local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(ds_args.local_rank)

    ds_config = read_ds_config(ds_args.deepspeed_config)
    ds_args.deepspeed_config = munchify(ds_config)
    ds_args.use_cpu_initialization = False
    ds_args.params_dtype = torch.bfloat16
    ds_args.use_mup = False

    initialize_megatron(ds_args)
    merged_configs = {
        "model": asdict(model_args),
        "data": asdict(data_args),
        "trainer": asdict(trainer_args),
        "deepspeed": ds_config,
    }

    data_args.num_workers = (
        2
        * ds_args.world_size
        // ds_args.pipe_parallel_size
        // ds_args.model_parallel_size
    )

    data_args.batch_size = ds_config.get("train_micro_batch_size_per_gpu", 1)
    activation_checkpointing_config = ds_config.pop("activation_checkpointing", None)

    random.seed(ds_args.seed)
    np.random.seed(ds_args.seed)
    torch.manual_seed(ds_args.seed)
    deepspeed.runtime.utils.set_random_seed(ds_args.seed)

    patch_llama(model_args.use_flash_attn, model_args.use_fused_ops, ds_args)
    replace_neox_attn_with_flash_attn()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.init_ckpt,
        model_max_length=trainer_args.max_seq_len,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config = transformers.AutoConfig.from_pretrained(model_args.init_ckpt)
    # model_config = SigmaConfig.from_pretrained(model_args.init_ckpt)
    if "jsonl" in data_args.data_path:
        train_dataloader = get_jsonl_dataloader(
            data_args.data_path,
            tokenizer=tokenizer,
            args={
                "seq_length": trainer_args.max_seq_len,
                "batch_size": data_args.batch_size,
            },
        )
    else:
        # load from HF dataset
        stream_dataset = get_stream_dataset(data_args.data_path)
        train_dataloader = get_dataloader_from_datasets(
            stream_dataset, tokenizer=tokenizer
        )

    _tmp = torch.nn.Linear.reset_parameters
    torch.nn.Linear.reset_parameters = lambda x: None
    print("sliding window size:", ds_args.window_size)
    model = get_model(model_config, ds_args, activation_checkpointing_config)

    if ds_config.get("precision", "bfloat16"):
        print("Using bfloat16")
        # TODO(xiaoyuan): lora is better used with fp32
        model = model.bfloat16()

    if "lora" in ds_config:
        for n, p in model.named_parameters():
            if "lora" in n.lower():
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        # print total trainable params
        print(
            f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}/{sum(p.numel() for p in model.parameters())}"
        )
    torch.nn.Linear.reset_parameters = _tmp

    ds_config["data_path"] = data_args.data_path

    if "lora" in ds_config:
        load_module_strict = False
    else:
        load_module_strict = True

    trainer = LLMTrainer(
        model=model,
        ds_args=ds_args,
        dataloader=train_dataloader,
        ds_config=ds_config,
        init_ckpt=model_args.init_ckpt,
        save_dir=trainer_args.output_dir,
        pretrain=trainer_args.pretrain,
        dry_run=trainer_args.dry_run,
        load_module_strict=load_module_strict,
        callbacks=[speed_monitor, wandb_monitor],
    )

    trainer.fit(
        steps=trainer_args.train_steps,
        profile=ds_args.deepspeed_config.flops_profiler.enabled,
        save_per_steps=trainer_args.save_steps,
        configs=merged_configs,
        project=trainer_args.project_name,
        experiment=trainer_args.experiment_name,
    )

    exp_res_dir = pathlib.Path(trainer_args.res_dir) / trainer_args.experiment_name
    exp_res_dir.mkdir(parents=True, exist_ok=True)
    end = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    print(f"[rank :{get_rank()}] cuda memory peak {(peak - start) / 2**30} GB")
    with open(exp_res_dir / f"mem-{get_rank()}.txt", "w") as f:
        f.write(f"{(peak - start)}\n")
