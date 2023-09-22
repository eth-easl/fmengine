import os
import torch
import random
import deepspeed
import numpy as np
import transformers
from typing import Optional
from dataclasses import dataclass, field, asdict

from fmengine.utils import jload
from fmengine.trainer.llm_trainer import LLMTrainer
from fmengine.modeling._common.model import get_model
from fmengine.dataloader.jsonl_loader import get_jsonl_dataloader
from munch import munchify
from fmengine.utils.megatron import initialize_megatron
from fmengine.modeling.llama.patching import patch_llama
from fmengine.modeling.neox.flash_attention import replace_neox_attn_with_flash_attn
from fmengine.callbacks.monitor import speed_monitor, wandb_monitor


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


if __name__ == "__main__":
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
    print(ds_args)
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
        padding_side="right",
        use_fast=True,
        repo_type="",
    )
    tokenizer.pad_token = tokenizer.eos_token
    model_config = transformers.AutoConfig.from_pretrained(model_args.init_ckpt)

    train_dataloader = get_jsonl_dataloader(
        data_args.data_path,
        tokenizer=tokenizer,
        args={
            "seq_length": trainer_args.max_seq_len,
            "batch_size": data_args.batch_size,
        },
    )
    _tmp = torch.nn.Linear.reset_parameters
    torch.nn.Linear.reset_parameters = lambda x: None
    model = get_model(model_config, ds_args, activation_checkpointing_config)
    torch.nn.Linear.reset_parameters = _tmp

    ds_config["data_path"] = data_args.data_path
    
    if "lora" in ds_config:
        load_module_strict=False
    else:
        load_module_strict=True

    trainer = LLMTrainer(
        model=model,
        ds_args=ds_args,
        dataloader=train_dataloader,
        ds_config=ds_config,
        init_ckpt=model_args.init_ckpt,
        save_dir=trainer_args.output_dir,
        pretrain=trainer_args.pretrain,
        load_module_strict=load_module_strict,
        callbacks=[speed_monitor, wandb_monitor],
    )
    trainer.fit(
        steps=trainer_args.train_steps,
        profile=True,
        save_per_steps=trainer_args.save_steps,
        configs=merged_configs,
    )
