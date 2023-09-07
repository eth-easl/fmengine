import torch
import random
import deepspeed
import numpy as np
import transformers
from typing import Optional
from dataclasses import dataclass, field

from fmengine.utils import jload
from fmengine.trainer.llm_trainer import LLMTrainer
from fmengine.modeling._common.model import get_model
from fmengine.dataloader.jsonl_loader import get_jsonl_dataloader
from fmengine.modeling.neox.optimizations import replace_neox_attn_with_flash_attn
from fmengine.modeling.llama.optimizations import replace_llama_attn_with_flash_attn

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
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    num_workers: int = field(default=1)
    seq_length: int = field(default=1024)

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

if __name__=="__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainerArguments, DeepspeedArguments))
    model_args, data_args, trainer_args, ds_args = parser.parse_args_into_dataclasses()

    # setup deepspeed and other stuff
    assert ds_args.use_deepspeed
    deepspeed.init_distributed(dist_backend="nccl")

    ds_args.world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(ds_args.local_rank)

    ds_config = read_ds_config(ds_args.deepspeed_config)

    data_args.num_workers = 2 * ds_args.world_size // ds_args.pipe_parallel_size // ds_args.model_parallel_size
    data_args.batch_size = ds_config.get("train_micro_batch_size_per_gpu", 1)
    activation_checkpointing_config = ds_config.pop("activation_checkpointing", None)

    random.seed(ds_args.seed)
    np.random.seed(ds_args.seed)
    torch.manual_seed(ds_args.seed)
    deepspeed.runtime.utils.set_random_seed(ds_args.seed)

    if model_args.use_flash_attn:
        print("⚡⚡⚡ [Flash Attention] Enabled")
        replace_neox_attn_with_flash_attn()
        replace_llama_attn_with_flash_attn()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.init_ckpt,
        model_max_length=trainer_args.max_seq_len,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model_config = transformers.AutoConfig.from_pretrained(model_args.init_ckpt)

    train_dataloader = get_jsonl_dataloader(
        data_args.data_path,
        tokenizer = tokenizer,
        args = {
            'seq_length': trainer_args.max_seq_len,
            'batch_size': data_args.batch_size
        }
    )
    model = get_model(
        model_config,
        ds_args,
        activation_checkpointing_config
    )
    ds_config['data_path'] = data_args.data_path
    trainer = LLMTrainer(
        model = model,
        ds_args = ds_args,
        dataloader = train_dataloader,
        ds_config = ds_config,
        init_ckpt = model_args.init_ckpt,
        save_dir=trainer_args.output_dir,
        pretrain = trainer_args.pretrain
    )
    trainer.fit(
        steps = trainer_args.train_steps,
        profile = True,
        log_per_steps = trainer_args.log_steps,
        save_per_steps = trainer_args.save_steps
    )