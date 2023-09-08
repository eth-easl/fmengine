import wandb
from fmengine.utils import rank_zero

@rank_zero
def rank0_init_wandb(**kwargs):
    wandb.init(**kwargs)

@rank_zero
def rank0_log(metrics):
    wandb.log(metrics)

@rank_zero
def rank0_print(msg):
    print(msg)