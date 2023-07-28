import time
import wandb
import deepspeed
from typing import Dict
from deepspeed.pipe import PipelineModule
from fmengine.utils import logger_rank0
from deepspeed.profiling.flops_profiler import FlopsProfiler

class FMTrainer:
    def __init__(
        self,
        model: PipelineModule,
        ds_args: Dict,
        dataloader: Dict,
        ds_config: Dict,
        init_ckpt: str = None,
        save_dir: str = None,
    ) -> None:
        self.ds_args = ds_args
        self.model = model
        self.dataloader = dataloader
        self.init_ckpt = init_ckpt
        self.save_dir = save_dir
        self.ds_config = ds_config
        self.config = self.ds_config

    def fit(
        self,
        steps: int,
        profile: bool = True,
        log_per_steps: int = 10,
        save_per_steps: int = 100,
        profile_step = 10,
    ):
        wandb.init(
            # set the wandb project where this run will be logged
            project="fmengine",
            config = self.config
        )
        engine, _, _, _ = deepspeed.initialize(
            self.ds_args,
            model=self.model,
            model_parameters=[p for p in self.model.parameters() if p.requires_grad],
        )
        engine.load_checkpoint(self.init_ckpt, load_module_only=True)
        if profile:
            prof = FlopsProfiler(self.model)
        start = time.time()
        for step in range(1, steps + 1):
            if profile and step % profile_step == 0:
                prof.start_profile()
            loss = engine.train_batch(data_iter=self.dataloader)
            wandb.log({
                "loss": loss.item(),
                "lr": engine.optimizer.param_groups[0]["lr"],
                "step": step,
            })
            if self.ds_args.local_rank == 0:
                if step % log_per_steps == 0:
                    now = time.time()
                    avg_time = (now-start) / log_per_steps
                    logger_rank0.info(f"Step={step:>6}, loss={loss.item():.4f}, {avg_time:.2f} it/s")
                    start = now
                if step == profile_step:
                    prof.stop_profile()
                    prof.print_model_profile(profile_step=profile_step)
                    prof.end_profile()
            if step % save_per_steps == 0:
                logger_rank0.info(f"Saving at step {step}")
                engine.save_checkpoint(self.save_dir)
        logger_rank0.info("Finished training... saving checkpoints & closing monitoring")
        engine.save_checkpoint(self.save_dir)
        wandb.finish()
