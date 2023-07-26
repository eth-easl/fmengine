import time
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
        init_ckpt: str = None,
        save_dir: str = None,
    ) -> None:
        self.ds_args = ds_args
        self.model = model
        self.dataloader = dataloader
        self.init_ckpt = init_ckpt
        self.save_dir = save_dir
    
    def fit(
        self,
        steps: int,
        profile: bool = True,
        log_per_steps: int = 10,
        save_per_steps: int = 100,
        profile_step = 10,
    ):
        engine, _, _, _ = deepspeed.initialize(
            self.ds_args,
            model=self.model,
            model_parameters=[p for p in self.model.parameters() if p.requires_grad],
        )
        engine.load_checkpoint(self.init_ckpt, load_module_only=True)
        #ds_iter = iter(self.dataloader)
        if profile:
            prof = FlopsProfiler(self.model)
        start = time.time()
        for step in range(1, steps + 1):
            if profile and step % profile_step == 0:
                prof.start_profile()
            loss = engine.train_batch(data_iter=self.dataloader)
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