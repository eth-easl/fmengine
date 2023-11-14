import wandb
import deepspeed
from typing import Dict
from deepspeed.pipe import PipelineModule
from deepspeed.profiling.flops_profiler import FlopsProfiler
from fmengine.utils import logger_rank0
from fmengine.utils.monitor import rank0_init_wandb
from fmengine.profiler.malloc import TorchTracemalloc
from timeit import default_timer as timer
from torch.profiler import profile as torch_profiler, record_function, ProfilerActivity


class LLMTrainer:
    """
    LLM Trainer
    """

    def __init__(
        self,
        model: PipelineModule,
        ds_args: Dict,
        dataloader: Dict,
        ds_config: Dict,
        init_ckpt: str = None,
        save_dir: str = None,
        pretrain: bool = False,
        dry_run: bool = False,
        load_module_strict: bool = True,
        callbacks: list = [],
    ) -> None:
        self.ds_args = ds_args
        self.model = model
        self.dataloader = dataloader
        self.init_ckpt = init_ckpt
        self.save_dir = save_dir
        self.ds_config = ds_config
        self.pretrain = pretrain
        self.dry_run = dry_run
        self.callbacks = callbacks
        self.load_module_strict = load_module_strict

    def fit(
        self,
        steps: int,
        profile: bool = True,
        save_per_steps: int = 100,
        profile_step=10,
        project="fmengine",
        configs: dict = None,
        experiment: str = None,
    ):
        rank0_init_wandb(
            # set the wandb project where this run will be logged
            project=project,
            config=configs,
            name=experiment,
        )
        engine, _, _, _ = deepspeed.initialize(
            self.ds_args,
            model=self.model,
            model_parameters=[p for p in self.model.parameters() if p.requires_grad],
        )
        if not self.pretrain:
            engine.load_checkpoint(
                self.init_ckpt,
                load_module_only=True,
                load_optimizer_states=False,
                load_module_strict=self.load_module_strict,
            )
        engine.optimizer.refresh_fp32_params()
        if profile:
            prof = FlopsProfiler(self.model)
        for step in range(1, steps + 1):
            if profile and step == profile_step:
                prof.start_profile()
            start = timer()
            with TorchTracemalloc() as tracemalloc:
                loss = engine.train_batch(data_iter=self.dataloader)
            end = timer()
            if self.ds_args.local_rank == 0:
                [cb(end - start, step, loss, configs, engine) for cb in self.callbacks]
                if profile and step == profile_step:
                    prof.stop_profile()
                    prof.print_model_profile(profile_step=step)
                    prof.end_profile()
                    del prof
            if step % save_per_steps == 0:
                logger_rank0.info(f"Saving at step {step}")
                engine.save_checkpoint(self.save_dir)
        logger_rank0.info(
            "Finished training... saving checkpoints & closing monitoring"
        )
        if not self.dry_run:
            engine.save_checkpoint(self.save_dir)
        else:
            print("Dry run, not saving checkpoint")
        wandb.finish()
