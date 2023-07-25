import time
import deepspeed
from typing import Dict
from deepspeed.pipe import PipelineModule
from timeit import default_timer as timer
class FMTrainer:
    def __init__(self,
                 model: PipelineModule, 
                 ds_args: Dict,
                 dataloader: Dict,
                 init_ckpt: str = None
                ) -> None:
        
        self.ds_args = ds_args
        self.model = model
        self.dataloader = dataloader
        self.init_ckpt = init_ckpt

    def fit(self, steps: int, profile:bool=False, log_per_steps:int=10):
        engine, _, _, _ = deepspeed.initialize(
            self.ds_args,
            model=self.model,
            model_parameters=[p for p in self.model.parameters() if p.requires_grad]
        )
        engine.load_checkpoint(self.init_ckpt, load_module_only=True)
        start = timer.time()
        for step in range(1, steps + 1):
            loss = engine.train_batch(data_iter=self.dataloader)
            if self.ds_args.local_rank == 0:
                if step % log_per_steps == 0:
                    now = time.time()