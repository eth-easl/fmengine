import torch.nn as nn
from typing import Union, Optional
from fmengine.config import LlamaConfig, MistralConfig, ParallelismArgs
from fmengine import distributed as dist
from fmengine.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelLinearMode,
    TensorParallelRowLinear,
)
from fmengine.nn.activations import GLUActivation

class MLP(nn.Module):
    def __init__(
        self,
        config: Union[LlamaConfig, MistralConfig],
        parallel_config: Optional[ParallelismArgs],
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()

        # TODO @thomasw21: refactor so that we store that default in a single place.
        tp_mode = (
            parallel_config.tp_mode
            if parallel_config is not None
            else TensorParallelLinearMode.ALL_REDUCE
        )
        tp_linear_async_communication = (
            parallel_config.tp_linear_async_communication
            if parallel_config is not None
            else False
        )

        gate_up_contiguous_chunks = (
            config.intermediate_size,  # shape of gate_linear
            config.intermediate_size,  # shape of up_linear
        )
        self.gate_up_proj = TensorParallelColumnLinear(
            config.hidden_size,
            2 * config.intermediate_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication,
            contiguous_chunks=gate_up_contiguous_chunks,
        )

        self.down_proj = TensorParallelRowLinear(
            config.intermediate_size,
            config.hidden_size,
            pg=tp_pg,
            mode=tp_mode,
            bias=False,
            async_communication=tp_linear_async_communication
            and tp_mode is TensorParallelLinearMode.REDUCE_SCATTER,
        )
        # TODO @nouamane: why can't we torch.jit.script GLUActivation?
        self.split_silu_mul = GLUActivation(config.hidden_act)

    def forward(self, hidden_states):  # [seq_length, batch_size, hidden_dim]
        merged_states = self.gate_up_proj(hidden_states)
        hidden_states = self.down_proj(self.split_silu_mul(merged_states))
        return {"hidden_states": hidden_states}