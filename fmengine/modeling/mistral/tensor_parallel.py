import torch
import torch.nn as nn
from transformers.activations import ACT2FN
import fmengine.mpu as mpu

from .modeling_mistral import MistralFlashAttention2, MistralMLP, MistralRMSNorm
from .configuration_mistral import MistralConfig

    
class TensorParallelMistralMLP(MistralMLP):
    """Tensor Parallelism for MistralMLP layer."""
    def __init__(self, args, config: MistralConfig, no_reduce=False):
        super().__init__(config)
        self.gate_proj = mpu.ColumnParallelLinear(
            args=args,
            input_size=self.hidden_size,
            output_size=self.intermediate_size,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.up_proj = mpu.ColumnParallelLinear(
            args=args,
            input_size=self.hidden_size,
            output_size=self.intermediate_size,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.down_proj = mpu.RowParallelLinear(
            args=args,
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            input_is_parallel=True,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            parallel_output=no_reduce,
            bias=False,
        )
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)[0]) * self.up_proj(x)[0])[0]

class TensorParallelMistralFlashAttention2(MistralFlashAttention2):
    def __init__(self, args, config: MistralConfig, no_reduce=False):
        super().__init__(config)
        self.q_proj = mpu.ColumnParallelLinear(
            args=args,
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.k_proj = mpu.ColumnParallelLinear(
            args=args,
            input_size=self.hidden_size,
            output_size=self.num_key_value_heads * self.head_dim,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.v_proj = mpu.ColumnParallelLinear(
            args=args,
            input_size=self.hidden_size,
            output_size=self.num_key_value_heads * self.head_dim,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.o_proj = mpu.RowParallelLinear(
            args=args,
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            input_is_parallel=True,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            parallel_output=no_reduce,
            bias=False,
        )

class LastMistralRMSNorm(MistralRMSNorm):
    def forward(self, fw_args):
        hidden_states, *_ = fw_args
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states.to(input_dtype), )
    

