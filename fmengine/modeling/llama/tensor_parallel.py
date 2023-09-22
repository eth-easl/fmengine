import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, ACT2FN
from transformers.models.llama.configuration_llama import LlamaConfig
import fmengine.mpu as mpu


class TensorParallelLlamaMLP(nn.Module):
    def __init__(
        self,
        args,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        no_reduce=False,
    ):
        super().__init__()
        self.gate_proj = mpu.ColumnParallelLinear(
            args=args,
            input_size=hidden_size,
            output_size=intermediate_size,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.down_proj = mpu.RowParallelLinear(
            args=args,
            input_size=intermediate_size,
            output_size=hidden_size,
            input_is_parallel=True,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            parallel_output=no_reduce,  # True if gpt-j-parallel
            bias=False,
        )
        self.up_proj = mpu.ColumnParallelLinear(
            args=args,
            input_size=hidden_size,
            output_size=intermediate_size,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)[0]) * self.up_proj(x)[0])[0]


class TensorParallelLlamaAttention(LlamaAttention):
    def __init__(self, args, config: LlamaConfig, no_reduce=False):
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
            # True if gpt-j-parallel
            parallel_output=no_reduce,
            bias=False,
        )
