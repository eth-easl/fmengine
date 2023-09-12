from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention
import torch.nn as nn
import fmengine.mpu as mpu

def get_tp_llama_attention(args: dict):
    class TensorParallelLlamaAttention(LlamaAttention):
        def __init__(self, config: LlamaConfig):
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
                output_size=self.num_kv_heads * self.head_dim,
                gather_output=False,
                init_method=nn.init.xavier_normal_,
                skip_bias_add=True,
                bias=False,
            )
            self.v_proj = mpu.ColumnParallelLinear(
                args=args,
                input_size=self.hidden_size,
                output_size=self.num_kv_heads * self.head_dim,
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
                parallel_output=False, # True if gpt-j-parallel
                bias=False,
            )
    return TensorParallelLlamaAttention