import math
from typing import Union

import torch
import torch.nn as nn

import fmengine.mpu as mpu
from transformers.models.llama.modeling_llama import LlamaAttention


class LoraParallelLinear(nn.Module):
    """
    Compatible with Tensor Parallel: initialize it with a ColumnParallelLinear or RowParallelLinear
    Both Column/RowParallelLinear is encoded in this single class by col_or_row (bool): true for column
    Only one of LoRA_A and LoRA_B is parallelized (split), the other is a regular nn.Linear

    if rank == 0 then LoRA is disabled

    TODO: lora should be float32, but the original code is probably bf16
    """

    def __init__(
        self,
        args,
        base_layer: Union[mpu.ColumnParallelLinear, mpu.RowParallelLinear],
        init_method=nn.init.xavier_normal_,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        # TODO: if we use base_layer, it is incompatible with original
        #  checkpointing save/load plan
        self.base_layer = base_layer
        self.input_size = base_layer.input_size
        self.output_size = base_layer.output_size

        # parallel settings
        self.is_para_b = False
        self.gather_output = False
        self.input_is_parallel = True
        if isinstance(base_layer, mpu.ColumnParallelLinear):
            self.is_para_b = True
            self.gather_output = base_layer.gather_output
        else:
            self.input_is_parallel = base_layer.input_is_parallel

        # lora config
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        # Mark the weight as unmerged
        self.merged = False

        # lora weights
        if r > 0:
            if self.is_para_b:  # parallel lora_B
                # lora_A (input_size, r) same copies on all ranks
                self.lora_A = nn.Linear(self.input_size, r, bias=False, dtype=torch.float32)
                # lora_B (r, output_size) split across ranks
                self.lora_B = mpu.ColumnParallelLinear(
                    args=args,
                    input_size=r,
                    output_size=self.output_size,
                    bias=False,
                    gather_output=self.gather_output,
                    init_method=init_method,
                    skip_bias_add=True,  # maybe it is not necessary
                )
            else:  # parallel lora_A
                # lora_A (input_size, r) split across ranks
                self.lora_A = mpu.RowParallelLinear(
                    args=args,
                    input_size=self.input_size,
                    output_size=r,
                    bias=False,
                    input_is_parallel=self.input_is_parallel,
                    init_method=init_method,
                    skip_bias_add=True,  # maybe it is not necessary
                )
                # lora_B (r, output_size) same copies on all ranks
                self.lora_B = nn.Linear(r, self.output_size, bias=False, dtype=torch.float32)
            self.scaling = self.lora_alpha / self.r

            # reset by default
            self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if self.r > 0:
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def merge(self):
        """
        Merges the LoRA weights into the full-rank weights (W = W + delta_W).
        """
        if self.r > 0 and not self.merged:
            # Merge the weights and mark it
            self.base_layer.weight.data += (self.lora_A @ self.lora_B) * self.scaling
            self.merged = True
            print("LoRA weights merged into the full-rank weights.")

    def forward(self, x: torch.Tensor):
        """
        if weights are merged or rank is less or equal to zero (LoRA is
        disabled) it's only a regular nn.Linear forward pass;
        otherwise in addition do the forward pass with LoRA weights
        and add it's output to the output from pretrained weights
        """
        # TODO is it possible that bias is not None?
        main_out, _ = self.base_layer(x)

        # directly return if LoRA is disabled or already merged
        if self.r == 0 or self.merged:
            return (main_out,)

        original_dtype = x.dtype
        x = x.to(self.lora_A.weight.dtype)  # should always be float32
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        if isinstance(lora_out, tuple):  # may return bias if mpu
            lora_out = lora_out[0]
        lora_out = self.lora_B(lora_out)
        if isinstance(lora_out, tuple):  # may return bias if mpu
            lora_out = lora_out[0]
        lora_out = lora_out * self.scaling
        main_out = main_out + lora_out
        # convert back to original dtype
        main_out = main_out.to(original_dtype)

        return (main_out + lora_out,)


class TensorParallelLoraAttention(LlamaAttention):
    def __init__(
        self,
        args,
        config,
    ):
        super().__init__(config)
        # we now apply lora to all linear layers in a single attention
        # https://arxiv.org/abs/2305.14314 suggests that applying lora to all layers is better than just q,v
        q_base = mpu.ColumnParallelLinear(
            args=args,
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.q_proj = LoraParallelLinear(
            args=args,
            base_layer=q_base,
            r=args.deepspeed_config.lora.r,
            lora_alpha=args.deepspeed_config.lora.lora_alpha,
            lora_dropout=args.deepspeed_config.lora.lora_dropout,
        )

        k_base = mpu.ColumnParallelLinear(
            args=args,
            input_size=self.hidden_size,
            output_size=self.num_key_value_heads * self.head_dim,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.k_proj = LoraParallelLinear(
            args=args,
            base_layer=k_base,
            r=args.deepspeed_config.lora.r,
            lora_alpha=args.deepspeed_config.lora.lora_alpha,
            lora_dropout=args.deepspeed_config.lora.lora_dropout,
        )

        v_base = mpu.ColumnParallelLinear(
            args=args,
            input_size=self.hidden_size,
            output_size=self.num_key_value_heads * self.head_dim,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
        )
        self.v_proj = LoraParallelLinear(
            args=args,
            base_layer=v_base,
            r=args.deepspeed_config.lora.r,
            lora_alpha=args.deepspeed_config.lora.lora_alpha,
            lora_dropout=args.deepspeed_config.lora.lora_dropout,
        )

        o_base = mpu.RowParallelLinear(
            args=args,
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            input_is_parallel=True,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            parallel_output=False,  # True if gpt-j-parallel
            bias=False,
        )
        self.o_proj = LoraParallelLinear(
            args=args,
            base_layer=o_base,
            r=args.deepspeed_config.lora.r,
            lora_alpha=args.deepspeed_config.lora.lora_alpha,
            lora_dropout=args.deepspeed_config.lora.lora_dropout,
        )
