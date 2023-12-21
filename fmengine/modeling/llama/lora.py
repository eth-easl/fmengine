import math
from typing import Union

import torch
import torch.nn as nn

import fmengine.mpu as mpu
from transformers.models.llama.modeling_llama import LlamaAttention


class LoRARowParallelLinear(mpu.ColumnParallelLinear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        args,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=nn.init.xavier_normal_,
        # only for lora
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            args,
            input_size,
            output_size,
            bias=bias,
            gather_output=gather_output,
            init_method=init_method,
            **kwargs,
        )

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output

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
            self.weight.data += (self.lora_A.weight @ self.lora_B.weight) * self.scaling
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
        main_out, _ = super().forward(x)

        # directly return if LoRA is disabled or already merged
        if self.r == 0 or self.merged:
            return (main_out,)

        original_dtype = x.dtype
        x = x.to(self.lora_A.weight.dtype)  # should always be float32
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        lora_out, _ = self.lora_B(lora_out)  # mpu return tuple because of bias
        lora_out = lora_out * self.scaling
        main_out = main_out + lora_out
        # convert back to original dtype
        main_out = main_out.to(original_dtype)

        return main_out + lora_out


class LoRARowParallelLinear(mpu.RowParallelLinear):
    """
    lora_A: RowParallelLinear
    lora_B: Linear

    if rank == 0 then LoRA is disabled

    TODO: lora should be float32, but the original code is probably bf16
    """

    def __init__(
        self,
        args,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        parallel_output=False,
        init_method=nn.init.xavier_normal_,
        # only for lora
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            args,
            input_size,
            output_size,
            bias=bias,
            input_is_parallel=input_is_parallel,
            init_method=init_method,
            parallel_output=parallel_output,
            **kwargs,
        )

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.parallel_output = parallel_output

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
        """
        Reset all the weights, even including pretrained ones.
        TODO: initialize parallel weights is indeed should not be the same as nn.Linear
        """
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
            self.weight.data += (self.lora_A @ self.lora_B) * self.scaling
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
        main_out, _ = super().forward(x)

        # directly return if LoRA is disabled or already merged
        if self.r == 0 or self.merged:
            return (main_out,)

        original_dtype = x.dtype
        x = x.to(self.lora_A.weight.dtype)  # should always be float32
        lora_out = self.lora_dropout(x)
        lora_out, _ = self.lora_A(lora_out)  # mpu return tuple because of bias
        lora_out = self.lora_B(lora_out)
        lora_out = lora_out * self.scaling
        main_out = main_out + lora_out
        # convert back to original dtype
        main_out = main_out.to(original_dtype)

        return main_out + lora_out


class TensorParallelLoraAttention(LlamaAttention):
    def __init__(
        self,
        args,
        config,
    ):
        super().__init__(config)
        # we now apply lora to all linear layers in a single attention
        # https://arxiv.org/abs/2305.14314 suggests that applying lora to all layers is better than just q,v
        self.q_proj = LoRARowParallelLinear(
            args=args,
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            r=args.deepspeed_config.lora.r,
            lora_alpha=args.deepspeed_config.lora.lora_alpha,
            lora_dropout=args.deepspeed_config.lora.lora_dropout,
            skip_bias_add=True,
        )

        self.k_proj = LoRARowParallelLinear(
            args=args,
            input_size=self.hidden_size,
            output_size=self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
            r=args.deepspeed_config.lora.r,
            lora_alpha=args.deepspeed_config.lora.lora_alpha,
            lora_dropout=args.deepspeed_config.lora.lora_dropout,
        )
        self.v_proj = LoRARowParallelLinear(
            args=args,
            input_size=self.hidden_size,
            output_size=self.num_key_value_heads * self.head_dim,
            gather_output=False,
            init_method=nn.init.xavier_normal_,
            skip_bias_add=True,
            bias=False,
            r=args.deepspeed_config.lora.r,
            lora_alpha=args.deepspeed_config.lora.lora_alpha,
            lora_dropout=args.deepspeed_config.lora.lora_dropout,
            skip_bias_add=True,
        )
        self.o_proj = LoRARowParallelLinear(
            args=args,
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=nn.init.xavier_normal_,
            r=args.deepspeed_config.lora.r,
            lora_alpha=args.deepspeed_config.lora.lora_alpha,
            lora_dropout=args.deepspeed_config.lora.lora_dropout,
            skip_bias_add=True,
            parallel_output=False,  # True if gpt-j-parallel
            bias=False,
            r=args.deepspeed_config.lora.r,
            lora_alpha=args.deepspeed_config.lora.lora_alpha,
            lora_dropout=args.deepspeed_config.lora.lora_dropout,
        )
