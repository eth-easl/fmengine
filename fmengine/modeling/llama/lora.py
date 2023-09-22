import math
import torch
import torch.nn as nn
from fmengine.modeling.llama.llama_model import TensorParallelLlamaAttention
import fmengine.mpu as mpu
import torch.nn.functional as F


class LoRARowParallelLinear(mpu.ColumnParallelLinear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        args,
        # ↓ this part is for pretrained ColumnParallelLinear weights
        input_size: int,
        output_size: int,
        gather_output=False,
        init_method=nn.init.xavier_normal_,
        skip_bias_add=True,
        bias=False,
        # ↓ the remaining part is for LoRA
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            args=args,
            input_size=input_size,
            output_size=output_size,
            gather_output=gather_output,
            init_method=init_method,
            skip_bias_add=skip_bias_add,
            bias=bias,
        )
        assert gather_output == False
        assert r >= 0
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, self.weight.size(1))))
            self.lora_B = nn.Parameter(self.weight.new_zeros((self.weight.size(0)), r))
            self.scaling = self.lora_alpha / self.r
            self.reset_parameters()

    def reset_parameters(self):
        """Reset all the weights, even including pretrained ones."""
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and not self.merged:
            # Merge the weights and mark it
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        # if weights are merged or rank is less or equal to zero (LoRA is disabled) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = super().forward(x)[0]
        if self.r == 0 or self.merged:
            return (pretrained,)
        x = self.lora_dropout(x)
        x = F.linear(x, self.lora_A)
        x = F.linear(x, self.lora_B)
        x = x * self.scaling
        return (pretrained + x,)
