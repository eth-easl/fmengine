import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
from typing import Any, Dict, Tuple, Union, Mapping


class LoRALayer(nn.Module):
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float):
        """Store LoRA specific attributes in a class.
        Args:
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__()
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


class LoRALinear(LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        """LoRA wrapper around linear class.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        """
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.linear.weight.new_zeros((out_features, r)))
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
        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        lora = (
            self.lora_dropout(x)
            @ self.lora_A.transpose(0, 1)
            @ self.lora_B.transpose(0, 1)
        ) * self.scaling
        return pretrained + lora


class LoRAQKVLinear(LoRALinear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        n_head: int,
        n_query_groups: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: Union[bool, Tuple[bool, bool, bool]] = False,
        **kwargs,
    ):
        """LoRA wrapper around linear class that is used for calculation of q, k and v matrices.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            n_head: number of attention heads
            n_query_groups: number of query groups (see diagram in `lit_gpt/config.py`)
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            enable_lora: MergeLinear class is for attention mechanism where qkv are calculated with a single weight matrix. If we
                don't want to apply LoRA we can set it as False. For example if we want to apply LoRA only to `query`
                and `value` but keep `key` without weight updates we should pass `[True, False, True]`
        """
        super(LoRALinear, self).__init__(
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        if isinstance(enable_lora, bool):
            enable_lora = [enable_lora] * 3
        assert len(enable_lora) == 3
        self.enable_lora = enable_lora

        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ r: 2
        # ⚬ enable_lora: [True, False, True]
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.linear.weight.new_zeros((r * sum(enable_lora), in_features))
            )  # (4, 128)
            enable_q, enable_k, enable_v = enable_lora
            self.kv_embd_size = self.linear.in_features // (n_head // n_query_groups)
            # qkv_shapes will be used to split a tensor with weights correctly
            qkv_shapes = (
                self.linear.in_features * enable_q,
                self.kv_embd_size * enable_k,
                self.kv_embd_size * enable_v,
            )
            self.qkv_shapes = [s for s in qkv_shapes if s]
            self.lora_B = nn.Parameter(
                self.linear.weight.new_zeros(sum(self.qkv_shapes), r)
            )  # (256, 2))
            # Notes about shapes above
            # - self.lora_A has shape (4, 128): 4 because rank is 2 and LoRA is applied only to two matrices;
            # 128 is the input size of the x (embedding size). (4, 128) and not (128, 4) because later on in
            # F.linear function weights are automatically transposed. In addition conv1d requires channels to
            # be before seq length
            # - self.lora_B has shape (256, 2): 256 because LoRA is applied only to two matrices, so the output is
            # 128*2; 2 tells to have two channels per group for group convolution

            # Scaling:
            # This balances the pretrained model`s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/
            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora
            self.scaling = self.lora_alpha / self.r

            # Compute the indices
            # Indices are needed to properly pad weight updates with zeros. If we want to fine-tune queries and values,
            # but not keys, then the weights update should be:
            #
            # [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
            #  [....................................],
            #  [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            #      ↑              ↑            ↑
            # ________________________________________
            # | query         | key       | value    |
            # ----------------------------------------
            self.lora_ind = []
            if enable_q:
                self.lora_ind.extend(range(0, self.linear.in_features))
            if enable_k:
                self.lora_ind.extend(
                    range(
                        self.linear.in_features,
                        self.linear.in_features + self.kv_embd_size,
                    )
                )
            if enable_v:
                self.lora_ind.extend(
                    range(
                        self.linear.in_features + self.kv_embd_size,
                        self.linear.out_features,
                    )
                )
            self.reset_parameters()

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Properly pad weight updates with zeros.

        If, based on `self.enable_lora`, we want to fine-tune queries and values, but not keys,
        then the weights update should be:

        [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
         [....................................],
         [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            ↑              ↑            ↑
        ________________________________________
        | query         | key       | value    |
        ----------------------------------------

        Args:
            x: tensor with weights update that will be padded with zeros if necessary

        Returns:
            A tensor with weight updates and zeros for deselected q, k or v
        """
        # we need to do zero padding only if LoRA is disabled for one of QKV matrices
        if all(self.enable_lora):
            return x

        # Let's image that:
        # ⚬ input x has shape (64, 64, 256): (batch_size, sequence_length, embeddings_size)
        # ⚬ embeddings_size: 128
        # ⚬ self.linear.out_features: 384 (3 * embeddings_size)
        # ⚬ enable_lora: [True, False, True]
        # Then x has embeddings_size of 256 (2 * 128 as enable_lora only for query and value, not keys) and expected
        # embeddings_size is 384 (self.linear.out_features), so that means that we need to pad from 256 to 384 with zeros, but
        # only for key updates (this is where self.lora_ind comes in handy)
        # Note: double transpose (in the beginning and in the end) is basically a guard for two-dimensional tensors
        # for example when we want to merge/unmerge LoRA weights and pretrained weights
        x = x.transpose(0, 1)
        result = x.new_zeros((*x.shape[:-1], self.linear.out_features))  # (64, 64, 384)
        result = result.view(-1, self.linear.out_features)  # (4096, 384)
        result = result.index_copy(
            1,
            torch.tensor(self.lora_ind, device=result.device),
            x.reshape(-1, sum(self.qkv_shapes)),
        )  # (4096, 256)
        return result.view((*x.shape[:-1], self.linear.out_features)).transpose(
            0, 1
        )  # (64, 64, 384)

    def conv1d(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """An extension of the `torch.nn.functional.conv1d` function with a logic specific to grouped queries.

        If the number of heads is equal to the number of query groups - grouped queries are disabled
        (see scheme in `lit_gpt/config.py:Config`). In this case the combined QKV matrix consists of equally sized
        query, key and value parts, which means we can utilize `groups` argument from `conv1d`: with this argument the
        input and weight matrices will be splitted in equally sized parts and applied separately (like having multiple
        conv layers side by side).

        Otherwise QKV matrix consists of unequally sized parts and thus we have to split input and weight matrices manually,
        apply each part of the weight matrix to the corresponding input's part and concatenate the result.

        Args:
            input: input matrix of shape (B, C, T)
            weight: weight matrix of shape (C_output, rank, 1).
                "C_output" is defined as a sum of embedding sizes for each enabled LoRA layer (see init method of the class).

        Returns:
            A tensor with a shape (B, C_output, T)
        """
        if self.n_head == self.n_query_groups:
            return F.conv1d(
                input, weight, groups=sum(self.enable_lora)
            )  # (B, C_output, T)

        # Notation:
        # ⚬ N: number of enabled LoRA layers (self.enable_lora)
        # ⚬ C_output': embeddings size for each LoRA layer (not equal in size)
        # ⚬ r: rank of all LoRA layers (equal in size)

        input_splitted = input.chunk(sum(self.enable_lora), dim=1)  # N * (B, C // N, T)
        weight_splitted = weight.split(self.qkv_shapes)  # N * (C_output', r, 1)
        return torch.cat(
            [F.conv1d(a, b) for a, b in zip(input_splitted, weight_splitted)],
            dim=1,  # (B, C_output', T)
        )  # (B, C_output, T)

    def merge(self):
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""

        # Let's assume that:
        # ⚬ self.linear.weight.data: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)
        if self.r > 0 and any(self.enable_lora) and not self.merged:
            delta_w = self.conv1d(
                self.lora_A.data.unsqueeze(0),  # (4, 128) -> (1, 4, 128)
                self.lora_B.data.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
            ).squeeze(
                0
            )  # (1, 4, 128) @ (256, 2, 1) -> (1, 256, 128) -> (256, 128)
            # W = W + delta_W (merge)
            self.linear.weight.data += self.zero_pad(
                delta_w * self.scaling
            )  # (256, 128) after zero_pad (384, 128)
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass.

        If LoRA's weights are merged with pretrained ones then it's a simple matrix multiplication.
        If not, then multiply pretrained weights with input, apply LoRA on input and do summation.

        Args:
            x: input tensor of shape (batch_size, context_length, embedding_size)

        Returns:
            Output tensor of shape (batch_size, context_length, 3 * embedding_size)
        """

        # Let's assume that:
        # ⚬ x: (64, 64, 128) or (batch_size, context_length, embedding_size)
        # ⚬ self.linear.weight: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)

        # if weights are merged or LoRA is disabled (r <= 0 or all `enable_lora` are False) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or not any(self.enable_lora) or self.merged:
            return pretrained
        after_A = F.linear(
            self.lora_dropout(x), self.lora_A
        )  # (64, 64, 128) @ (4, 128) -> (64, 64, 4)
        # For F.conv1d:
        # ⚬ input: input tensor of shape (mini-batch, in_channels, iW)
        # ⚬ weight: filters of shape (out_channels, in_channels/groups, kW)
        after_B = self.conv1d(
            after_A.transpose(-2, -1),  # (64, 64, 4) -> (64, 4, 64)
            self.lora_B.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
        ).transpose(
            -2, -1
        )  # (64, 4, 64) @ (256, 2, 1) -> (64, 256, 64) -> (64, 64, 256)
        lora = (
            self.zero_pad(after_B) * self.scaling
        )  # (64, 64, 256) after zero_pad (64, 64, 384)
        return pretrained + lora


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Args:
        model: model with LoRA layers
        bias:
            ``"none"``: all bias weights will be frozen,
            ``"lora_only"``: only bias weight for LoRA layers will be unfrozen,
            ``"all"``: all bias weights will be unfrozen.

    Raises:
        NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True

    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise ValueError(
            f"bias should be one of ['none', 'lora_only', 'all'], got {bias}"
        )


def lora_filter(key: str, value: Any) -> bool:
    return "lora_" in key


@dataclass
class LoRAConfig:
    """
    Args:
        r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        to_*: either apply LoRA to the specified weights or not
    """

    r: int = 0
    alpha: int = 1
    dropout: float = 0.0
    to_query: bool = False
    to_key: bool = False
    to_value: bool = False
    to_projection: bool = False
    to_mlp: bool = False
    to_head: bool = False
    bias: Union[str, bool] = "none"


def map_old_state_dict_weights(state_dict: Dict, mapping: Mapping, prefix: str) -> Dict:
    for checkpoint_name, attribute_name in mapping.items():
        full_checkpoint_name = prefix + checkpoint_name
        if full_checkpoint_name in state_dict:
            full_attribute_name = prefix + attribute_name
            state_dict[full_attribute_name] = state_dict.pop(full_checkpoint_name)
    return state_dict
