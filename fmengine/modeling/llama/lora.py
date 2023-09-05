# LoRA components
import torch
from typing import Dict, Any
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention
from fmengine.modeling._common.lora import LoRAConfig, LoRALinear, map_old_state_dict_weights

class LoRALlamaMLP(LlamaMLP):
    def __init__(self, config: LlamaConfig, lora_config: LoRAConfig) -> None:
        super().__init__(config)
        self.gate_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            r=(lora_config.r if lora_config.to_mlp else 0),
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
        )
        self.up_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            r=(lora_config.r if lora_config.to_mlp else 0),
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
        )
        self.down_proj = LoRALinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            r=(lora_config.r if lora_config.to_mlp else 0),
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
        )

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "gate_proj.weight": "gate_proj.linear.weight",
            "gate_proj.bias": "gate_proj.linear.bias",
            "up_proj.weight": "up_proj.linear.weight",
            "up_proj.bias": "up_proj.linear.bias",
            "down_proj.weight": "down_proj.linear.weight",
            "down_proj.bias": "down_proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class LoRALlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, lora_config: LoRAConfig) -> None:
        super().__init__(config)
        self.lora_config = lora_config
        self.q_proj = LoRALinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            r = lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            bias=False
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            r = lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            bias=False
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            r = lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            bias=False
        )
        self.o_proj = LoRALinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            r = lora_config.r,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            bias=False
        )

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "q_proj.weight": "q_proj.linear.weight",
            "q_proj.bias": "q_proj.linear.bias",
            "k_proj.weight": "k_proj.linear.weight",
            "k_proj.bias": "k_proj.linear.bias",
            "v_proj.weight": "v_proj.linear.weight",
            "v_proj.bias": "v_proj.linear.bias",
            "o_proj.weight": "o_proj.linear.weight",
            "o_proj.bias": "o_proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        # initialize LoRA weights
        state_dict['self_attn.q_proj.lora_A'] = torch.nn.Parameter(torch.zeros((self.lora_config.r, self.config.hidden_size)))
        state_dict['self_attn.q_proj.lora_B'] = torch.nn.Parameter(torch.zeros((self.num_key_value_heads * self.head_dim, self.lora_config.r)))
        state_dict['self_attn.k_proj.lora_A'] = torch.nn.Parameter(torch.zeros((self.lora_config.r, self.config.hidden_size)))
        state_dict['self_attn.k_proj.lora_B'] = torch.nn.Parameter(torch.zeros((self.num_key_value_heads * self.head_dim, self.lora_config.r)))
        state_dict['self_attn.v_proj.lora_A'] = torch.nn.Parameter(torch.zeros((self.lora_config.r, self.config.hidden_size)))
        state_dict['self_attn.v_proj.lora_B'] = torch.nn.Parameter(torch.zeros((self.num_key_value_heads * self.head_dim, self.lora_config.r)))
        state_dict['self_attn.o_proj.lora_A'] = torch.nn.Parameter(torch.zeros((self.lora_config.r, self.num_heads * self.head_dim)))
        state_dict['self_attn.o_proj.lora_B'] = torch.nn.Parameter(torch.zeros((self.hidden_size, self.lora_config.r)))
        self.rotary_emb.inv_freq = state_dict['self_attn.rotary_emb.inv_freq']

        del state_dict['self_attn.rotary_emb.inv_freq']

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
