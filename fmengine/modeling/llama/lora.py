# LoRA components
from torch import nn
from typing import Dict, Any
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention
from fmengine.modeling._common.lora import LoRAConfig, LoRALinear, map_old_state_dict_weights

class LoRALlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, lora_config: LoRAConfig) -> None:
        super().__init__(config)

class LoRALlamaMLP(LlamaMLP):
    def __init__(self, config: LlamaConfig, lora_config: LoRAConfig) -> None:
        nn.Module.__init__(self)
        self.gate_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )
        self.up_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )
        self.down_proj = LoRALinear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.bias,
            r=(config.r if config.to_mlp else 0),
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
        )

    def _load_from_state_dict(self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "fc_1.weight": "fc_1.linear.weight",
            "fc_1.bias": "fc_1.linear.bias",
            "fc_2.weight": "fc_2.linear.weight",
            "fc_2.bias": "fc_2.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
