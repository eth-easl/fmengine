import torch
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaConfig,
)
from fmengine.modeling._common._nn import EmbeddingPipe, LMLayerPipe
from fmengine.modeling._common.lora import LoRAConfig, mark_only_lora_as_trainable
from fmengine.modeling.llama.lora import LoRALlamaMLP, LoRALlamaAttention

class ParallelTransformerLayerPipe(LlamaDecoderLayer):
    def __init__(self, 
                 config: LlamaConfig, activation_checkpointing=False,
                 lora_config:LoRAConfig=None):
        super().__init__(config)
        self.activation_checkpointing = activation_checkpointing
        self.lora_config = lora_config
        if self.lora_config:
            self.self_attn = LoRALlamaAttention(config, lora_config)
            self.mlp = LoRALlamaMLP(config, lora_config)
    
    def forward(self, args):
        if self.activation_checkpointing:
            return self._ckpt_forward(args)

        hidden_states, position_ids, mask = args
        attention_mask = torch.where(mask == True, float("-inf"), 0).long()

        outputs = LlamaDecoderLayer.forward(
            self,
            hidden_states,
            attention_mask,
            position_ids,
        )
        return (outputs[0], position_ids, mask)

    def _ckpt_forward(self, args):
        hidden_states, position_ids, mask = args
        attention_mask = torch.where(mask == True, float("-inf"), 0).long()

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return LlamaDecoderLayer.forward(module, *inputs)

            return custom_forward

        # deepspeed checkpoint auto use outputs[0] if len(outputs) == 1
        outputs = deepspeed.checkpointing.checkpoint(
            create_custom_forward(self),
            hidden_states,
            attention_mask,
            position_ids,
        )
        return (outputs, position_ids, mask)


class LayerNormPipe(LlamaRMSNorm):
    def forward(self, args):
        hidden_states, *_ = args
        last_hidden_states = super().forward(hidden_states)
        return (last_hidden_states,)


class LlamaModelPipe(PipelineModule):
    def __init__(self, 
                 model_config,
                 activation_checkpointing_config, 
                 lora_config: LoRAConfig,
                 **kwargs):
        if activation_checkpointing_config:
            deepspeed.checkpointing.configure(
                None,
                partition_activations=activation_checkpointing_config.get(
                    "partition_activations", False
                ),
                contiguous_checkpointing=activation_checkpointing_config.get(
                    "contiguous_memory_optimization", False
                ),
                checkpoint_in_cpu=activation_checkpointing_config.get(
                    "cpu_checkpointing", False
                ),
                num_checkpoints=activation_checkpointing_config.get(
                    "number_checkpoints", None
                ),
                synchronize=activation_checkpointing_config.get(
                    "synchronize_checkpoint_boundary", False
                ),
                profile=activation_checkpointing_config.get("profile", False),
            )
        super().__init__(
            layers=[
                LayerSpec(
                    EmbeddingPipe,
                    model_config.vocab_size + 1,
                    model_config.hidden_size,
                ),
                *[
                    LayerSpec(
                        ParallelTransformerLayerPipe,
                        model_config,
                        activation_checkpointing_config is not None,
                        lora_config=lora_config
                    )
                    for _ in range(model_config.num_hidden_layers)
                ],
                LayerSpec(
                    LayerNormPipe,
                    model_config.hidden_size,
                    model_config.rms_norm_eps,
                ),
                LayerSpec(
                    LMLayerPipe,
                    model_config.hidden_size,
                    model_config.vocab_size + 1,
                    bias=False,
                ),
            ],
            **kwargs
        )
        if lora_config:
            print(f"🌴 Low Rank Adapters Enabled: r={lora_config.r}")
            mark_only_lora_as_trainable(self, lora_config.bias)