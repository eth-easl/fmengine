import torch
import deepspeed

from .modeling_mistral import MistralDecoderLayer
from .configuration_mistral import MistralConfig

from deepspeed.pipe import PipelineModule, LayerSpec
from fmengine.modeling._common._nn import ParallelEmbeddingPipe, ParallelLMLayerPipe
from fmengine.modeling.mistral.lora import TensorParallelLoraAttention
from fmengine.modeling.mistral.tensor_parallel import (
    TensorParallelMistralMLP,
    TensorParallelMistralFlashAttention2,
    LastMistralRMSNorm
)
from fmengine import mpu

from .tensor_parallel import LastMistralRMSNorm


class ParallelTransformerLayerPipe(MistralDecoderLayer):
    def __init__(
        self,
        args, 
        config: MistralConfig,
        activation_checkpointing=False,
        layer_id=0,
    ):
        super().__init__(config)
        self.activation_checkpointing = activation_checkpointing
        self.layer_id = layer_id
        if "lora" in args.deepspeed_config:
            self.self_attn = TensorParallelLoraAttention(args, config)
            print(f"🌴 Low Rank Adapters Enabled: r={args.deepspeed_config.lora.r}")
        else:
            self.self_attn = TensorParallelMistralFlashAttention2(args, config)
        self.mlp = TensorParallelMistralMLP(args, config)
        
        def mlp_res(hidden_states: torch.Tensor) -> torch.Tensor:
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

        def attn_res(hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=position_ids,
            )
            hidden_states = residual + hidden_states
            return hidden_states
    
        self.attn_res = attn_res
        self.mlp_res = mlp_res
    
    def forward(self, args):
        x, position_ids, mask = args
        attention_mask = None
        
        if position_ids is None:
            position_ids = torch.arange(
                0, x.size(-2), dtype=torch.long, device=x.device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, x.size(-2))

        if self.activation_checkpointing:
            x.requires_grad_(True)
            x = deepspeed.checkpointing.checkpoint(self.attn_res, x, position_ids)
        else:
            x = self.attn_res(x, attention_mask, position_ids)

        if self.activation_checkpointing:
            x.requires_grad_(True)
            x = deepspeed.checkpointing.checkpoint(self.mlp_res, x)
        else:
            x = self.mlp_res(x)

        return (x, position_ids, mask)

class MistralModelPipe(PipelineModule):
    def __init__(
        self,
        args,
        model_config: MistralConfig,
        activation_checkpointing_config,
        **kwargs,
    ):
        if activation_checkpointing_config:
            deepspeed.checkpointing.configure(
                mpu,
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
                    ParallelEmbeddingPipe,
                    args,
                    model_config.vocab_size,
                    model_config.hidden_size,
                ),
                *[
                    LayerSpec(
                        ParallelTransformerLayerPipe,
                        args,
                        model_config,
                        activation_checkpointing_config is not None,
                        layer_id=layer_id,
                    )
                    for layer_id in range(model_config.num_hidden_layers)
                ],
                LayerSpec(
                    LastMistralRMSNorm,
                    model_config.hidden_size,
                    model_config.rms_norm_eps,
                ),
                LayerSpec(
                    ParallelLMLayerPipe,
                    args,
                    model_config.hidden_size,
                    model_config.vocab_size,
                    bias=False,
                ),
            ],
            **kwargs,
        )
