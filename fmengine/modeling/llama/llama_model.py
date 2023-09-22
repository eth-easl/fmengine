import torch
import deepspeed
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaConfig,
)
from deepspeed.pipe import PipelineModule, LayerSpec
from fmengine.modeling._common._nn import (
    EmbeddingPipe,
    LMLayerPipe,
    ParallelEmbeddingPipe,
    ParallelLMLayerPipe,
)
from fmengine.modeling._common.lora import LoRAConfig, mark_only_lora_as_trainable
from fmengine.modeling.llama.lora import LoRALlamaMLP, LoRALlamaAttention
from fmengine.modeling.llama.tensor_parallel import (
    TensorParallelLlamaAttention,
    TensorParallelLlamaMLP,
)


class ParallelTransformerLayerPipe(LlamaDecoderLayer):
    def __init__(
        self,
        args,
        config: LlamaConfig,
        activation_checkpointing=False,
        lora_config: LoRAConfig = None,
        layer_id=0,
    ):
        super().__init__(config)
        self.activation_checkpointing = activation_checkpointing
        self.lora_config = lora_config
        self.layer_id = layer_id
        if self.lora_config:
            self.self_attn = LoRALlamaAttention(config, lora_config)
            self.mlp = LoRALlamaMLP(config, lora_config)
            mark_only_lora_as_trainable(self, lora_config.bias)

        self.self_attn = TensorParallelLlamaAttention(args, config)
        self.mlp = TensorParallelLlamaMLP(
            args, config.hidden_size, config.intermediate_size, config.hidden_act
        )

        def mlp_res(hidden_states: torch.Tensor) -> torch.Tensor:
            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

        def attn_res(hidden_states: torch.Tensor) -> torch.Tensor:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            # Self Attention
            hidden_states, _, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
            )
            hidden_states = residual + hidden_states
            return hidden_states

        self.attn_res = attn_res
        self.mlp_res = mlp_res

    def forward(self, args):
        x, position_ids, mask = args
        attention_mask = None

        if self.activation_checkpointing:
            x.requires_grad_(True)
            x = deepspeed.checkpointing.checkpoint(self.attn_res, x)
        else:
            x = self.attn_res(x, attention_mask)

        if self.activation_checkpointing:
            x.requires_grad_(True)
            x = deepspeed.checkpointing.checkpoint(self.mlp_res, x)
        else:
            x = self.mlp_res(x)

        return (x, position_ids, mask)


class LayerNormPipe(LlamaRMSNorm):
    def forward(self, args):
        hidden_states, *_ = args
        last_hidden_states = super().forward(hidden_states)
        return (last_hidden_states,)


class LlamaModelPipe(PipelineModule):
    def __init__(
        self,
        args,
        model_config,
        activation_checkpointing_config,
        lora_config: LoRAConfig,
        **kwargs,
    ):
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
        tensor_parallel_enabled = args.model_parallel_size > 1

        if tensor_parallel_enabled:
            embedding_pipe = LayerSpec(
                ParallelEmbeddingPipe,
                args,
                model_config.vocab_size,
                model_config.hidden_size,
            )
            lmlayer_pipe = LayerSpec(
                ParallelLMLayerPipe,
                args,
                model_config.hidden_size,
                model_config.vocab_size,
                bias=False,
            )
        else:
            embedding_pipe = LayerSpec(
                EmbeddingPipe, model_config.vocab_size, model_config.hidden_size
            )
            lmlayer_pipe = LayerSpec(
                LMLayerPipe,
                model_config.hidden_size,
                model_config.vocab_size,
                bias=False,
            )

        super().__init__(
            layers=[
                embedding_pipe,
                *[
                    LayerSpec(
                        ParallelTransformerLayerPipe,
                        args,
                        model_config,
                        activation_checkpointing_config is not None,
                        lora_config=lora_config,
                        layer_id=layer_id,
                    )
                    for layer_id in range(model_config.num_hidden_layers)
                ],
                LayerSpec(
                    LayerNormPipe,
                    model_config.hidden_size,
                    model_config.rms_norm_eps,
                ),
                lmlayer_pipe,
            ],
            **kwargs,
        )
        if lora_config:
            print(f"🌴 Low Rank Adapters Enabled: r={lora_config.r}")
