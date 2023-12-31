import torch
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer, GPTNeoXConfig

from fmengine.modeling._common._nn import EmbeddingPipe, LMLayerPipe, LayerNormPipe


class ParallelTransformerLayerPipe(GPTNeoXLayer):
    def __init__(self, config: GPTNeoXConfig, activation_checkpointing=False):
        super().__init__(config)
        self.activation_checkpointing = activation_checkpointing

    def forward(self, args):
        if self.activation_checkpointing:
            return self._ckpt_forward(args)
        hidden_states, position_ids, mask = args
        attention_mask = torch.where(mask == True, float("-inf"), 0).long()

        outputs = GPTNeoXLayer.forward(
            self, hidden_states, attention_mask, position_ids
        )
        return (outputs[0], position_ids, mask)

    def _ckpt_forward(self, args):
        hidden_states, position_ids, mask = args
        attention_mask = torch.where(mask == True, float("-inf"), 0).long()

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return GPTNeoXLayer.forward(module, *inputs)

            return custom_forward

        outputs = deepspeed.checkpointing.checkpoint(
            create_custom_forward(self),
            hidden_states,
            attention_mask,
            position_ids,
        )
        return (outputs, position_ids, mask)


class NeoxModelPipe(PipelineModule):
    def __init__(
        self,
        args,
        model_config: GPTNeoXConfig,
        activation_checkpointing_config,
        **kwargs
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
        super().__init__(
            layers=[
                LayerSpec(
                    EmbeddingPipe,
                    model_config.vocab_size,
                    model_config.hidden_size,
                ),
                *[
                    LayerSpec(
                        ParallelTransformerLayerPipe,
                        model_config,
                        activation_checkpointing_config is not None,
                    )
                    for _ in range(model_config.num_hidden_layers)
                ],
                LayerSpec(
                    LayerNormPipe,
                    model_config.hidden_size,
                ),
                LayerSpec(
                    LMLayerPipe,
                    model_config.hidden_size,
                    model_config.vocab_size,
                    bias=False,
                ),
            ],
            **kwargs
        )
