import torch
import deepspeed

from .modeling_phi import PhiDecoderLayer
from .configuration_phi import PhiConfig

from deepspeed.pipe import PipelineModule, LayerSpec
from fmengine import mpu
from fmengine.modeling._common._nn import (
    ParallelEmbeddingPipe,
    LayerNormPipe,
    ParallelLMLayerPipe,
)


class ParallelTransformerLayerPipe(PhiDecoderLayer):
    def __init__(
        self, args, config: PhiConfig, activation_checkpointing=False, layer_id=0
    ):
        super().__init__(config)
        self.activation_checkpointing = activation_checkpointing
        self.layer_id = layer_id


class PhiModelPipe(PipelineModule):
    def __init__(
        self,
        args,
        model_config: PhiConfig,
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
                LayerSpec(
                    torch.nn.Dropout,
                    model_config.embd_pdrop,
                )
                * [
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
                    LayerNormPipe,
                    args,
                    model_config.hidden_size,
                    model_config.layer_norm_eps,
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
