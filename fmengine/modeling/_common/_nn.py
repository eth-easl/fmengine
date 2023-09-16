import torch
from torch.nn import LayerNorm
import fmengine.mpu as mpu

class EmbeddingPipe(torch.nn.Embedding):
    def forward(self, args):
        input_ids, position_ids, attention_mask = args
        inputs_embeds = super().forward(input_ids)
        return (inputs_embeds, position_ids, attention_mask)

class LMLayerPipe(torch.nn.Linear):
    def forward(self, args):
        hidden_states, = args
        logits = super().forward(hidden_states)
        return (logits,)
    
class LayerNormPipe(LayerNorm):
    def forward(self, args):
        hidden_states, *_ = args
        last_hidden_states = super().forward(hidden_states)
        return (last_hidden_states,)

class ParallelEmbeddingPipe(mpu.layers.VocabParallelEmbedding):
    def forward(self, args):
        input_ids, position_ids, attention_mask = args
        inputs_embeds = super().forward(input_ids)
        return (inputs_embeds, position_ids, attention_mask)

class ParallelLMLayerPipe(mpu.ColumnParallelLinear):
    def forward(self, args):
        hidden_states, = args
        logits = super().forward(hidden_states)[0]
        return (logits,)