import transformers
from fmengine.utils.monitor import rank0_print
from fmengine.modeling.llama.rotary_embedding import RotaryEmbedding

def _init_rope(self):
    if self.config.rope_scaling is None:
        scaling_factor = max(self.max_position_embeddings / 4096, 1.0)
    else:
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]
        assert scaling_type == "linear", "Only linear scaling is supported for now"
    if self.config.rope_theta is None:
        print("Warning: rope_theta is None. Using default value of 10000.")
        self.config.rope_theta = 10000
    
    self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            base=self.config.rope_theta,
            interleaved=False,
            scaling_factor=scaling_factor,
        )

def replace_llama_attn_with_fused_ops():
    rank0_print("[Warning] Replacing Rotary Embedding with Fused Ops. Only linear scaling is supported for now.")
    transformers.models.llama.modeling_llama.LlamaAttention._init_rope = _init_rope