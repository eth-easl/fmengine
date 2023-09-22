import torch
from typing import Optional, Tuple
from flash_attn.flash_attn_interface import flash_attn_kvpacked_func
from fmengine.modeling.llama.rotary_embedding import RotaryEmbedding


def init_rope(self):
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


def fused_rotary_emb_llama_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)[0].view(bsz, q_len, -1, self.head_dim)
    key_states = self.k_proj(hidden_states)[0].view(bsz, q_len, -1, self.head_dim)
    value_states = self.v_proj(hidden_states)[0].view(bsz, q_len, -1, self.head_dim)

    q = query_states

    kv = torch.stack([key_states, value_states], dim=2)
    q, kv = self.rotary_emb(q, kv)

    attn_output = flash_attn_kvpacked_func(
        q,
        kv,
        0.0,
        causal=True,
    )
    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)[0]

    return attn_output, None, None
