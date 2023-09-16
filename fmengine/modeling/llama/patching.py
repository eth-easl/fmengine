import transformers
from fmengine.utils.monitor import rank0_print

def replace_llama_attn_with_fused_ops():
    from .fused_ops import fused_rotary_emb_llama_flash_attn_forward, init_rope
    rank0_print("[Warning] Replacing Rotary Embedding with Fused Ops. Only linear scaling is supported for now.")
    transformers.models.llama.modeling_llama.LlamaAttention._init_rope = init_rope
    transformers.models.llama.modeling_llama.LlamaAttention.forward = fused_rotary_emb_llama_flash_attn_forward

def replace_llama_attn_with_flash_attn():
    from .flash_attention import prepare_decoder_attention_mask
    from .flash_attention import llama_flash_attn_forward
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = prepare_decoder_attention_mask
 
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn_forward

def patch_llama(enable_flash_attention: bool, enable_fused_ops: bool, args):
    if enable_flash_attention:
        replace_llama_attn_with_flash_attn()
    if enable_fused_ops:
        replace_llama_attn_with_fused_ops()