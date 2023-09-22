import torch.nn.functional as F
import torch

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
    print("⚡⚡⚡ [Flash Attention] fused cross entropy enabled")
except ImportError:
    CrossEntropyLoss = torch.nn.CrossEntropyLoss

def cross_entropy_fn(outputs, labels):
    # unpack
    (logits,) = outputs
    # all labels are `ignore_index` will cause nan
    return CrossEntropyLoss()(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
    )
