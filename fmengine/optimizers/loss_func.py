import torch.nn.functional as F
from fmengine.fused_ops.fused_crossentropy import SoftmaxCrossEntropyLossFn

def cross_entropy_fn(outputs, labels):
    # unpack
    logits, = outputs
    # all labels are `ignore_index` will cause nan
    return SoftmaxCrossEntropyLossFn(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
    )