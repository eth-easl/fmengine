GPU_AVAILABLE_FLOPS = {
    # source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    # nvidia publishes spec sheet with a 2x sparsity factor
    "h100-sxm": {
        "64-true": 67e12,
        "32-true": 67e12,
        "16-true": 1.979e15 / 2,
        "16-mixed": 1.979e15 / 2,
        "bf16-true": 1.979e15 / 2,
        "bf16-mixed": 1.979e15 / 2,
        "8-true": 3.958e15 / 2,
        "8-mixed": 3.958e15 / 2,
    },
    "h100-pcie": {
        "64-true": 51e12,
        "32-true": 51e12,
        "16-true": 1.513e15 / 2,
        "16-mixed": 1.513e15 / 2,
        "bf16-true": 1.513e15 / 2,
        "bf16-mixed": 1.513e15 / 2,
        "8-true": 3.026e15 / 2,
        "8-mixed": 3.026e15 / 2,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    "a100": {
        "64-true": 19.5e12,
        "32-true": 19.5e12,
        "16-true": 312e12,
        "16-mixed": 312e12,
        "bf16-true": 312e12,
        "bf16-mixed": 312e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
    "a10g": {
        "32-true": 31.2e12,
        "16-true": 125e12,
        "16-mixed": 125e12,
        "bf16-true": 125e12,
        "bf16-mixed": 125e12,
    },
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    "v100-sxm": {
        "64-true": 7.8e12,
        "32-true": 15.7e12,
        "16-true": 125e12,
        "16-mixed": 125e12,
    },
    "v100-pcie": {
        "64-true": 7e12,
        "32-true": 14e12,
        "16-true": 112e12,
        "16-mixed": 112e12,
    },
    "v100s-pcie": {
        "64-true": 8.2e12,
        "32-true": 16.4e12,
        "16-true": 130e12,
        "16-mixed": 130e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    # sxm and pcie have same flop counts
    "t4": {
        "32-true": 8.1e12,
        "16-true": 65e12,
        "16-mixed": 65e12,
        "8-true": 130e12,
        "int4": 260e12,
    },
    # https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-5000-data-sheet-us-nvidia-704120-r4-web.pdf
    "quadro rtx 5000": {"32-true": 11.2e12, "16-true": 89.2e12, "16-mixed": 89.2e12},
}

TPU_AVAILABLE_FLOPS = {
    # flop count for each TPU generation is the same for all precisions
    # since bfloat16 precision is always used for performing matrix operations
    # for more info: https://cloud.google.com/tpu/docs/bfloat16#choosing_bfloat16
    # source: https://arxiv.org/pdf/1907.10701.pdf
    "v2": 45e12,
    # source: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v3
    "v3": 123e12,
    # source: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm#tpu_v4
    "v4": 275e12,
}
