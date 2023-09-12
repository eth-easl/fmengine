import torch
import fmengine.mpu as mpu

def initialize_megatron(args, fp32_allreduce=False):

    device_count = torch.cuda.device_count()
    assert torch.distributed.is_initialized()

    # Setup 3D topology.
    pp = args.pipe_parallel_size if args.pipe_parallel_size >= 1 else 1
    mp = args.model_parallel_size if args.model_parallel_size >= 1 else 1
    assert (
        args.world_size % (pp * mp) == 0
    ), f"world_size={args.world_size}, pp={pp}, mp={mp}"
    dp = args.world_size // (pp * mp)

    from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

    # this does pipe on the most outside, then data, then model.
    # PipeModelDataParallelTopology is just a wrapper over ProcessTopology that predefines this order.
    topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=mp, num_dp=dp)

    # Offset base seeds for the interior pipeline stages.
    # TODO: adjust last stage too once IO is improved.
    stage_id = topo.get_coord(rank=torch.distributed.get_rank()).pipe
    if 0 < stage_id < topo.get_dim("pipe") - 1:
        offset = args.seed + 1138
        args.seed = offset + (stage_id * mp)

    # Set the model-parallel / data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print(
                "_initialize_distributed() model parallel is already initialized",
                flush=True,
            )
        else:
            mpu.initialize_model_parallel(
                args.model_parallel_size,
                topology=topo,
                fp32_allreduce=fp32_allreduce,
            )
