import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXConfig
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology

from fmengine.optimizers.loss_func import cross_entropy_fn
from fmengine.modeling.llama.llama_model import LlamaModelPipe
from fmengine.modeling.neox.neox_model import NeoxModelPipe
from fmengine.modeling.mistral.mistral_model import MistralModelPipe, MistralConfig
#from fmengine.modeling.mistral.flacon_model import FalconModelPipe, FalconConfig
_SEQUENCE_PARALLEL_GROUP = None

# from https://www.deepspeed.ai/tutorials/ds-sequence/
def initialize_model_parallel(
    args
):
    num_sequence_parallel_groups: int = args.world_size // args.sequence_parallel_size
    num_sequence_data_parallel_groups: int = args.world_size // args.sequence_parallel_size // args.data_parallel_size # do we define args.data_parallel_size ?
    global _SEQUENCE_PARALLEL_GROUP
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size,
                      (i + 1) * sequence_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group

def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP

def get_model(
    model_config: PretrainedConfig,
    args,
    activation_checkpointing_config=None,
):
    
    pp = args.pipe_parallel_size
    mp = args.model_parallel_size
    # from https://www.deepspeed.ai/tutorials/ds-sequence/
    sp = args.sequence_parallel_size # need to define this arg
    if sp and _SEQUENCE_PARALLEL_GROUP is None:
        ## set the degree of parallelism using the –ds-sequence-parallel-size argument. You also need to ensure that the number of attention heads is divisible by this value.
        assert (args.sequence_parallel_size % args.num_head)== 0 ## is args.num_head defined?
        initialize_model_parallel(args)
        
    assert args.world_size % (pp * mp) == 0
    dp = args.world_size // (pp * mp)

    topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=mp, num_dp=dp)
    # Offset base seeds for the interior pipeline stages.
    stage_id = topo.get_coord(rank=torch.distributed.get_rank()).pipe
    if 0 < stage_id < topo.get_dim("pipe") - 1:
        args.seed = args.seed + (stage_id * mp)
    if isinstance(model_config, LlamaConfig):
        return LlamaModelPipe(
            args,
            model_config,
            loss_fn=cross_entropy_fn,
            topology=topo,
            base_seed=args.seed,
            activation_checkpointing_config=activation_checkpointing_config,
        )
    elif isinstance(model_config, GPTNeoXConfig):
        return NeoxModelPipe(
            args,
            model_config,
            loss_fn=cross_entropy_fn,
            topology=topo,
            base_seed=args.seed,
            activation_checkpointing_config=activation_checkpointing_config,
        )
    #elif isinstance(model_config, FalconConfig):
    #    return FalconModelPipe(
    #        args,
    #        model_config,
    #        loss_fn=cross_entropy_fn,
    #        topology=topo,
    #        base_seed=args.seed,
    #        activation_checkpointing_config=activation_checkpointing_config,
    #    )
    elif isinstance(model_config, MistralConfig):
        return MistralModelPipe(
            args,
            model_config,
            loss_fn=cross_entropy_fn,
            topology=topo,
            base_seed=args.seed,
            activation_checkpointing_config=activation_checkpointing_config,
        )          
    else:
        raise NotImplementedError
