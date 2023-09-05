
import transformers
from typing import Optional
from dataclasses import dataclass, field
from fmengine.modeling.llama.hf_interface import from_hf
from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=2, lora_alpha=32, lora_dropout=0.1
)

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default="/path/to/llama-7b-hf")
    output_dir: str = field(default="./llama-7B-init-ckpt")
    mp_world_size: int = field(default=1)

def main():
    parser = transformers.HfArgumentParser((Arguments,))
    args, = parser.parse_args_into_dataclasses()
    from_hf(
        args.model_name_or_path,
        args.output_dir,
        args.mp_world_size,
        peft_config=peft_config,
    )
 
if __name__ == "__main__":
    main()