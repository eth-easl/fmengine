import transformers
from typing import Optional
from dataclasses import dataclass, field
from fmengine.modeling.mistral.hf_interface import from_hf


@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default="mistralai/Mistral-7B-v0.1")
    output_dir: str = field(default="mistral-7b")
    mp_world_size: int = field(default=1)


def main():
    parser = transformers.HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()
    from_hf(
        args.model_name_or_path,
        args.output_dir,
        args.mp_world_size,
    )


if __name__ == "__main__":
    main()
