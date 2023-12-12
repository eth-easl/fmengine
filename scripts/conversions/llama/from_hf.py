import os
import transformers
from typing import Optional
from dataclasses import dataclass, field
from fmengine.modeling.llama.hf_interface import from_hf


@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field()
    output_dir: str = field()
    mp_world_size: int = field(default=1)


def main():
    if "HF_HOME" not in os.environ:
        # avoid crashing the user directory
        print("[warning]: huggingface cache directory not specified")
        exit(1)

    parser = transformers.HfArgumentParser((Arguments,))
    (args,) = parser.parse_args_into_dataclasses()
    from_hf(
        args.model_name_or_path,
        args.output_dir,
        args.mp_world_size,
    )


if __name__ == "__main__":
    main()
