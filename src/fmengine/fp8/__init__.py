import warnings

from fmengine.fp8.dtypes import DTypes  # noqa
from fmengine.fp8.linear import FP8Linear  # noqa
from fmengine.fp8.parameter import FP8Parameter  # noqa
from fmengine.fp8.tensor import FP8Tensor  # noqa

try:
    import transformer_engine as te  # noqa
    import transformer_engine_extensions as tex  # noqa
except ImportError:
    warnings.warn("Please install Transformer engine for FP8 training!")
