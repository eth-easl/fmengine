import io
import os
import json
from loguru import logger
from functools import wraps
import torch.distributed as dist

__all__ = ["rank_zero"]


def rank_zero(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_rank_0():
            return
        result = func(*args, **kwargs)
        return result

    return wrapper

def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


# Log config
LOG_FILENAME = "ds_training.log"


class GetLogger:
    __instance = None
    __init_flag = True

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(GetLogger, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

    def __init__(self):
        if self.__init_flag:
            logger.add(LOG_FILENAME)
            self.__init_flag: False

    @rank_zero
    def trace(self, *args, **kwargs):
        logger.trace(*args, **kwargs)

    @rank_zero
    def debug(self, *args, **kwargs):
        logger.debug(*args, **kwargs)

    @rank_zero
    def info(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    @rank_zero
    def warning(self, *args, **kwargs):
        logger.warning(*args, **kwargs)

    @rank_zero
    def error(self, *args, **kwargs):
        logger.error(*args, **kwargs)


logger_rank0 = GetLogger()
