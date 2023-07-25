import copy
import json
import random
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Optional, Dict, Sequence, Union

import torch
import deepspeed
import transformers
from tqdm import tqdm
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

from fmengine.utils import is_rank_0
from fmengine.utils import logger_rank0 as logger

