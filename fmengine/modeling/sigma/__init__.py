from transformers import AutoConfig, AutoModelForCausalLM
from .modeling_sigma import SigmaConfig, SigmaForCausalLM

AutoConfig.register("sigma", SigmaConfig)
AutoModelForCausalLM.register(SigmaConfig, SigmaForCausalLM)
