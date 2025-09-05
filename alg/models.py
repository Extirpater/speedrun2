import copy
import functools
import math

import transformers
import torch
import torch.nn as nn

def zero_grad_hook(grad):
    return torch.zeros_like(grad)

def get_model_tokenizer(model_args, **model_kwargs):
    if model_args.use_lk:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        # automodel_cls = AutoLigerKernelForCausalLM
        # TODO: remove hack below, use above comment once https://github.com/linkedin/Liger-Kernel/issues/242 is fixed
        class PatchedAutoLiger(AutoLigerKernelForCausalLM):
            @staticmethod
            def from_config(config, *args, **kwargs):
                AutoLigerKernelForCausalLM.from_pretrained(config._name_or_path)
                return AutoLigerKernelForCausalLM.from_config(config, *args, **kwargs)
        automodel_cls = PatchedAutoLiger

    else:
        automodel_cls = transformers.AutoModelForCausalLM
    model = automodel_cls.from_pretrained(
        model_args.model_name,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        **model_kwargs
    )
    
    model = model.type(torch.bfloat16)
    # model.model called here 
    replace_linear_layers(model.model)
    # weight copy upcasts to float32, so we downcast
    model = model.type(torch.bfloat16)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name)
    except:
        raise ValueError
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

class QuantizeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.weight 
        # straight through estimator for grads
        x = x + (quantize(x) - x).detach()
        output = torch.nn.functional.linear(input, x, bias=self.bias)
        return output

def replace_linear_layers(module):
    """
    Recursively replace all nn.Linear layers in 'module' (and its submodules)
    with LpLqLinear layers, preserving weight and bias.
    """
    for name, child in module.named_children():
        # If the child is a Linear layer, replace it with LpLqLinear
        if isinstance(child, nn.Linear):
            # Create new LpLqLinear with matching hyperparameters and dimensions
            new_layer = QuantizeLinear(child.in_features, child.out_features, bias=False)
            # Copy old weights
            new_layer.weight.data.copy_(child.weight.data) 
            # Assign the new layer back to the parent
            setattr(module, name, new_layer)
        else:
            # Recursively descend into child modules
            replace_linear_layers(child)

def quantize(input, max_scale=0.7):
    # TWN (Ternary Weight Networks) per row Quantizer
    out = input.clone().detach()
    out = out.reshape(-1, input.shape[-1])
    # Per Channel/Group Quantization
    n = out[0].nelement()
    m = out.data.norm(p=1, dim=1).div(n)
    thres = (max_scale * m).view(-1, 1).expand_as(out)
    pos = (out > thres).float()
    neg = (out < -thres).float()
    mask = (out.abs() > thres).float()
    alpha = ((mask * out).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
    result = alpha * pos - alpha * neg

    result = result.reshape(input.shape) 

    return result