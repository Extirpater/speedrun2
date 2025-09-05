import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# lm-eval imports
from lm_eval import evaluator, tasks
import lm_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HF model with TWN-quantized Linear layers via lm-eval")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model identifier")
    parser.add_argument(
        "--tasks", nargs='+', default=["mmlu"], 
        help="List of lm-eval tasks to run (e.g., wikitext, lambada)")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Device to run on")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of shots for evaluation.")
    return parser.parse_args()

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
                                                     

class TWNQuantLinear(nn.Linear):
    """
    Linear layer with TWN (Ternary Weight Network) quantization applied to weights.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q_weight = quantize(self.weight)
        return F.linear(input, q_weight.to(input.dtype))


def replace_linears(module: nn.Module) -> None:
    """
    Recursively replace all nn.Linear (except lm_head) with TWNQuantLinear
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name != 'lm_head':
            quant = TWNQuantLinear(child.in_features, child.out_features, bias=(child.bias is not None))
            quant.weight.data = child.weight.data.clone()
            if child.bias is not None:
                quant.bias.data = child.bias.data.clone()
            setattr(module, name, quant)
        else:
            replace_linears(child)

def main():
    args = parse_args()
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype = torch.bfloat16)
    replace_linears(model.model)
    model = model.type(torch.bfloat16)
    lm = lm_eval.models.huggingface.HFLM(pretrained=model.to(args.device), batch_size = "auto")
    results = lm_eval.simple_evaluate(model = lm, tasks = args.tasks, num_fewshot = args.num_fewshot)

    for task, res in results.items():
        print(f"== {task}, shots: {args.num_fewshot} ==")
        try:
            for metric, val in res.items():
                print(f"{metric}: {val}")
            if task == "results":
                    break
        except:
            break


if __name__ == '__main__':
    main()
