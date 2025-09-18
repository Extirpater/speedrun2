# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from muon import zeropower_via_newtonschulz5  # unchanged

from transformers.utils.versions import require_version


class SignEMAAdamW(Optimizer):
    """
    AdamW with cautious masking (kept) + flip hysteresis:
      - sign-EMA to require persistent evidence for sign changes
      - zero-crossing clamp unless flip is allowed
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = True,
        # --- new knobs ---
        signema_beta: float = 0.9,
        flip_thresh: float = 0.8,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. "
                "Use torch.optim.AdamW instead, or set no_deprecation_warning=True to silence.",
                FutureWarning,
            )
        require_version("torch>=1.5.0")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = {
            "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            # store new knobs in defaults so per-group override is possible
            "signema_beta": signema_beta,
            "flip_thresh": flip_thresh,
        }
        super().__init__(params, defaults)
        self.init_lr = lr
        self.rs = 20

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lr = group["lr"]
            correct_bias = group["correct_bias"]
            signema_beta = group["signema_beta"]
            flip_thresh = group["flip_thresh"]

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # state init
                step_t = state.get("step", 0) + 1
                state["step"] = step_t
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                if "sign_ema" not in state:
                    # initialize EMA of grad sign with the current sign
                    state["sign_ema"] = torch.sign(grad).to(grad.dtype).clone()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # decoupled weight decay on params (shadow weights)
                if wd > 0.0:
                    p.add_(p, alpha=(-lr * wd))

                # adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1.0 - beta2))
                denom = exp_avg_sq.sqrt().add_(eps)

                step_size = lr
                if correct_bias:
                    bias_correction1 = 1.0 - beta1 ** step_t
                    bias_correction2 = 1.0 - beta2 ** step_t
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # ---- cautious mask (kept from your code) ----
                # only keep components where momentum agrees with current grad
                mask = (exp_avg * grad > 0).to(grad.dtype)
                # normalize mask density
                mask.div_(mask.mean().clamp_(min=1e-3))
                norm_grad = (exp_avg * mask) / denom

                # ---- flip hysteresis additions ----
                # 1) update EMA of grad sign
                s = state["sign_ema"]
                gsign = torch.sign(grad).to(grad.dtype)
                s.lerp_(gsign, 1 - signema_beta)     # s := beta*s + (1-beta)*sign(g)
                state["sign_ema"] = s

                # 2) proposed delta
                delta = -step_size * norm_grad

                # 3) allow-flip mask:
                #    strong EMA, aligned with current grad, and wants opposite (or zero) sign vs p
                strong = s.abs() >= flip_thresh
                want = torch.sign(s)
                aligned = (want * torch.sign(grad)) >= 0
                opposite = (p * want) <= 0
                allow_flip = strong & aligned & opposite

                # 4) clamp zero-crossings unless allowed
                new_p = p + delta
                cross = ((p > 0) & (new_p < 0)) | ((p < 0) & (new_p > 0))
                block = cross & (~allow_flip)
                if block.any():
                    eps_z = 1e-8 if p.dtype.is_floating_point else 0.0
                    # bring to (almost) zero instead of crossing
                    delta[block] = -p[block] + torch.sign(-p[block]) * eps_z

                # apply
                p.add_(delta)

        return loss

