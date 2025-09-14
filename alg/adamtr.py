# --- TernaryTRPrecondWithAuxAdamV2: anti-stall, warmup decay, soft mask ---
import math
from typing import Iterable, Optional
import torch
from torch.optim import Optimizer

def _rms(x: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(x)
    if not finite.any():
        return torch.tensor(0.0, device=x.device, dtype=torch.float32)
    x = x[finite]
    return x.pow(2).mean().sqrt().clamp_min(1e-16)

def _flatten_to_2d(t: torch.Tensor):
    if t.ndim == 2:
        return t, t.shape, t.shape[0], t.shape[1]
    elif t.ndim == 1:
        return t.view(-1, 1), t.shape, t.numel(), 1
    else:
        out_dim = t.shape[0]
        in_dim = t.numel() // out_dim
        return t.view(out_dim, in_dim), t.shape, out_dim, in_dim

def _restore_shape(M2d: torch.Tensor, orig_shape):
    if len(orig_shape) == 2:
        return M2d.view(*orig_shape)
    elif len(orig_shape) == 1:
        return M2d.view(orig_shape[0])
    else:
        out_dim = orig_shape[0]
        rest = orig_shape[1:]
        return M2d.view(out_dim, *rest)

def _polar_unit_factor(A: torch.Tensor, iters: int = 2, eps: float = 1e-8) -> torch.Tensor:
    if A.numel() == 0:
        return A
    if A.shape[0] == 1 or A.shape[1] == 1:
        denom = _rms(A)
        return A / (denom if denom > 0 else 1.0)
    B = A.transpose(-1, -2) @ A
    if not torch.isfinite(B).all():
        denom = _rms(A)
        return A / (denom if denom > 0 else 1.0)
    trace = torch.trace(B)
    if not torch.isfinite(trace) or trace <= 0:
        denom = _rms(A)
        return A / (denom if denom > 0 else 1.0)
    c = trace / B.shape[-1]
    Y = B / (c + eps)
    I = torch.eye(Y.shape[-1], device=Y.device, dtype=Y.dtype); Z = I.clone()
    for _ in range(iters):
        if not torch.isfinite(Y).all() or not torch.isfinite(Z).all():
            denom = _rms(A)
            return A / (denom if denom > 0 else 1.0)
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T; Z = T @ Z
    inv_sqrt_B = Z / math.sqrt(float(c) + float(eps))
    Q = A @ inv_sqrt_B
    if not torch.isfinite(Q).all():
        denom = _rms(A)
        return A / (denom if denom > 0 else 1.0)
    return Q

class TernaryTRPrecondWithAuxAdamV2(Optimizer):
    """
    Fast ternary trust-region optimizer with anti-plateau features.

    TR group keys (use_tr=True):
      precond: 'adafactor' | 'shampoo'
      lr, weight_decay, beta1=0.9, wmax=1.0
      target_update_rms=1.0, polar_iters=2, polar_max_dim=2048
      beta2_ada=0.999            # Adafactor row/col EMA
      beta2_sham=0.999           # Shampoo EMA
      precond_update_freq=25     # Shampoo invsqrt refresh
      max_precond_dim=1024       # Shampoo only if dims <= this
      max_grad_norm=None         # per-param grad clip
      eps=1e-8

      # Anti-stall knobs:
      stall_window=50            # steps with tiny updates to trigger boost
      stall_tol=0.2              # upd_ema < stall_tol * target_update_rms => "tiny"
      boost_factor=2.0           # temporarily multiply target_update_rms
      boost_steps=5              # for this many steps
      upd_ema_alpha=0.98         # EMA for update RMS
      mask_relax_every=0         # every K steps, skip cautious mask (0=never)
      mask_tau=0.0               # only zero momentum if |g| > mask_tau
      decay_warmup_steps=0       # disable weight decay for first N steps (per-param)

    Aux AdamW group keys (use_tr=False):
      lr, weight_decay, betas=(0.9,0.999), eps=1e-8, max_grad_norm=None, decay_warmup_steps=0
    """
    def __init__(self, params: Iterable, defaults: Optional[dict] = None, **global_overrides):
        base = dict(
            # shared
            lr=2e-3, weight_decay=0.0, eps=1e-8,
            # TR core
            use_tr=False, precond="adafactor",
            beta1=0.9, wmax=1.0, target_update_rms=1.0,
            polar_iters=2, polar_max_dim=2048, max_grad_norm=None,
            beta2_ada=0.999,
            beta2_sham=0.999, precond_update_freq=25, max_precond_dim=1024,
            # Anti-stall
            stall_window=50, stall_tol=0.2, boost_factor=2.0, boost_steps=5,
            upd_ema_alpha=0.98, mask_relax_every=0, mask_tau=0.0,
            decay_warmup_steps=0,
            # AdamW
            betas=(0.9, 0.999),
        )
        if defaults is None: defaults = {}
        base.update(defaults); base.update(global_overrides)
        super().__init__(params, base)

    @torch.no_grad()
    def _precond_adafactor(self, g: torch.Tensor, state: dict, beta2: float, eps: float) -> torch.Tensor:
        g2d, orig, out_dim, in_dim = _flatten_to_2d(g)
        if out_dim == 0 or in_dim == 0: return g
        if "vr" not in state:
            device, dtype = g2d.device, g2d.dtype
            state["vr"] = torch.zeros(out_dim, 1, device=device, dtype=dtype)
            state["vc"] = torch.zeros(1, in_dim, device=device, dtype=dtype)
        vr, vc = state["vr"], state["vc"]
        g2 = g2d * g2d
        vr.mul_(beta2).add_(g2.mean(dim=1, keepdim=True), alpha=(1 - beta2))
        vc.mul_(beta2).add_(g2.mean(dim=0, keepdim=True), alpha=(1 - beta2))
        gp2d = (vr + eps).rsqrt() * g2d * (vc + eps).rsqrt()
        return _restore_shape(gp2d, orig)

    @torch.no_grad()
    def _precond_lite_shampoo(self, g: torch.Tensor, state: dict, group: dict) -> torch.Tensor:
        g2d, orig, out_dim, in_dim = _flatten_to_2d(g)
        if out_dim == 0 or in_dim == 0: return g
        if max(out_dim, in_dim) > group["max_precond_dim"]:
            return self._precond_adafactor(g, state, group["beta2_sham"], group["eps"])
        if "G" not in state:
            device, dtype = g2d.device, g2d.dtype
            state["G"] = torch.zeros(out_dim, out_dim, device=device, dtype=dtype)
            state["H"] = torch.zeros(in_dim, in_dim, device=device, dtype=dtype)
            state["G_inv_sqrt"] = torch.eye(out_dim, device=device, dtype=dtype)
            state["H_inv_sqrt"] = torch.eye(in_dim, device=device, dtype=dtype)
            state["step_sham"] = 0
        beta2 = group["beta2_sham"]
        state["G"].mul_(beta2).add_(g2d @ g2d.transpose(0, 1), alpha=(1 - beta2))
        state["H"].mul_(beta2).add_(g2d.transpose(0, 1) @ g2d, alpha=(1 - beta2))
        state["step_sham"] += 1
        if (state["step_sham"] % group["precond_update_freq"]) == 0:
            def _eig_inv_sqrt(mat):
                mat = 0.5 * (mat + mat.transpose(-1, -2))
                eye = torch.eye(mat.shape[-1], device=mat.device, dtype=mat.dtype)
                evals, evecs = torch.linalg.eigh(mat + group["eps"] * eye)

