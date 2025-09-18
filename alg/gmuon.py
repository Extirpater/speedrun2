import torch
import torch.distributed as dist
import random


# -------------------------------
# Newton–Schulz orthogonalization
# -------------------------------
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute orthogonalization of G.
    Produces US'V^T with S' ~ Uniform(0.5, 1.5) instead of UV^T.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# -------------------------------
# Expander graph utilities
# -------------------------------
_expander_cache = {}
import torch

_ramanujan_cache = {}

def ramanujan_mask(rows, cols, degree=4, device=None):
    """
    Cached Ramanujan-style expander mask.
    Deterministic, only built once per (rows, cols, degree, device).
    """
    key = (rows, cols, degree, device)
    if key in _ramanujan_cache:
        return _ramanujan_cache[key]

    # Compute quadratic residues mod cols
    residues = sorted({(i * i) % cols for i in range(1, cols)})
    if len(residues) < degree:
        raise ValueError(f"Not enough quadratic residues to build degree {degree} Ramanujan graph")

    generators = residues[:degree]
    mask = torch.zeros(rows, cols, device=device)
    for r in range(rows):
        for g in generators:
            c = (r + g) % cols
            mask[r, c] = 1.0

    _ramanujan_cache[key] = mask
    return mask

def build_expander_mask(shape, degree=4, device=None):
    """
    Build binary expander-style mask for matrix updates.
    Approximated by giving each row 'degree' random neighbors.
    """
    rows, cols = shape[-2], shape[-1]
    mask = torch.zeros(rows, cols, device=device)
    for r in range(rows):
        neighbors = random.sample(range(cols), k=min(degree, cols))
        mask[r, neighbors] = 1.0
    return mask

def apply_ramanujan(update, degree=4):
    rows, cols = update.shape[-2], update.shape[-1]
    mask = ramanujan_mask(rows, cols, degree=degree, device=update.device)
    return update * mask

def apply_expander(update, degree=4):
    shape = update.shape
    key = (shape[-2], shape[-1], degree, update.device)
    if key not in _expander_cache:
        _expander_cache[key] = build_expander_mask(shape, degree=degree, device=update.device)
    return update * _expander_cache[key]


# -------------------------------
# Muon update (cautious + expander)
# -------------------------------
def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, degree=4):
    # Gradient power transform
    power = 1.2
    grad = torch.sign(grad) * torch.pow(torch.abs(grad) + 1e-6, power)

    # Momentum smoothing
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum

    # Orthogonalization
    if update.ndim == 4:  # conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5

    # Cautious mask (sign-consistency)
    mask = (update * grad > 0).to(update.dtype)
    mask.div_(mask.mean().clamp_(min=1e-3))
    update = update * mask

    # Expander mask
    #update = apply_expander(update, degree=degree)
    update = apply_ramanujan(update, degree=degree)

    return update


# -------------------------------
# Optimizers
# -------------------------------
class Muon(torch.optim.Optimizer):
    """
    Distributed Muon with cautious + expander masking.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, degree=4):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, degree=degree)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad, state["momentum_buffer"],
                        beta=group["momentum"],
                        degree=group["degree"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Non-distributed Muon with cautious + expander masking.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, degree=4):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, degree=degree)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(
                    p.grad, state["momentum_buffer"],
                    beta=group["momentum"],
                    degree=group["degree"]
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class GraphMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon + Adam, with cautious + expander masking on Muon params.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["degree"] = group.get("degree", 4)
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(
                            p.grad, state["momentum_buffer"],
                            beta=group["momentum"],
                            degree=group["degree"]
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed Muon + Adam, with cautious + expander masking on Muon params.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["degree"] = group.get("degree", 4)
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad, state["momentum_buffer"],
                        beta=group["momentum"],
                        degree=group["degree"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        return loss

