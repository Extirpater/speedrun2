import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.functional import mse_loss, normalize


def _stable_kl_div(P_log_prob, Q_prob, epsilon=1e-10):
    """
    Stabilize by clamping Q_prob to avoid log(0) and division by zero
    """
    # ensure numerical stability
    Q_prob = Q_prob.clamp(min=epsilon)
    return F.kl_div(P_log_prob, Q_prob, reduction="none").sum(-1).mean()


def selected_kl_div( student_logits, teacher_logits, topk_indices):
    """
    Compute KL divergence **only on selected tokens**.
    """
    student_logits = student_logits.view(-1, student_logits.size(-1))
    teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
    student_logits = student_logits[topk_indices]  # Select tokens
    teacher_logits = teacher_logits[topk_indices]
    student_log_prob = F.log_softmax(student_logits, dim=-1)
    teacher_prob = F.softmax(teacher_logits, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(-1).mean()

def _cdist(x: torch.Tensor, y: torch.Tensor, p: float = 1.0) -> torch.Tensor:
    """Builtin cdist only works for float32"""
    import einops
    if x.dtype != torch.float32:
        # Reshape x and y for broadcasting
        x = einops.rearrange(x, "b l r -> b l () r")
        y = einops.rearrange(y, "b l r -> b () l r")
        # Compute the distance using the specified norm
        return (x - y).norm(dim=-1, p=p)
    # Use PyTorch's built-in cdist for other cases
    return torch.cdist(x, y, p=p)


def soft_mse_loss(feat_s, feat_t):
    student_prob = F.softmax(feat_s, dim=-1)
    teacher_prob = F.softmax(feat_t, dim=-1)
    return F.mse_loss(student_prob, teacher_prob)


def soft_mse_sum_loss(feat_s, feat_t):
    student_prob = F.softmax(feat_s, dim=-1)
    teacher_prob = F.softmax(feat_t, dim=-1)
    return F.mse_loss(student_prob, teacher_prob, reduction="none").sum(-1).mean()


def soft_cross_entropy_loss(feat_s, feat_t):
    student_prob = F.softmax(feat_s, dim=-1)
    teacher_prob = F.softmax(feat_t, dim=-1)
    return F.cross_entropy(student_prob, teacher_prob)

def compute_kl_from_logits(quant_logits: torch.Tensor,
                           base_logits: torch.Tensor) -> torch.Tensor:
    temperature = 10.0
    q_logp = F.log_softmax(quant_logits / temperature, dim=-1)
    b_logp = F.log_softmax(base_logits  / temperature, dim=-1)
    return F.kl_div(q_logp, b_logp, reduction="batchmean", log_target=True) / 16

def kl_divergence_loss(feat_s, feat_t, epsilon=1e-10):
    student_log_prob = F.log_softmax(feat_s, dim=-1)
    teacher_prob = F.softmax(feat_t, dim=-1)
    return _stable_kl_div(student_log_prob, teacher_prob)

def top_k_kl_divergence(x, y, k= 10000):
    topk_p_values, topk_p_indices = torch.topk(x, k, dim=-1)
    topk_q_values = torch.gather(y, -1, topk_p_indices)
    return kl_divergence_loss(topk_p_values, topk_q_values)

def reverse_kl_divergence_loss(feat_s, feat_t):
    teacher_log_prob = F.log_softmax(feat_t, dim=-1)
    student_prob = F.softmax(feat_s, dim=-1)
    return _stable_kl_div(teacher_log_prob, student_prob)


def cakld_loss(feat_s, feat_t, beta_prob=0.5):
    teacher_output_log_prob = F.log_softmax(feat_t, dim=-1)
    student_output_soft = F.softmax(feat_s, dim=-1)
    reverse_kl = _stable_kl_div(teacher_output_log_prob, student_output_soft)

    student_output_log_prob = F.log_softmax(feat_s, dim=-1)
    teacher_output_soft = F.softmax(feat_t, dim=-1)
    forward_kl = _stable_kl_div(student_output_log_prob, teacher_output_soft)

    kl_loss = beta_prob * reverse_kl + (1 - beta_prob) * forward_kl
    return kl_loss

def sparse_mse_loss(feat_s, feat_t):
    num = F.mse_loss(feat_s, feat_t)
    denom = F.mse_loss(torch.zeros_like(feat_t), feat_t)
    loss = num/denom
    return loss

def frob_norm(transfer_matrix, attn_matrix):
    loss = torch.linalg.matrix_norm(
            transfer_matrix - attn_matrix, ord="fro"
        ).mean()
    return loss

def jsd_loss(feat_s, feat_t, beta_prob=0.5):
    student_log_prob = F.log_softmax(feat_s, dim=-1)
    teacher_log_prob = F.log_softmax(feat_t, dim=-1)

    # Convert logits to probabilities for the mixed distribution calculation
    student_prob = student_log_prob.exp()
    teacher_prob = teacher_log_prob.exp()

    # Compute the mixed probability distribution
    m_prob = 0.5 * (student_prob + teacher_prob)

    # Calculate KL divergences between student/teacher log_probs and the mixed distribution
    kl_loss_f = _stable_kl_div(teacher_log_prob, m_prob)
    kl_loss_r = _stable_kl_div(student_log_prob, m_prob)

    # Compute the JSD as the average of the two KL divergences
    jsd = (kl_loss_f + kl_loss_r) / 2.0

    return jsd


def logsum_loss(student_proj, teacher_features, alpha=4.0):
    """
    Based on https://arxiv.org/pdf/2303.11098
    Experimentally they determine 4.0 to 5.0 performs well
    """
    # Calculate the absolute difference between the projected student features and teacher features
    diff = torch.abs(student_proj - teacher_features)

    # Apply the LogSum function
    loss = torch.logsumexp(alpha * torch.log(diff + 1e-12), dim=-1)

    # Return the mean loss over the batch
    return loss.mean()


def logsum_v2_loss(student_proj, teacher_features, alpha=4.0):
    """
    Found in https://arxiv.org/pdf/2303.11098
    Based on
    https://github.com/roymiles/Simple-Recipe-Distillation/blob/31e8477cfd/imagenet/torchdistill/losses/single.py
    Compute the LogSum loss based on the paper, with an adjustable smoothing factor alpha.
    Experimentally they determine 4.0 to 5.0 performs well
    """
    diff = torch.abs(student_proj - teacher_features)
    diff_pow = torch.pow(diff, alpha)
    sum_diff = torch.sum(diff_pow)
    return torch.log(sum_diff)


def cosine_distance_loss(feat_s, feat_t):
    cosine_sim = F.cosine_similarity(feat_s, feat_t, dim=-1)
    cosine_distance = 1 - cosine_sim
    return cosine_distance.mean()


def mutual_information_loss(feat_s, feat_t, alpha=0.1):
    feat_s = feat_s.squeeze(1)
    feat_t = feat_t.squeeze(1)

    # TODO: this function doesn't work, fix or remove
    similarities = torch.matmul(feat_s, feat_t.T) / alpha

    # Create labels for the diagonal entries (correct matches)
    batch_size = feat_s.shape[0]
    labels = torch.arange(batch_size).to(feat_s.device)

    # cross entropy requires float32
    with torch.amp.autocast("cuda", dtype=similarities.device.type):
        loss = F.cross_entropy(similarities, labels, reduction="none").sum(-1).mean()
    return loss


def sinkhorn_loss(feat_s, feat_t, epsilon=0.1, n_iters=20):
    """Based on algorithm in https://github.com/2018cx/SinKD/blob/main/loss.py#L119"""
    def sinkhorn_normalized(K, n_iters):
        for _ in range(n_iters):
            K = K / K.sum(dim=2, keepdim=True)
            K = K / K.sum(dim=1, keepdim=True)
        return K

    p_s = F.softmax(feat_s, dim=-1)
    p_t = F.softmax(feat_t, dim=-1)

    Wxy = _cdist(p_s, p_t, p=1)  # Cost Matrix
    K = torch.exp(-Wxy / epsilon)  # kernel matrix
    P = sinkhorn_normalized(K, n_iters)  # Sinkhorn iterations

    # EMD loss for batch
    return torch.sum(P * Wxy, dim=(1, 2)).mean()

def cosine_similarity(x, y, eps=1e-8):
    return (x * y).sum(dim=1) / (x.norm(dim=1) * y.norm(dim=1) + eps)

def pearson_correlation(x, y, eps=1e-8):
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    return (x * y).sum(dim=1) / (x.norm(dim=1) * y.norm(dim=1) + eps)

def token_relation_loss(x, y):
    R_x = x.unsqueeze(1).transpose(-1, -2)
    R_y = y.unsqueeze(1).transpose(-1, -2)
    R_x = R_x.view(R_x.size(0), -1, R_x.size(-1))
    R_y = R_y.view(R_y.size(0), -1, R_y.size(-1))
    loss = mse_loss(R_x, R_y)
    return loss

def sample_relation_loss(x, y, ratio=0.5):
    y_s = y[: int(y.size(0) * ratio)]
    x_s = x[: int(x.size(0) * ratio)]
    y_s = y_s.permute(0, 2, 1)
    x_s = x_s.permute(0, 2, 1)
    return token_relation_loss(y_s, x_s)

class FCDLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(FCDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, F_s, F_t):
        F_s = normalize(F_s, dim=1)
        F_t = normalize(F_t, dim=1)
        # knowledge distillation loss
        kd_loss = mse_loss(F_s, F_t)
        # correlation loss
        corr_loss = 1 - pearson_correlation(F_s, F_t).mean()
        # sample relation loss
        sr_loss = 0.0 # sample_relation_loss(F_s, F_t)
        # token relation loss
        tr_loss = 0.0 # token_relation_loss(F_s, F_t)
        return self.alpha * kd_loss + self.beta * (corr_loss + sr_loss + tr_loss)

def kl_mse(F_s, F_t):
    kl_loss = _stable_kl_div(F.log_softmax(F_s, dim=-1), F.softmax(F_t, dim=-1))
    F_s = normalize(F_s, dim=1)
    F_t = normalize(F_t, dim=1)
    mse_loss = F.mse_loss(F_s, F_t)
    return kl_loss + mse_loss

LOSS_FUNCTIONS = {
    "kl": kl_divergence_loss,
    "raw_mse": F.mse_loss,
    "soft_mse": soft_mse_loss,
    "reverse_kl": reverse_kl_divergence_loss,
    "cakld": cakld_loss,
    "jsd": jsd_loss,
    "logsum_v2": logsum_v2_loss,
    "cos": cosine_distance_loss,
    "ce": soft_cross_entropy_loss,

    # TODO: fix
    "mi": mutual_information_loss,
    "sinkhorn": sinkhorn_loss,

    # not recommended (TODO: delete?)
    "mse_sum": soft_mse_sum_loss,
    #"raw_ce": F.cross_entropy,
    #"logsum": logsum_loss,
}
