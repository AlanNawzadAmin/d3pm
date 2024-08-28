import torch
import torch.nn


def _at(a, t, x):
    # t is 1-d, x is integer value of 0 to num_classes - 1
    bs = t.shape[0]
    t = t.reshape((bs, *[1] * (x.dim() - 1)))
    return a[t, x, :]

def kls(dist1, dist2, eps): # KL of dists on last dim
    out = torch.softmax(dist1 + eps, dim=-1) * (
        torch.log_softmax(dist1 + eps, dim=-1)
        - torch.log_softmax(dist2 + eps, dim=-1)
    )
    return out.sum(dim=-1)#.mean()

def convert_to_distribution(x_0, num_classes, eps):
    # returns log probs of x_0 as a distribution
    if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
        x_0_logits = torch.log(
            torch.nn.functional.one_hot(x_0, num_classes) + eps
        )
    else:
        x_0_logits = x_0.clone()
    return x_0_logits

def get_betas(schedule_type, n_T):
    steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T
    if schedule_type == 'cos':
        alpha_bar = 1-torch.cos((1 - steps) * torch.pi / 2)
        beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )
    return beta_t
