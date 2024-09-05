import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from .trainer import DiffusionTrainer

from .schedule_sample import sample_n_transitions, sample_full_transitions

def get_betas(schedule_type, n_T):
    steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T
    if schedule_type == 'cos':
        alpha_bar = 1-torch.cos((1 - steps) * torch.pi / 2)
        beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )
    return beta_t

class DiscreteTimeDiffusion(DiffusionTrainer): #schedule conditioning is True!
    def __init__(
        self,
        x0_model_class,
        nn_params,
        n_T: int,
        num_classes: int = 10,
        schedule_type="cos",
        hybrid_loss_coeff=0.001,
        logistic_pars=False,
        lr=1e-3,
    ) -> None:
        super().__init__(lr)
        self.save_hyperparameters(ignore=['x0_model_class'])
        self.hparams.update(x0_model_class=x0_model_class.__name__)
        self.x0_model = x0_model_class(**nn_params)
        self.n_T = n_T
        self.hybrid_loss_coeff = hybrid_loss_coeff
        self.eps = 1e-6
        self.num_classes = num_classes

        # Precalculate betas
        self.beta_t = get_betas(schedule_type, n_T)

    def get_stationary(self):
        raise NotImplementedError

    def get_kl_t1(self, x):
        raise NotImplementedError

    def model_predict(self, x_0, t, cond, S=None):
        if not self.logistic_pars:
            return self.x0_model(x_0, t, cond, S)
        else:
            pred = self.x0_model(x_0, t, cond, S)
            loc = pred[..., 0].unsqueeze(-1)
            log_scale = pred[..., 1].unsqueeze(-1)
            inv_scale = torch.exp(- (log_scale - 2.))
            bin_width = 2. / (self.num_classes - 1.)
            bin_centers = torch.linspace(-1., 1., self.num_classes).to(pred.device)
            bin_centers = bin_centers - loc
            log_cdf_min = torch.nn.LogSigmoid()(
                inv_scale * (bin_centers - 0.5 * bin_width))
            log_cdf_max = torch.nn.LogSigmoid()(
                inv_scale * (bin_centers + 0.5 * bin_width))
            logits = log_cdf_max + torch.log1p(-torch.exp(log_cdf_min-log_cdf_max) + self.eps)
            return logits

    def q_posterior_logits(self, x_0, x_t, t, S=None):
        raise NotImplementedError

    def x_t_sample(self, x_0, t, noise, S=None):
        raise NotImplementedError

    def sample_point(self, x):
        t = torch.randint(0, self.n_T, (x.shape[0],), device=x.device)
        S = sample_n_transitions(self.beta_t.to(x.device), x[0].flatten().shape[0], t)
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        x_t = self.x_t_sample(
            x, t, torch.rand((*x.shape, self.num_classes), device=x.device), S
        )
        return t, S, x_t

    def p_sample(self, x, t, cond, noise, S=None):
        # predict prev(x_t) or x_{t-1}
        predicted_x0_logits = self.model_predict(x, t, cond, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t, S)
        # sample
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 0).float().reshape((x.shape[0], *[1] * (x.dim())))
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample
