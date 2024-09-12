import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .utils import kls, convert_to_distribution, get_inf_gens
from .schedule_sample import sample_n_transitions_cont
from .continuous_time_diffusion import ContinuousTimeDiffusion

class SEDD(ContinuousTimeDiffusion): #schedule conditioning is True!
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes: int = 10,
        forward_kwargs={"type":"uniform"},
        schedule_type="cos",
        gamma=0,
        hybrid_loss_coeff=0.01,
        fix_x_t_bias=False,
        logistic_pars=False,
        **kwargs
    ):
        # Precalculate betas, define model_predict, p_sample
        super().__init__(x0_model_class, nn_params, num_classes, schedule_type, hybrid_loss_coeff, logistic_pars, **kwargs)
        self.save_hyperparameters(ignore=['x0_model_class'])
        self.fix_x_t_bias = fix_x_t_bias
        assert gamma >= 0 and gamma < 1 # full schedule and classical resp.

        # Precalculate Ks
        L = get_inf_gens(forward_kwargs, num_classes)
        self.register_buffer("L", L)

    def get_stationary(self):
        evals, evecs = torch.linalg.eig(self.L.T)
        norms_sq = torch.real(evals)
        max_eval = evals[torch.argmax(norms_sq)]
        assert torch.sqrt(torch.real(max_eval * max_eval.conj())) < self.eps
        stationary = evecs[:, torch.argmax(norms_sq)]
        assert torch.allclose(torch.imag(stationary), torch.tensor(0, dtype=self.L.dtype))
        stationary = torch.real(stationary)
        stationary = stationary * torch.sign(stationary)
        assert torch.all(stationary > 0)
        return stationary / stationary.sum()

    def get_trans_mats(self, t):
        return torch.matrix_exp(- self.log_alpha(t)[..., None, None] * self.L)

    def get_kl_t1(self, x):
        # sample S
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        x_0_logits = convert_to_distribution(x, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        x_1 = torch.log(torch.einsum("b...c,bcd->b...d", softmaxed, self.get_trans_mats(t)))
        kl = kls(x_1, self.get_stationary(), self.eps)
        return kl.mean()

    def x_t_sample(self, x_0, t, noise, S):
        # forward process, x_0 is the clean input.
        x_0_logits = convert_to_distribution(x_0, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        logits = torch.log(torch.einsum("b...c,bcd->b...d", softmaxed, self.get_trans_mats(t)))
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def r_posterior(self, x_0, x_t, t, S): # returns backward inf_gen
        x_0_logits = convert_to_distribution(x_0, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        p_y = torch.einsum("b...c,bcd->b...d", softmaxed, self.get_trans_mats(t))
        p_xt = p_y.gather(-1, x_t.unsqueeze(-1)).squeeze(-1)
        ratios = (p_y + self.eps)/(p_xt[..., None] + self.eps)
        ratios = ((ratios * self.L[x_t, :]).transpose(0, -1) * self.beta(t)).transpose(0, -1)
        return ratios

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        t, _, x_t = self.sample_point(x)
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, cond, None)
        true_r_posterior = self.r_posterior(x, x_t, t, None)
        pred_r_posterior = self.r_posterior(predicted_x0_logits, x_t, t, None)

        # get kls and loss
        kl = (- (true_r_posterior * torch.log(F.relu(pred_r_posterior)+self.eps)
                 - pred_r_posterior)
              + (true_r_posterior * torch.log(F.relu(true_r_posterior)+self.eps)
                 - true_r_posterior))
        vb_loss = kl.sum(-1).mean() * self.t_max

        # Also calculate cross entropy loss
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        return self.hybrid_loss_coeff * ce_loss + vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def p_sample(self, x, t, cond, noise, delta_t, S=None):
        # predict prev(x_t) or x_{t-1}
        predicted_x0_logits = self.model_predict(x, t, cond,None)
        bwd_inf_gen = self.r_posterior(predicted_x0_logits, x, t,None)
        x_0_logits = convert_to_distribution(x, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)
        pred_r_posterior = torch.log(softmaxed + delta_t * bwd_inf_gen)
        # sample
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 0).float().reshape((x.shape[0], *[1] * (x.dim())))
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_r_posterior + gumbel_noise * not_first_step, dim=-1
        )
        return sample


    def sample_with_image_sequence(self, x, cond=None, n_T=200, stride=10):
        steps = 0
        images = []
        pbar = tqdm(torch.flip(torch.linspace(0, self.t_max, n_T, dtype=torch.float32), (0,)), position=0, leave=True)
        for t in pbar:
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x_next = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classes), device=x.device), 1/n_T
            )
            x = x_next

            steps += 1
            if steps % stride == 0:
                images.append(x)

        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images
