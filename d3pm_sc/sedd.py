import numpy as np
import torch
import torch.nn as nn
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
        self.L = L
        self.register_buffer("L", L)

    def get_stationary(self):
        evals, evecs = torch.linalg.eig(self.K.T)
        norms_sq = torch.real(evals * evals.conj())
        assert torch.isclose(evals[torch.argmax(norms_sq)], torch.tensor(1, dtype=torch.complex64))
        stationary = evecs[:, torch.argmax(norms_sq)]
        assert torch.allclose(torch.imag(stationary), torch.tensor(0, dtype=self.K.dtype))
        stationary = torch.real(stationary)
        stationary = stationary * torch.sign(stationary)
        assert torch.all(stationary > 0)
        return stationary / stationary.sum()

    def get_kl_t1(self, x):
        # sample S
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        x_0_logits = convert_to_distribution(x, self.num_classes, self.eps)
        trans_mats = self.K_powers[S, :, :]
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        x_1 = torch.log(torch.einsum("b...c,b...cd->b...d", softmaxed, trans_mats))
        kl = kls(x_1, self.get_stationary(), self.eps)
        return kl.mean()

    def x_t_sample(self, x_0, t, noise, S):
        # forward process, x_0 is the clean input.
        logits = torch.log(self.K_powers[S, x_0, :] + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(self, x_0, x_t, t, S): # returns backward inf_gen
        x_0_logits = convert_to_distribution(x_0, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        P_0t = torch.matrix_exp(- self.log_alpha(t) * self.L)
        p_y = torch.einsum("b...c,cd->b...d", softmaxed, P_0t)
        p_xt = p_y[x_t]
        ratios = (p_y + self.eps)/(p_xt[..., None] + self.eps)
        ratios = ratios * self.L[x_t, :] * self.beta(t)
        return ratios

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        t, _, x_t = self.sample_point(x)
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, cond, S)
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)

        # get kls and loss
        kl = (- (true_q_posterior_logits * torch.log(F.ReLU(pred_q_posterior_logits)+self.eps)
                 - pred_q_posterior_logits))
              + (true_q_posterior_logits * torch.log(F.ReLU(true_q_posterior_logits)+self.eps)
                 - true_q_posterior_logits)))
        vb_loss = kl.mean() * self.t_max

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
        predicted_x0_logits = self.model_predict(x, t, cond, S)
        bwd_inf_gen = self.q_posterior_logits(predicted_x0_logits, x, t, S)
        pred_q_posterior_logits = torch.log(torch.matrix_exp(delta_t * bwd_inf_gen))
        # sample
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = (t != 0).float().reshape((x.shape[0], *[1] * (x.dim())))
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample


    def sample_with_image_sequence(self, x, cond=None, stride=10):
        steps = 0
        images = []
        pbar = tqdm(np.arange(0, self.n_T)[::-1], position=0, leave=True)
        for t in pbar:
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x_next = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classes), 1/self.n_T, device=x.device)
            )
            x = x_next

            steps += 1
            if steps % stride == 0:
                images.append(x)

        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images
