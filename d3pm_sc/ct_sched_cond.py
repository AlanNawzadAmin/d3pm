import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import kls, convert_to_distribution, get_inf_gens
from .schedule_sample import sample_n_transitions_cont
from .continuous_time_diffusion import ContinuousTimeDiffusion

class ScheduleCondition(ContinuousTimeDiffusion): #schedule conditioning is True!
    def __init__(
        self,
        x0_model: nn.Module,
        num_classes: int = 10,
        forward_kwargs={"type":"uniform"},
        schedule_type="cos",
        gamma=0,
        hybrid_loss_coeff=0.01,
        fix_x_t_bias=False,
        logistic_pars=False,
    ):
        # Precalculate betas, define model_predict, p_sample
        super().__init__(x0_model, num_classes, schedule_type, hybrid_loss_coeff, logistic_pars)
        self.save_hyperparameters(ignore=['x0_model'])
        self.fix_x_t_bias = fix_x_t_bias
        assert gamma >= 0 and gamma < 1 # full schedule and classical resp.

        # Precalculate Ks
        L = get_inf_gens(forward_kwargs, num_classes)
        rate = - (L.diagonal().min()) / (1-gamma) # L^* in sec 6.6 of the notes
        K = L / rate + torch.eye(num_classes)
        self.beta_scale = self.beta
        self.beta = lambda t: rate * self.beta_scale(t)
        self.log_alpha_scale = self.log_alpha
        self.log_alpha  = lambda t: self.log_alpha_scale(t) * rate
        K_powers = torch.stack([torch.linalg.matrix_power(K, i) for i in range(5000)])
        self.register_buffer("K", K)
        self.register_buffer("K_powers", K_powers)

    def get_stationary(self):
        evals, evecs = torch.linalg.eig(self.K.T)
        assert torch.isclose(evals[torch.argmax(torch.norm(evals))], torch.tensor(1, dtype=torch.complex64))
        stationary = evecs[:, torch.argmax(torch.norm(evals))]
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

    def q_posterior_logits(self, x_0, x_t, t, S):
        x_0_logits = convert_to_distribution(x_0, self.num_classes, self.eps)
        fact1 = self.K.T[x_t, :] # x_t | x_{t-1}
        trans_mats = self.K_powers[S - 1, :, :] # make sure we ignore S=0 later!
        if self.fix_x_t_bias:
            x_0_logits -= torch.log(self.K_powers[S, :, x_t] + self.eps) # x_{t} | x_{0}
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        fact2 = torch.einsum("b...c,b...cd->b...d", softmaxed, trans_mats) # x_{t-1} | x_{0}
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        
        # check if t==0; then we calculate the distance to x_0
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        bc = torch.where(t_broadcast == 0, x_0_logits, out)
        return bc

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        t, S, x_t = self.sample_point(x)
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, cond, S)
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)

        # get kls and loss
        kl = kls(true_q_posterior_logits, pred_q_posterior_logits, self.eps) # shape x
        weight = - self.beta(t) / self.log_alpha(t)
        weight = (S.swapaxes(0, -1) * weight).swapaxes(0, -1)
        vb_loss = (kl * weight).mean() * self.t_max

        # Also calculate cross entropy loss
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        return self.hybrid_loss_coeff * ce_loss + vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def sample_with_image_sequence(self, x, cond=None, trans_step=1, stride=10):
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        steps = 0
        images = []
        pbar = tqdm(total=S.sum(-1).sum(-1).sum(-1).max().item(), unit="iteration",
                    position=0, leave=True)
        while S.sum() > 0:
            # predict what comes next
            x_next = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classes), device=x.device), S
            )
            for b in range(len(x)):
                trans_indices = torch.argwhere(S[b] > 0)
                trans_indices = trans_indices[torch.randperm(len(trans_indices))]
                if len(trans_indices) > 0:
                    # randomly transiiton
                    for i, j, k in trans_indices[:trans_step]:
                        x[b, i, j, k] = x_next[b, i, j, k]
                        S[b, i, j, k] -= 1
            pbar.update(trans_step)
            steps += 1
            if steps % stride == 0:
                images.append(torch.clone(x))
        pbar.close()
        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images
