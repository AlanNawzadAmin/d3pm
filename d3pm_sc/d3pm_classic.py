import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import _at, kls, convert_to_distribution
from .discrete_time_diffusion import DiscreteTimeDiffusion

class D3PMClassic(DiscreteTimeDiffusion):
    def __init__(
        self,
        x0_model_class,
        nn_params,
        n_T: int,
        num_classes: int = 10,
        forward_kwargs={"type":"uniform"},
        schedule_type="cos",
        hybrid_loss_coeff=0.001,
        fix_x_t_bias=False,
        logistic_pars=False,
        lr=1e-3,
        **kwargs
    ):
        # Precalculate betas, define model_predict, p_sample
        super().__init__(x0_model_class, nn_params, n_T, num_classes, schedule_type, hybrid_loss_coeff, logistic_pars, lr)
        self.fix_x_t_bias = fix_x_t_bias

        # Precalculate betas and Qs
        q_onestep_mats = []
        q_mats = []  # these are cumulative
        for beta in self.beta_t:
            if forward_kwargs['type'] == "uniform":
                mat = torch.ones(num_classes, num_classes) * beta / num_classes
                mat.diagonal().fill_(1 - (num_classes - 1) * beta / num_classes)
                q_onestep_mats.append(mat)
            elif forward_kwargs['type'] == "masking":
                mat = torch.eye(num_classes) * (1 - beta)
                mat[:, -1] += beta
                q_onestep_mats.append(mat)
            else:
                raise NotImplementedError
        q_one_step_mats = torch.stack(q_onestep_mats, dim=0)
        q_one_step_transposed = q_one_step_mats.transpose(1, 2)
        q_mat_t = q_onestep_mats[0]
        q_mats = [q_mat_t]
        for idx in range(1, self.n_T):
            q_mat_t = q_mat_t @ q_onestep_mats[idx]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.register_buffer("q_one_step_transposed", q_one_step_transposed)
        self.register_buffer("q_mats", q_mats)
        
    def get_stationary(self):
        evals, evecs = torch.linalg.eig(q_mats[-1].T)
        norms_sq = torch.real(evals * evals.conj())
        assert torch.isclose(evals[torch.argmax(norms_sq)], torch.tensor(1, dtype=torch.complex64))
        stationary = evecs[:, torch.argmax(norms_sq)]
        assert torch.allclose(torch.imag(stationary), torch.tensor(0, dtype=self.K.dtype))
        stationary = torch.real(stationary)
        stationary = stationary * torch.sign(stationary)
        assert torch.all(stationary > 0)
        return stationary / stationary.sum()

    def get_kl_t1(self, x):
        # get p(x_1|x_0)
        x_0_logits = convert_to_distribution(x, self.num_classes, self.eps)
        trans_mats = self.q_mats[-1, :, :]
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        x_1 = torch.einsum("b...c,...cd->b...d", softmaxed, trans_mats)
        kl = kls(x_1, self.get_stationary(), self.eps)
        return kl.mean()
    
    def x_t_sample(self, x_0, t, noise, S=None):
        # forward process, x_0 is the clean input.
        logits = torch.log(_at(self.q_mats, t, x_0) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(self, x_0, x_t, t):
        # If we have t == 0, then we calculate the distance to x_0
        # Here, we caclulate equation (3) of the paper. Note that the x_0 Q_t x_t^T is a normalizing constant, so we don't deal with that.
        x_0_logits = convert_to_distribution(x_0, self.num_classes, self.eps)
        fact1 = _at(self.q_one_step_transposed, t, x_t) # x_t | x_{t-1}
        trans_mats = self.q_mats[t - 1].to(dtype=x_0_logits.dtype)
        if self.fix_x_t_bias:
            x_0_logits -= torch.log(self.q_mats[t, :, x_t] + self.eps) # x_{t} | x_{0}
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        fact2 = torch.einsum("b...c,b...cd->b...d", softmaxed, trans_mats) # x_{t-1} | x_{0}
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        
        # check if t==0
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        bc = torch.where(t_broadcast == 0, x_0_logits, out)
        return bc

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        t, S, x_t = self.sample_point(x)
        
        # predict x_0 and x_{t-1}
        predicted_x0_logits = self.model_predict(x_t, t, cond)
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t)

        # get kls and loss
        kl = kls(true_q_posterior_logits, pred_q_posterior_logits, self.eps) # shape x
        vb_loss = kl.mean() * self.n_T

        # Also calculate cross entropy loss
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        return self.hybrid_loss_coeff * ce_loss + vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def sample_with_image_sequence(self, x, cond=None, stride=10):
        steps = 0
        images = []
        pbar = tqdm(np.arange(0, self.n_T)[::-1], position=0, leave=True)
        for t in pbar:
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x_next = self.p_sample(
                x, t, cond, torch.rand((*x.shape, self.num_classes), device=x.device)
            )
            x = x_next

            steps += 1
            if steps % stride == 0:
                images.append(x)

        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images
