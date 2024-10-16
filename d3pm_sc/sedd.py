import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time

from .utils import kls, convert_to_distribution, get_inf_gen
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
        hybrid_loss_coeff=0.01,
        fix_x_t_bias=False,
        logistic_pars=False,
        **kwargs
    ):
        # Precalculate betas, define model_predict, p_sample
        super().__init__(x0_model_class, nn_params, num_classes, schedule_type, hybrid_loss_coeff, logistic_pars, **kwargs)
        self.save_hyperparameters(ignore=['x0_model_class'])
        self.fix_x_t_bias = fix_x_t_bias

        # Get L
        L = get_inf_gen(forward_kwargs, num_classes)
        self.register_buffer("L", L)
        eigenvalues, eigenvectors = torch.linalg.eig(L.double())
        eigenvalues[torch.real(eigenvalues) > -self.eps] = 0
        eigenvectors_inv = torch.linalg.inv(eigenvectors)
        self.register_buffer("eigenvalues", eigenvalues)
        self.register_buffer("eigenvectors", eigenvectors)
        self.register_buffer("eigenvectors_inv", eigenvectors_inv)
        # Get Ks just for fast sampling
        gamma = 0
        rate = - (L.diagonal().min()) / (1-gamma)
        K = L / rate + torch.eye(num_classes)
        self.rate = rate
        
        # Precalculate K_powers
        num_powers = 5000
        assert (num_classes <= 512 and forward_kwargs['type'] != "bert_embed")
        K_powers = torch.stack([torch.linalg.matrix_power(K, i) for i in tqdm(range(5000), desc="Making K powers")])
        self.register_buffer("K", K)
        self.register_buffer("K_powers", K_powers)

    def pre_configure_model(self, dataloader):
        self.calc_p0(dataloader)
        self.log_alpha, self.beta, *_ = self.get_beta_func(self.L, self.p0, 'SEDD')
        
    def get_stationary(self):
        evals, evecs = torch.linalg.eig(self.L.T)
        norms_sq = torch.real(evals)
        max_eval = evals[torch.argmax(norms_sq)]
        assert torch.sqrt(torch.real(max_eval * max_eval.conj())) < self.eps
        stationary = evecs[:, torch.argmax(norms_sq)]
        assert torch.allclose(torch.imag(stationary), torch.tensor(0, dtype=self.L.dtype))
        stationary = torch.real(stationary)
        stationary = stationary * torch.sign(stationary)
        assert torch.all(stationary >= 0)
        return stationary / stationary.sum()

    # def get_trans_mats_mvp(self, t, v):
    #     mat = torch.matrix_exp(- self.log_alpha(t)[..., None, None] * self.L)
    #     return torch.einsum("b...c,bcd->b...d", v, mat)

    def get_trans_mats_mvp(self, t, v):
        # 
        """ v is a b...c matrix where L is cd, and t is b """
        dv = v.to(dtype=self.eigenvectors.dtype).reshape(v.shape[0], -1, v.shape[-1])
        diag = torch.exp(-self.log_alpha(t)[..., None] * self.eigenvalues)
        dv = dv @ self.eigenvectors
        dv = dv * diag.unsqueeze(-2)
        return (dv @ self.eigenvectors_inv).float().reshape(v.shape)

    def get_kl_t1(self, x):
        # sample S
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        x_0_logits = convert_to_distribution(x, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        x_1 = torch.log(self.get_trans_mats_mvp(t, softmaxed)+self.eps)
        # x_1 = torch.log(torch.einsum("b...c,bcd->b...d", softmaxed, self.get_trans_mats(t)))
        kl = kls(x_1, torch.log(self.get_stationary() + self.eps), self.eps)
        return kl.mean()

    def x_t_sample(self, x_0, t, noise, S):
        # forward process, x_0 is the clean input.
        logits = torch.log(self.K_powers[S, x_0, :] + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def r_posterior(self, x_0, x_t, t, S): # returns backward inf_gen
        x_0_logits = convert_to_distribution(x_0, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        p_y = self.get_trans_mats_mvp(t, softmaxed)
        p_xt = p_y.gather(-1, x_t.unsqueeze(-1)).squeeze(-1)
        ratios = (p_y + self.eps)/(p_xt[..., None] + self.eps)
        ratios = ((ratios * self.L.T[x_t, :]).transpose(0, -1) * self.beta(t)).transpose(0, -1)
        return ratios

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, attn_mask=None, *args) -> torch.Tensor:
        t, S, x_t = self.sample_point(x)
        print(S.float().mean())
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, cond if cond is not None else attn_mask, None)
        true_r_posterior = self.r_posterior(x, x_t, t, None)
        pred_r_posterior = self.r_posterior(predicted_x0_logits, x_t, t, None)

        # get kls and loss
        kl = (- (true_r_posterior * torch.log(F.relu(pred_r_posterior)+self.eps)
                 - pred_r_posterior)
              + (true_r_posterior * torch.log(F.relu(true_r_posterior)+self.eps)
                 - true_r_posterior))
        kl = (kl * (true_r_posterior>=0)).sum(-1)
        if attn_mask is not None:
            kl = kl * attn_mask
        vb_loss = kl.mean() * self.t_max
        if attn_mask is not None:
            vb_loss = vb_loss / attn_mask.mean()
        print(vb_loss)

        # Also calculate cross entropy loss
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss(reduction='none')(predicted_x0_logits, x)
        print(ce_loss.mean())
        if attn_mask is not None:
            ce_loss = (ce_loss * attn_mask.flatten()).sum() / attn_mask.sum()
        else:
            ce_loss = ce_loss.mean()
        return self.hybrid_loss_coeff * ce_loss + vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def p_sample(self, x, t, cond, attn_mask, noise, delta_t, S=None):
        # predict prev(x_t) or x_{t-1}
        predicted_x0_logits = self.model_predict(x, t, cond if cond is not None else attn_mask,None)
        bwd_inf_gen = self.r_posterior(predicted_x0_logits, x, t,None)
        bwd_inf_gen.scatter_(-1, x.unsqueeze(-1),
                             - F.relu(bwd_inf_gen).sum(-1).unsqueeze(-1))
        
        x_0_logits = convert_to_distribution(x, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)
        pred_r_posterior = torch.clip(softmaxed + delta_t * bwd_inf_gen, self.eps, 1.0)
        pred_r_posterior = torch.log(pred_r_posterior)
        # sample
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise)+self.eps)
        sample = torch.argmax(
            pred_r_posterior + gumbel_noise, dim=-1
        )
        return sample


    def sample_sequence(self, x, cond=None, attn_mask=None, n_T=200, stride=10):
        steps = 0
        images = []
        pbar = tqdm(torch.flip(torch.linspace(0, self.t_max, n_T, dtype=torch.float32), (0,)), position=0, leave=True)
        for t in pbar:
            t = torch.tensor([t] * x.shape[0], device=x.device)
            x_next = self.p_sample(
                x, t, cond, attn_mask, torch.rand((*x.shape, self.num_classes), device=x.device), 1/n_T
            )
            x = x_next

            steps += 1
            if steps % stride == 0:
                images.append(x)

        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images
