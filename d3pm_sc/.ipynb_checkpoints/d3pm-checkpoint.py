import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from schedule_sample import sample_n_transitions, sample_full_transitions

# TODO: add KL(p(x_1)||q(x_1)), stop conditioning on t, different alpha schedules

class D3PM(nn.Module): #schedule conditioning is True!
    def __init__(
        self,
        x0_model: nn.Module,
        n_T: int,
        num_classes: int = 10,
        forward_type="uniform",
        gamma=1,
        hybrid_loss_coeff=0.001,
    ) -> None:
        super(D3PM, self).__init__()
        self.x0_model = x0_model
        self.n_T = n_T
        self.hybrid_loss_coeff = hybrid_loss_coeff
        self.eps = 1e-6
        self.num_classses = num_classes
        assert gamma > 0 and gamma <= 1 # classical and full schedule resp.

        # Precalculate betas and Ks
        steps = torch.arange(n_T + 1, dtype=torch.float64) / n_T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        self.beta_t = torch.minimum(
            1 - alpha_bar[1:] / alpha_bar[:-1], torch.ones_like(alpha_bar[1:]) * 0.999
        )
        if forward_type == "uniform":
            L = torch.ones(num_classes, num_classes) / (num_classes-1)
            L.diagonal().fill_(-1)
        rate = - (L.diagonal().min()) / gamma # L^* in sec 6.6 of the notes
        K = L / rate + torch.eye(num_classes)
        self.beta_t = rate * self.beta_t
        K_powers = torch.stack([torch.linalg.matrix_power(K, i) for i in range(n_T)])
        self.register_buffer("K", K)
        self.register_buffer("K_powers", K_powers)

        assert self.K_powers.shape == (
            self.n_T,
            num_classes,
            num_classes,
        ), self.K_powers.shape

    def convert_to_distribution(self, x_0):
        # returns log probs of x_0 as a distribution
        if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
            x_0_logits = torch.log(
                torch.nn.functional.one_hot(x_0, self.num_classses) + self.eps
            )
        else:
            x_0_logits = x_0.clone()
        return x_0_logits

    def model_predict(self, x_0, t, cond, S):
        return self.x0_model(x_0, t, cond, S)

    def kls(self, dist1, dist2): # KL of dists on last dim
        out = torch.softmax(dist1 + self.eps, dim=-1) * (
            torch.log_softmax(dist1 + self.eps, dim=-1)
            - torch.log_softmax(dist2 + self.eps, dim=-1)
        )
        return out.sum(dim=-1)

    def x_t_sample(self, x_0, t, noise, S):
        # forward process, x_0 is the clean input.
        logits = torch.log(self.K_powers[S, x_0, :] + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(self, x_0, x_t, t, S):
        # If we have t == 0, then we calculate the distance to x_0
        # Here, we caclulate equation (3) of the paper. Note that the x_0 Q_t x_t^T is a normalizing constant, so we don't deal with that.
        x_0_logits = self.convert_to_distribution(x_0)
        fact1 = self.K.T[x_t, :] # x_t | x_{t-1}
        trans_mats = self.K_powers[S - 1, :, :] # make sure we ignore S=0 later!
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        fact2 = torch.einsum("b...c,b...cd->b...d", softmaxed, trans_mats) # x_{t-1} | x_{0}
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        
        # check if t==0
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        bc = torch.where(t_broadcast == 0, x_0_logits, out)
        return bc

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        # sample t, S, x_t
        t = torch.randint(0, self.n_T, (x.shape[0],), device=x.device)
        S = sample_n_transitions(self.beta_t.to(x.device), x[0].flatten().shape[0], t)
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        x_t = self.x_t_sample(
            x, t, torch.rand((*x.shape, self.num_classses), device=x.device), S
        )
        
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, cond, S)
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)

        # get kls and loss
        kls = self.kls(true_q_posterior_logits, pred_q_posterior_logits) # shape x
        weight = self.beta_t.to(x.device)[t] / torch.cumsum(self.beta_t.to(x.device), 0)[t]
        weight = (S.swapaxes(0, -1) * weight).swapaxes(0, -1) * self.n_T
        vb_loss = (kls * weight).mean()

        # Also calculate cross entropy loss
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        return self.hybrid_loss_coeff * ce_loss + vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def p_sample(self, x, t, cond, noise, S):
        # predict prev(x_t)
        predicted_x0_logits = self.model_predict(x, t, cond, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x, t, S)
        # sample
        noise = torch.clip(noise, self.eps, 1.0)
        not_first_step = 1#(t != 1).float().reshape((x.shape[0], *[1] * (x.dim())))
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * not_first_step, dim=-1
        )
        return sample

    def sample_with_image_sequence(self, x, cond=None, stride=10):
        # returns x_1
        transitions = sample_full_transitions(self.beta_t.to(x.device), len(x.flatten())).reshape(x.shape + (-1,))
        Ss = torch.cumsum(transitions, -1).long()
        steps = 0
        images = []
        pbar = tqdm(np.arange(1, self.n_T)[::-1], position=0, leave=True)
        for t in pbar:
            S = Ss[..., t]
            transition = transitions[..., t]
            if transition.sum() > 0:
                t = torch.tensor([t] * x.shape[0], device=x.device)
                x_next = self.p_sample(
                    x, t, cond, torch.rand((*x.shape, self.num_classses), device=x.device), S
                )
                x = torch.where(transition, x_next, x)

            steps += 1
            if steps % stride == 0:
                images.append(x)

        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images


