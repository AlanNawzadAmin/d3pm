import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import kls, convert_to_distribution
from .schedule_sample import sample_n_transitions_cont
from d3pm_sc.ct_sched_cond import ScheduleCondition


class MaskingDiffusion(ScheduleCondition): #schedule conditioning is True!
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes: int = 10,
        schedule_type="cos",
        hybrid_loss_coeff=0.01,
        fix_x_t_bias=False,
        logistic_pars=False,
        **kwargs,
    ):
        forward_kwargs={"type":"uniform"}
        gamma = 1 / num_classes
        if 'gamma' in kwargs:
            del kwargs['gamma']
        super().__init__(x0_model_class, nn_params, num_classes, forward_kwargs, schedule_type, gamma, hybrid_loss_coeff,
                         fix_x_t_bias, logistic_pars, **kwargs)
        # with this choice, x_t_sample is uniform and 
        # q_posterior_logits returns uniform if S>1 and x_0 pred if S==1
        # The only differences is the predictions and marginalizing over S>1 in the weight
        # so we always assume S==1.
        # in principle we could also speed up sampling by ignoring S>1

    def model_predict(self, x_t, t, cond, S):
        masked_pos = S > 0
        masked_x_t = torch.where(masked_pos, self.num_classes, x_t)
        return self.x0_model(masked_x_t, t, cond, S=None)[..., :-1]
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        t, S, x_t = self.sample_point(x)
        S = (S>0).long()
        
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, cond, S)
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)

        # get kls and loss
        kl = kls(true_q_posterior_logits, pred_q_posterior_logits, self.eps) # shape x
        alpha_t = torch.exp(self.log_alpha(t))
        weight = self.beta(t) * alpha_t / (1 - alpha_t) #mult by p(S=1|t)
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
        S = (S>0).swapaxes(0, 1).reshape(*x.shape).long() # this is the only line changed
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