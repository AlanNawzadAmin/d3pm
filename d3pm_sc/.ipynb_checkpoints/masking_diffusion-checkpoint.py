import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import kls, convert_to_distribution
from .schedule_sample import sample_n_transitions_cont
from d3pm_sc.ct_sched_cond import SchdeuleCondition


class MaskingDiffusion(SchdeuleCondition): #schedule conditioning is True!
    def __init__(
        self,
        x0_model: nn.Module,
        num_classes: int = 10,
        schedule_type="cos",
        hybrid_loss_coeff=0.001,
    ):
        forward_type = "uniform"
        gamma = 1/num_classes
        super().__init__(x0_model, num_classes, schedule_type, hybrid_loss_coeff,
                         gamma=gamma, forward_type=forward_type)
        # with this choice, x_t_sample is uniform and 
        # q_posterior_logits returns uniform if S>1 and x_0 pred if S==1

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        t, S, x_t = self.sample_point(x)
        masked_pos = S > 0
        x_t[masked_pos] = self.num_classes # mask these positions
        
        # predict x_0 and prev(x_t)
        predicted_x0_logits = self.model_predict(x_t, t, cond, S=None) # use masks, not S
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)

        # get kls and loss
        kl = kls(true_q_posterior_logits, pred_q_posterior_logits, self.eps)
        weight = - self.beta_scale(t) * self.alpha_scale(t) / (1 - self.alpha_scale(t))
        weight = (S.swapaxes(0, -1) * weight).swapaxes(0, -1) # mult by S to ignore S=0
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
        S = sample_n_transitions_cont(self.alpha_scale, x[0].flatten().shape[0], t)
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
                images.append(x)
        pbar.close()
        # if last step is not divisible by stride, we add the last image.
        if steps % stride != 0:
            images.append(x)

        return images
