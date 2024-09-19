import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import time

from .utils import kls, convert_to_distribution
from .schedule_sample import sample_n_transitions_cont
from .continuous_time_diffusion import ContinuousTimeDiffusion

from .language_model_inf_gen import get_L_and_K

class ScheduleConditionSparseK(ContinuousTimeDiffusion): #schedule conditioning is True!
    def __init__(
        self,
        x0_model_class,
        nn_params,
        num_classes: int = 10,
        forward_kwargs={"type":"uniform"},
        schedule_type="cos",
        gamma=0,
        hybrid_loss_coeff=0.01,
        **kwargs
    ):
        # Precalculate betas, define model_predict, p_sample
        super().__init__(x0_model_class, nn_params, num_classes, schedule_type, hybrid_loss_coeff, **kwargs)
        self.save_hyperparameters(ignore=['x0_model_class'])
        assert gamma >= 0 and gamma < 1 # full schedule and classical resp.
        # not gonna implement fix_x_t_bias, input_logits, logistic_pars
        
        # Precalculate K_powers of KNN process
        L, K, L_cpu, K_cpu, rate = get_L_and_K(forward_kwargs, gamma)

        self.register_buffer("L", L)
        K_coo = K.to_sparse_coo()
        L_coo = L.to_sparse_coo()
        L_T = L_coo.transpose(0, 1).coalesce().to_sparse_csr()
        self.register_buffer("K_coo", K_coo)
        self.register_buffer("L_T", L_T)

        # Add in uniform process
        self.unif_rate = forward_kwargs['uniform_rate']
        self.rate = rate + self.unif_rate
        self.up = self.unif_rate / self.rate
        print("Uniform rate is:", self.up)
        def K_T_operator(L_T, x_0_probs):
            return (x_0_probs
                    + ((1-self.up) * L_T @ x_0_probs
                    +  self.up * (x_0_probs.mean(0) - x_0_probs) * (self.num_classes / (self.num_classes - 1))))
        def K_operator(x_t_index):
            struct = self.K_coo.index_select(1, x_t_index.flatten()
                                            ).to_dense().T.reshape(
                *x_t_index.shape, self.num_classes)
            uniform = torch.eye(self.num_classes,
                                device=x_t_index.device, dtype=torch.float32)
            uniform = (1 - uniform[x_t_index, :]) / (self.num_classes - 1)
            return (1-self.up) * struct + self.up * uniform
        self.K_T_operator = K_T_operator
        self.K_operator = K_operator
        
        self.second_eval = self.up * (self.num_classes / (self.num_classes - 1))
        self.calc_stationary()

    def calc_stationary(self):
        L_T_gpu = self.L_T.cuda()
        stationary = torch.ones(L_T_gpu.shape[0]).float().cuda()
        pbar = tqdm(range(100000))
        for i in pbar:
            stationary_new = self.K_T_operator(L_T_gpu, stationary)
            err = torch.sqrt(((stationary_new - stationary) ** 2).sum())
            if torch.allclose(err, torch.zeros_like(err)):
                break
            stationary = stationary_new
            if i%1000 == 0:
                pbar.set_description(f"Getting stationary. err:{err.item()}")
        stationary = stationary / (stationary ** 2).sum()
        self.register_buffer("stationary", stationary)

    def pre_configure_model(self, dataloader):
        if not hasattr(self, 'p0'):
            self.calc_p0(dataloader)
        mis = []
        p0_gpu = self.p0.cuda()
        p0_gpu = p0_gpu / p0_gpu.sum()
        mat = torch.eye(self.num_classes).float().cuda()
        L_T_gpu = self.L_T.cuda()
        stationary_gpu = self.stationary.cuda()
        ent_p0 = -torch.xlogy(p0_gpu, p0_gpu).sum()
        print("Getting Mutual info schedule")
        pbar = tqdm(range(int(10/self.second_eval)))
        for i in pbar:
            p = p0_gpu[:, None] * mat.T
            p = torch.where(p < 0, 0, p)
            p_sum = p.sum(0)
            mi = (torch.xlogy(p, p).sum(-1) - torch.xlogy(p_sum, p_sum)).sum(-1) / ent_p0 + 1
            mis.append(mi)
            if mi < 1e-5:
                break
            pbar.set_description(f"MI:{mi.item()}")
            mat = self.K_T_operator(L_T_gpu, mat)
        self.precompute_mis = mis
        self.log_alpha, self.beta, *_ = self.get_beta_func(None, self.p0, type_='schedule_condition', scale=self.rate, precompute_mis=mis, second_eval=self.second_eval)
        del mat
        import gc
        torch.cuda.empty_cache()
        gc.collect()

    def K_T_power_mult(self, S, x_0, period=1):
        shape = x_0.shape
        x_0 = x_0.reshape(-1, x_0.shape[-1]).T
        curr_S = S.reshape(-1)
        curr_liks = x_0[:, curr_S >= 0]
        liks = torch.ones_like(x_0)
        while torch.any(curr_S > 0):
            active = curr_S >= 0
            liks[:, curr_S == 0] = curr_liks[:, (curr_S == 0)[active]]
            if curr_liks.shape[-1] == 1:
                if not all((curr_S > 0)[active]):
                    break
            else:
                curr_liks = curr_liks[:, (curr_S > 0)[active]]
            curr_liks = self.K_T_operator(self.L_T, curr_liks)
            curr_S = curr_S - 1
        if curr_liks.shape[-1] > 0:
            liks[:, curr_S == 0] = curr_liks
        return liks.T.reshape(shape)
    
    def get_stationary(self):
        return self.stationary / self.stationary.sum()

    def get_kl_t1(self, x):
        # sample S
        t = self.t_max * torch.ones(x.shape[0], device=x.device)
        S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
        S = S.swapaxes(0, 1).reshape(*x.shape).long()
        x_0_logits = convert_to_distribution(x, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)
        x_1 = self.K_T_power_mult(S, softmaxed)
        kl = kls(x_1, self.get_stationary(), self.eps)
        return kl.mean()

    def x_t_sample(self, x_0, t, noise, S):
        # forward process, x_0 is the clean input.
        x_0_logits = convert_to_distribution(x_0, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)
        logits = torch.log(self.K_T_power_mult(S, softmaxed) + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(self, x_0, x_t, t, S):
        x_0_logits = convert_to_distribution(x_0, self.num_classes, self.eps)
        fact1 = self.K_operator(x_t)
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        fact2 = self.K_T_power_mult(S-1, softmaxed.float())
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        
        # check if t==0; then we calculate the distance to x_0
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        bc = torch.where(t_broadcast == 0, x_0_logits, out)
        return bc

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, *args) -> torch.Tensor:
        # t0 = time.time()
        t, S, x_t = self.sample_point(x)
        torch.cuda.synchronize()
        # print("Time to sample:",  time.time() - t0)
        # predict x_0 and prev(x_t)
        # t0 = time.time()
        predicted_x0_logits = self.model_predict(x_t, t, cond, S)
        torch.cuda.synchronize()
        # print("Time to predict:",  time.time() - t0)
        # t0 = time.time()
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)
        torch.cuda.synchronize()
        # print("Time to get logits:",  time.time() - t0)
        
        # get kls and loss
        kl = kls(true_q_posterior_logits, pred_q_posterior_logits, self.eps) # shape x
        weight = - self.beta(t) / self.log_alpha(t)
        weight = (S.swapaxes(0, -1) * weight).swapaxes(0, -1)
        vb_loss = (kl * weight).mean() * self.t_max

        # Also calculate cross entropy loss
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        print(vb_loss)
        return self.hybrid_loss_coeff * ce_loss + vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def sample_with_image_sequence(self, *args, **kwargs):
        return None

