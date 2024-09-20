import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
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
        self.cache_fact2 = True
        assert gamma >= 0 and gamma < 1 # full schedule and classical resp.
        # not gonna implement fix_x_t_bias, input_logits, logistic_pars
        
        # Precalculate K_powers of KNN process
        L, K, L_cpu, K_cpu, rate = get_L_and_K(forward_kwargs, gamma)

        self.register_buffer("K", K)
        self.register_buffer("L", L)
        K_coo = K.to_sparse_coo()
        K_T = K_coo.transpose(0, 1).coalesce().to_sparse_csr()
        L_coo = L.to_sparse_coo()
        L_T = L_coo.transpose(0, 1).coalesce().to_sparse_csr()
        self.register_buffer("K_coo", K_coo)
        self.register_buffer("K_T", K_T)
        self.register_buffer("L_T", L_T)

        # Add in uniform process
        self.unif_rate = forward_kwargs['uniform_rate']
        self.rate = rate + self.unif_rate
        self.up = self.unif_rate / self.rate
        print("Uniform rate is:", self.up)
        def K_T_operator(K_T, x_0_probs): # assumes x_0_probs are normalized
            return (1-self.up) * (K_T @ x_0_probs) + self.up / self.num_classes
        def K_operator(x_t_index):
            struct = self.K_coo.index_select(1, x_t_index.flatten()
                                            ).to_dense().T.reshape(
                *x_t_index.shape, self.num_classes)
            return (1-self.up) * struct + self.up / self.num_classes
        self.K_T_operator = K_T_operator
        self.K_operator = K_operator
        
        self.second_eval = self.up
        self.calc_stationary()

    def calc_stationary(self):
        K_T_gpu = self.K_T.cuda()
        stationary = torch.ones(K_T_gpu.shape[0]).float().cuda()
        stationary = stationary / stationary.sum()
        pbar = tqdm(range(100000))
        for i in pbar:
            stationary_new = self.K_T_operator(K_T_gpu, stationary)
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
        K_T_gpu = self.K_T.cuda()
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
            mat = self.K_T_operator(K_T_gpu, mat)
        self.precompute_mis = mis
        self.log_alpha, self.beta, *_ = self.get_beta_func(None, self.p0, type_='schedule_condition', scale=self.rate, precompute_mis=mis, second_eval=self.second_eval)
        del mat
        import gc
        torch.cuda.empty_cache()
        gc.collect()

    def K_T_power_mult(self, S, x_0):
        class K_T_power_mult_class(Function):
            @staticmethod
            def forward(ctx, S, x_0):
                ctx.save_for_backward(S)
                shape = x_0.shape
                liks = x_0.reshape(-1, x_0.shape[-1]).T
                max_power = S.max().item()
                power_mask = (torch.arange(1, max_power + 1, device=S.device).unsqueeze(1)
                              <= S.flatten().unsqueeze(0))
                for i in range(max_power):
                    active = power_mask[i]
                    liks[:, active] = self.K_T_operator(self.K_T, liks[:, active])
                return liks.T.reshape(shape)
        
            @staticmethod
            def backward(ctx, grad_output):
                S, = ctx.saved_tensors
                shape = grad_output.shape
                x_grad = grad_output.reshape(-1, grad_output.shape[-1]).T
                max_power = S.max().item()
                power_mask = (torch.arange(1, max_power + 1, device=S.device).unsqueeze(1)
                              <= S.flatten().unsqueeze(0))
                for i in range(max_power):
                    active = power_mask[i]
                    x_grad[:, active] = self.K_T_operator(self.K, x_grad[:, active])
                return None, x_grad.T.reshape(shape)
        return K_T_power_mult_class.apply(S, x_0)
    
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
        if self.cache_fact2:
            self.cache_sm1_power = self.K_T_power_mult(S-1, softmaxed)
            x_t_probs = self.K_T_power_mult((S>0).long(), self.cache_sm1_power)
        else:
            x_t_probs = self.K_T_power_mult(S, softmaxed)
        logits = torch.log(x_t_probs + self.eps)
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(self, x_0, x_t, t, S, use_cached_fact2=False):
        x_0_logits = convert_to_distribution(x_0, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)  # bs, ..., num_classes
        if use_cached_fact2:
            fact2 = self.cache_sm1_power
        else:
            fact2 = self.K_T_power_mult(S-1, softmaxed.float())
        fact1 = self.K_operator(x_t)
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        
        # check if t==0; then we calculate the distance to x_0
        t_broadcast = t.reshape((t.shape[0], *[1] * (x_t.dim())))
        bc = torch.where(t_broadcast == 0, x_0_logits, out)
        return bc

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, *args) -> torch.Tensor:
        # t0 = time.time()
        t, S, x_t = self.sample_point(x)
        # torch.cuda.synchronize()
        # print("Time to sample:",  time.time() - t0)
        # predict x_0 and prev(x_t)
        # t0 = time.time()
        predicted_x0_logits = self.model_predict(x_t, t, cond, S)
        # torch.cuda.synchronize()
        # print("Time to predict:",  time.time() - t0)
        # t0 = time.time()
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S, use_cached_fact2=self.cache_fact2)
        pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)
        # torch.cuda.synchronize()
        # print("Time to get logits:",  time.time() - t0)
        
        # get kls and loss
        kl = kls(true_q_posterior_logits, pred_q_posterior_logits, self.eps) # shape x
        weight = - self.beta(t) / self.log_alpha(t)
        weight = (S.swapaxes(0, -1) * weight).swapaxes(0, -1)
        vb_loss = (kl * weight).mean() * self.t_max
        print(weight.min)

        # Also calculate cross entropy loss
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        # print(vb_loss)
        # print(ce_loss)
        return self.hybrid_loss_coeff * ce_loss + vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def sample_with_image_sequence(self, *args, **kwargs):
        return None

