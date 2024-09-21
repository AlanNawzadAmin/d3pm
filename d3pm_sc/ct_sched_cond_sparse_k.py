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
        sedd_param=True,
        eff_num_classes=1000000,
        **kwargs
    ):
        # Precalculate betas, define model_predict, p_sample
        super().__init__(x0_model_class, nn_params, num_classes, schedule_type, hybrid_loss_coeff, **kwargs)
        self.save_hyperparameters(ignore=['x0_model_class'])
        self.cache_fact2 = True
        self.sedd_param = sedd_param
        self.eff_num_classes = min([eff_num_classes, num_classes])
        self.forward_kwargs = forward_kwargs
        self.gamma = gamma
        assert gamma >= 0 and gamma < 1 # full schedule and classical resp.
        # not gonna implement fix_x_t_bias, input_logits, logistic_pars

    def calc_stationary(self):
        K_T_gpu = self.K_T.cuda()
        stat_gpu = self.stat.cuda()
        stationary = torch.ones(K_T_gpu.shape[0]).float().cuda()
        stationary = stationary / stationary.sum()
        pbar = tqdm(range(100000))
        for i in pbar:
            stationary_new = self.K_T_operator(K_T_gpu, stationary, stat_gpu)
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
        p0_inds = torch.flip(torch.argsort(self.p0), (0,))
        p0_rank = torch.argsort(p0_inds)
        self.register_buffer("p0_inds", p0_inds)
        self.register_buffer("p0_rank", p0_rank)
        if self.eff_num_classes < self.num_classes:
            self.freq_order = True
            mat_inds = p0_inds[:self.eff_num_classes]
        else:
            self.freq_order = False
            self.eff_num_classes = elf.num_classes
            mat_inds = None
        
        # Precalculate K_powers of KNN process
        L, K, L_cpu, K_cpu, rate = get_L_and_K(self.forward_kwargs, self.gamma, mat_inds)

        self.rate = max([rate, 1 if self.freq_order else 1e-6])
        self.register_buffer("K", K)
        self.register_buffer("L", L)
        K_coo = K.to_sparse_coo()
        K_T = K_coo.transpose(0, 1).coalesce().to_sparse_csr()
        L_coo = L.to_sparse_coo()
        L_T = L_coo.transpose(0, 1).coalesce().to_sparse_csr()
        self.register_buffer("K_coo", K_coo)
        self.register_buffer("K_T", K_T)
        self.register_buffer("L_T", L_T)

        print("Setting up K and L operators...")
        # Add in uniform process
        self.unif_to_stat = self.forward_kwargs['unif_to_stat']
        stat = (self.p0[:self.eff_num_classes] / self.p0[:self.eff_num_classes].sum() if self.unif_to_stat
                else torch.ones(self.eff_num_classes) / self.eff_num_classes)
        self.register_buffer("stat", stat)
        self.unif_rate = self.forward_kwargs['uniform_rate'] * (1 if self.freq_order else torch.tensor(1-self.stat).max())
        self.rate = self.rate + self.unif_rate
        self.up = self.unif_rate / self.rate
        print("Uniform rate is:", self.up)
        def K_T_operator(K_T, x_0_probs, stat): # assumes x_0_probs are normalized
            if len(x_0_probs) > self.eff_num_classes:
                x_0_probs_top = x_0_probs[:self.eff_num_classes]
                other_part = x_0_probs[self.eff_num_classes:].sum(0)
            else:
                x_0_probs_top = x_0_probs
                other_part = 0
            stat_broadcast = stat.view((self.eff_num_classes,) + (1,) * (x_0_probs_top.dim()-1))
            return (1-self.up) * (K_T @ x_0_probs_top) + (self.up + other_part*(1-self.up)) * stat_broadcast
        def K_operator_vec(K, x_t_probs, stat):        
            """ Ignores rare tokens in input and output! """
            return (1-self.up) * (K @ x_t_probs) + self.up * stat @ x_t_probs
        def K_operator(x_t_index):
            """ Gives wrong answer when x_t has rare tokens! """
            struct = self.K_coo.index_select(1, torch.clamp(x_t_index.flatten(), max=self.eff_num_classes-1)
                                            ).to_dense().T.reshape(
                *x_t_index.shape, self.eff_num_classes)
            unif_part = self.stat[torch.clamp(x_t_index, max=self.eff_num_classes-1)].unsqueeze(-1)   
            struct = (1-self.up) * struct + self.up *  unif_part
            return torch.cat([struct, unif_part.expand(*unif_part.shape[:-1], self.num_classes - self.eff_num_classes)], dim=-1)
        self.K_T_operator = K_T_operator
        self.K_operator_vec = K_operator_vec
        self.K_operator = K_operator
        self.second_eval = self.up
        self.calc_stationary()

        print("Getting Mutual info schedule")
        mis = []
        p0_gpu = self.p0.cuda()
        p0_gpu = p0_gpu / p0_gpu.sum()
        stat_cuda = self.stat.cuda()
        mat = torch.eye(self.num_classes).float().cuda()
        K_T_gpu = self.K_T.cuda()
        ent_p0 = -torch.xlogy(p0_gpu, p0_gpu).sum()
        pbar = tqdm(np.arange(int(10/self.second_eval)))
        for i in pbar:
            p = p0_gpu[:, None] * mat.T
            p = torch.where(p < 0, 0, p)
            p_sum = p.sum(0)
            mi = (torch.xlogy(p, p).sum() - torch.xlogy(p_sum, p_sum).sum()) / ent_p0 + 1
            mis.append(mi)
            if mi < 1e-5:
                break
            pbar.set_description(f"MI:{mi.item()}")
            mat = self.K_T_operator(K_T_gpu, mat, stat_cuda)
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
                flat_S = S.flatten()
                ctx.save_for_backward(flat_S)
                shape = x_0.shape
                liks = x_0.reshape(-1, x_0.shape[-1]).clone().T
                max_power = S.max().item()
                power_mask = (torch.arange(1, max_power + 1, device=S.device).unsqueeze(1)
                              <= flat_S.unsqueeze(0))
                # S=1
                active = power_mask[0]
                liks[:self.eff_num_classes, active] = self.K_T_operator(
                        self.K_T, liks[:, active], self.stat)
                liks[self.eff_num_classes:, active] = 0
                submat = liks[:self.eff_num_classes, :]
                # S>1
                for i in np.arange(1, max_power):
                    active = power_mask[i]
                    submat[:, active] = self.K_T_operator(self.K_T, submat[:, active], self.stat)
                return liks.T.reshape(shape)
        
            @staticmethod
            def backward(ctx, grad_output):
                flat_S, = ctx.saved_tensors
                shape = grad_output.shape
                x_grad = grad_output.reshape(-1, grad_output.shape[-1]).T
                max_power = S.max().item()
                power_mask = (torch.arange(1, max_power + 1, device=S.device).unsqueeze(1)
                              <= flat_S.unsqueeze(0))
                final_mask = (torch.arange(1, max_power + 1, device=S.device).unsqueeze(1)
                              == flat_S.unsqueeze(0))
                for i in range(max_power):
                    if self.freq_order:
                        x_grad[self.eff_num_classes:, final_mask[i]] = self.stat @ x_grad[:self.eff_num_classes, final_mask[i]]
                    x_grad[:self.eff_num_classes, power_mask[i]] = self.K_operator_vec(self.K, x_grad[:self.eff_num_classes, power_mask[i]], self.stat)
                return None, x_grad.T.reshape(shape).to(torch.bfloat16)
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
        kl = kls(x_1[..., :self.eff_num_classes], self.get_stationary(), self.eps)
        return kl.mean()

    def x_t_sample(self, x_0, t, noise, S):
        # forward process, x_0 is the clean input.
        t0 = time.time()
        x_0_logits = convert_to_distribution(x_0, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)
        if self.cache_fact2:
            self.cache_sm1_power = self.K_T_power_mult(S-1, softmaxed.float())
            x_t_probs = self.K_T_power_mult((S>0).long(), self.cache_sm1_power)
        else:
            x_t_probs = self.K_T_power_mult(S, softmaxed)
        probs = torch.cumsum(x_t_probs, -1)
        # print("normalization check:", probs[..., [-1]].min(), probs[..., [-1]].max())
        x_t_sample = torch.argmax((probs > noise * probs[..., [-1]]).long(), dim=-1)
        # logits = torch.log(x_t_probs + self.eps)
        # noise = torch.clip(noise, self.eps, 1.0)
        # gumbel_noise = -torch.log(-torch.log(noise))
        # x_t_sample = torch.argmax(logits + gumbel_noise, dim=-1)
        torch.cuda.synchronize()
        print("Time to sample wo noise gen:",  time.time() - t0)
        return x_t_sample

    def q_posterior_logits(self, x_0, x_t, t, S, use_cached_fact2=False):
        x_0_logits = convert_to_distribution(x_0, self.num_classes, self.eps)
        softmaxed = torch.softmax(x_0_logits, dim=-1)
        if use_cached_fact2:
            fact2 = self.cache_sm1_power
            # print("cache check:", ((fact2 - self.K_T_power_mult(S-1, softmaxed.float()))**2).sum())
        else:
            fact2 = self.K_T_power_mult(S-1, softmaxed.float())
        fact1 = self.K_operator(x_t)
        out = torch.log(fact2 + self.eps) + torch.log(fact1 + self.eps)
        return out

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None, *args) -> torch.Tensor:
        if self.freq_order:
            x = self.p0_rank[x]
        t0 = time.time()
        t, S, x_t = self.sample_point(x, 1)
        torch.cuda.synchronize()
        print("Time to sample:",  time.time() - t0)
        # predict x_0 and prev(x_t)
        t0 = time.time()
        predicted_x0_logits = self.model_predict(x_t, t, cond, S)
        torch.cuda.synchronize()
        print("Time to predict:",  time.time() - t0)
        t0 = time.time()
        true_q_posterior_logits = self.q_posterior_logits(x, x_t, t, S, use_cached_fact2=self.cache_fact2)
        if self.sedd_param:
            pred_q_posterior_logits = predicted_x0_logits
        else:
            pred_q_posterior_logits = self.q_posterior_logits(predicted_x0_logits, x_t, t, S)
        torch.cuda.synchronize()
        print("Time to get logits:",  time.time() - t0)
        
        # get kls and loss
        t0 = time.time()
        kl = kls(true_q_posterior_logits, pred_q_posterior_logits, self.eps) # shape x
        weight = - self.beta(t) / self.log_alpha(t)
        weight = (S.swapaxes(0, -1) * weight).swapaxes(0, -1)
        vb_loss = (kl * weight).mean() * self.t_max
        torch.cuda.synchronize()
        print("Time to get kls:",  time.time() - t0)

        # Also calculate cross entropy loss
        predicted_x0_logits = predicted_x0_logits.flatten(start_dim=0, end_dim=-2)
        x = x.flatten(start_dim=0, end_dim=-1)
        ce_loss = torch.nn.CrossEntropyLoss()(predicted_x0_logits, x)

        print(vb_loss)
        print(ce_loss)
        return self.hybrid_loss_coeff * ce_loss + vb_loss, {
            "vb_loss": vb_loss.detach().item(),
            "ce_loss": ce_loss.detach().item(),
        }

    def sample_with_image_sequence(self, *args, **kwargs):
        return None

    # def sample_with_text_sequence(self, x, cond=None, n_T=200, stride=10):
    #     print(1)
    #     t = self.t_max * torch.ones(x.shape[0], device=x.device)
    #     S = sample_n_transitions_cont(self.log_alpha, x[0].flatten().shape[0], t)
    #     S = S.swapaxes(0, 1).reshape(*x.shape).long()
    #     steps = 0
    #     images = []
    #     n_steps = S.sum(-1).sum(-1).sum(-1).max().item()
    #     if n_steps > 1e6:
    #         print("n_steps:", n_steps)
    #         return None
    #     pbar = tqdm(total=n_steps, unit="iteration",
    #                 position=0, leave=True)
    #     trans_step = n_steps // n_T
    #     while S.sum() > 0:
    #         # predict what comes next
    #         x_next = self.p_sample(
    #             x, t, cond, torch.rand((*x.shape, self.num_classes), device=x.device), S
    #         )
    #         for b in range(len(x)):
    #             trans_indices = torch.argwhere(S[b] > 0)
    #             trans_indices = trans_indices[torch.randperm(len(trans_indices))]
    #             if len(trans_indices) > 0:
    #                 # randomly transiiton
    #                 for k in trans_indices[:trans_step]:
    #                     x[b, k] = x_next[b, k]
    #                     S[b, k] -= 1
    #         pbar.update(trans_step)
    #         steps += 1
    #         if steps % stride == 0:
    #             images.append(torch.clone(x))
    #     pbar.close()
    #     # if last step is not divisible by stride, we add the last image.
    #     if steps % stride != 0:
    #         images.append(x)

    #     return images

