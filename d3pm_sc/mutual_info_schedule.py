import torch
from d3pm_sc.root_finder import root_finder, newton_root_finder

def get_a_b_func_cont(L, p0, **kwargs):
    ent_p0 = -torch.xlogy(p0, p0).sum()
    evals, V = torch.linalg.eig(L)
    evals[torch.real(evals) > -1e-6] = 0
    second_eval = torch.real(evals)
    second_eval = -second_eval.sort().values[-2]
    V_inv = torch.linalg.inv(V)
    def mi(t, t_shift=1):
        """ t_shift is 1-t """
        evals_skew = torch.exp(t[:, None, None] * evals[None, None, :])
        too_big = torch.real(evals_skew * evals_skew.conj()) > 1
        evals_skew = torch.where(too_big, 1, evals_skew)
        mat = torch.where((t > 1e-3)[:, None, None],
                          torch.real((V[None,:, :] * evals_skew) @ V_inv), 
                          torch.eye(len(L)) + t[:, None, None] * L) # stable for small t
        p = p0[None, :, None] * mat
        p = torch.where(p < 1e-12, 0, p)
        mi_m1 = (torch.xlogy(p, p).sum(-1) - torch.xlogy(p.sum(-2), p.sum(-2))).sum(-1) / ent_p0
        return mi_m1 + t_shift

    base_ts = torch.linspace(0., 0.9991, 10000) 
    def log_alpha_naive(ts):
        out = newton_root_finder(mi, 1/second_eval, ts)
        # out = root_finder(mi, 0, 20/second_eval, ts)
        return -torch.where(out>1e-6, out, 1e-6)
    base_alphas = -log_alpha_naive(base_ts)
    def log_alpha(ts):
        closest_index = torch.searchsorted(base_ts, ts.to('cpu'))
        best_guess_l = base_alphas[closest_index-1]
        # best_guess_u = base_alphas[closest_index]
        out = newton_root_finder(mi, best_guess_l, ts.to('cpu'))
        # out = root_finder(mi, best_guess_l, best_guess_u, ts)
        return -torch.where(out>1e-6, out, 1e-6).to(ts.device)
    
    def beta(ts):
        out_tensor = -log_alpha(ts.to('cpu'))
        grad = -1 / torch.func.grad(lambda o: mi(o).sum())(out_tensor)
        return grad.to(ts.device)
    return log_alpha, beta, mi


def get_a_b_func_sc(K, p0, precompute_mis=None, second_eval=None, **kwargs):
    max_n = int(40 / second_eval)
    ent_p0 = -torch.xlogy(p0, p0).sum()
    if precompute_mis is None:
        evals, V = torch.linalg.eig(K - torch.eye(len(K), dtype=K.dtype))
        evals[torch.real(evals) > -1e-6] = 0
        second_eval = torch.real(evals)
        second_eval = -second_eval.sort().values[-2]
        V_inv = torch.linalg.inv(V)
        def mi_p(n):
            if n > 0:
                mat = torch.real((V * ((1+evals[None, :]) ** n)) @ V_inv)
            else:
                mat = torch.eye(len(K), dtype=K.dtype)
            p = p0[:, None] * mat
            p = torch.where(p < 0, 0, p)
            return (torch.xlogy(p, p).sum(-1) - torch.xlogy(p.sum(-2), p.sum(-2))).sum(-1) / ent_p0 + 1
        precompute_mis = [mi_p(n) for n in range(max_n)]
    precompute_mis = torch.tensor(precompute_mis)
    precompute_mis = torch.maximum(precompute_mis, torch.zeros(1))
    max_n = (precompute_mis > 1e-7).sum()
    precompute_mis = torch.cummin(precompute_mis[:max_n], 0)[0]
    range_ = torch.arange(max_n, dtype=p0.dtype)
    precompute_log_factorial = torch.lgamma(1+range_)
    def mi(t, t_shift=1):
        log_probs = (range_ * torch.log(t[:, None]+1e-6) - t[:, None]
                     - precompute_log_factorial)
        return torch.exp(log_probs) @ precompute_mis - 1 + t_shift

    base_ts = torch.linspace(0., 0.9991, 10000) 
    def log_alpha_naive(ts):
        out = newton_root_finder(mi, max_n/20, ts)
        # out = root_finder(mi, 0, max_n/20, ts)
        return -torch.where(out>1e-6, out, 1e-6)
    base_alphas = -log_alpha_naive(base_ts)
    base_alphas = torch.cummax(base_alphas, 0)[0] # fix any errors
    def log_alpha(ts):
        closest_index = torch.searchsorted(base_ts, ts.to('cpu'))
        best_guess_l = base_alphas[closest_index-1]
        # best_guess_u = base_alphas[closest_index]
        out = newton_root_finder(mi, best_guess_l, ts.to('cpu'))
        # out = root_finder(mi, best_guess_l, best_guess_u, ts)
        return -torch.where(out>1e-6, out, 1e-6).to(ts.device)
    
    def beta(ts):
        out_tensor = -log_alpha(ts.to('cpu'))
        grad = -1 / torch.func.grad(lambda o: mi(o).sum())(out_tensor)
        return grad.to(ts.device)
    return log_alpha, beta, mi, precompute_mis

def get_a_b_func_mi(mat, p0, type_, **kwargs):
    if type_ == 'schedule_condition':
        return get_a_b_func_sc(mat, p0, **kwargs)
    elif type_ == 'SEDD':
        return get_a_b_func_cont(mat, p0, **kwargs)
