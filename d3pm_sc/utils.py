import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from transformers import BertModel
import faiss
import scipy
from scipy.sparse import coo_matrix

def _at(a, t, x):
    # t is 1-d, x is integer value of 0 to num_classes - 1
    bs = t.shape[0]
    t = t.reshape((bs, *[1] * (x.dim() - 1)))
    return a[t, x, :]

def kls(dist1, dist2, eps): # KL of dists on last dim
    out = F.kl_div(torch.log_softmax(dist2 + eps, dim=-1),
                   torch.log_softmax(dist1 + eps, dim=-1),
                  log_target=True, reduction='none').sum(-1)
    return out

def convert_to_distribution(x_0, num_classes, eps):
    # returns log probs of x_0 as a distribution
    if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
        x_0_logits = torch.log(
            torch.nn.functional.one_hot(x_0, num_classes) + eps
        )
    else:
        x_0_logits = x_0.clone()
    return x_0_logits

def get_inf_gen(forward_kwargs, num_classes, gamma):
    if forward_kwargs['type'] == "uniform":
        L = torch.ones(num_classes, num_classes) / (num_classes-1)
        L.diagonal().fill_(-1)
    elif forward_kwargs['type'] == "gaussian":
        bandwidth = forward_kwargs['bandwidth']
        range_ = torch.arange(num_classes)
        diff_mat = (range_[:, None] - range_[None, :]) ** 2
        L = torch.exp(- diff_mat / (2 * (bandwidth * num_classes) ** 2))
        L = L / (L.sum(-1).max() - 1)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
    if "normalize" in forward_kwargs.keys() and forward_kwargs['normalize']:
        L = L / (- L.diagonal()[:, None])
        range_ = torch.arange(num_classes)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
    return L

def extract_sparse_csr_submatrix(csr_tensor, row_indices, col_indices):
    # Convert to COO for easier manipulation
    coo_tensor = csr_tensor.to_sparse_coo()
    
    # Convert input indices to tensors and sort them for efficient processing
    row_indices = torch.as_tensor(row_indices, device=csr_tensor.device)
    col_indices = torch.as_tensor(col_indices, device=csr_tensor.device)
    sorted_row_indices, row_perm = torch.sort(row_indices)
    sorted_col_indices, col_perm = torch.sort(col_indices)
    
    # Create boolean masks for rows and columns
    row_mask = torch.zeros(csr_tensor.shape[0], dtype=torch.bool, device=csr_tensor.device)
    col_mask = torch.zeros(csr_tensor.shape[1], dtype=torch.bool, device=csr_tensor.device)
    row_mask[sorted_row_indices] = True
    col_mask[sorted_col_indices] = True
    
    # Apply masks to get desired indices and values
    mask = row_mask[coo_tensor.indices()[0]] & col_mask[coo_tensor.indices()[1]]
    filtered_indices = coo_tensor.indices()[:, mask]
    filtered_values = coo_tensor.values()[mask]
    
    # Remap row and column indices
    row_map = torch.empty_like(row_mask, dtype=torch.long)
    col_map = torch.empty_like(col_mask, dtype=torch.long)
    row_map[sorted_row_indices] = torch.arange(len(row_indices), device=csr_tensor.device)
    col_map[sorted_col_indices] = torch.arange(len(col_indices), device=csr_tensor.device)
    
    new_rows = row_map[filtered_indices[0]]
    new_cols = col_map[filtered_indices[1]]
    
    # Reorder to match input order
    new_rows = row_perm[new_rows]
    new_cols = col_perm[new_cols]
    
    # Create new sparse tensor
    new_indices = torch.stack([new_rows, new_cols])
    new_shape = (len(row_indices), len(col_indices))
    new_coo = torch.sparse_coo_tensor(new_indices, filtered_values, new_shape)
    
    # Convert back to CSR
    return new_coo.to_sparse_csr()

def get_sort_S(S):
    S_flat, sort = torch.sort(S.flatten(), descending=True)
    S_sort = S_flat.reshape(S.shape)
    unsort = torch.zeros_like(sort)
    unsort[sort] = torch.arange(len(S_flat), device=S_flat.device)
    return S_sort, sort, unsort
    
def get_counts_S_flat(S_flat):
    unique, counts = torch.unique(torch.clamp(S_flat, min=0), return_counts=True)
    full_counts = torch.zeros(unique.max()+1, device=unique.device, dtype=torch.long)
    full_counts[unique] = counts
    return full_counts.flip(0).cumsum(0)


