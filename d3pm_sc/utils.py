import numpy as np
import torch
import torch.nn
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
    out = torch.softmax(dist1 + eps, dim=-1) * (
        torch.log_softmax(dist1 + eps, dim=-1)
        - torch.log_softmax(dist2 + eps, dim=-1)
    )
    return out.sum(dim=-1)#.mean()

def convert_to_distribution(x_0, num_classes, eps):
    # returns log probs of x_0 as a distribution
    if x_0.dtype == torch.int64 or x_0.dtype == torch.int32:
        x_0_logits = torch.log(
            torch.nn.functional.one_hot(x_0, num_classes) + eps
        )
    else:
        x_0_logits = x_0.clone()
    return x_0_logits

def get_L_and_K(forward_kwargs, num_classes, gamma):
    if forward_kwargs['type'] == "uniform":
        L = torch.ones(num_classes, num_classes) / (num_classes-1)
        L.diagonal().fill_(-1)
        rate = - (L.diagonal().min()) / (1-gamma) # L^* in sec 6.6 of the notes
        K = L / rate + torch.eye(num_classes)
    elif forward_kwargs['type'] == "gaussian":
        bandwidth = forward_kwargs['bandwidth']
        range_ = torch.arange(num_classes)
        diff_mat = (range_[:, None] - range_[None, :]) ** 2
        L = torch.exp(- diff_mat / (2 * (bandwidth * num_classes) ** 2))
        L = L / (L.sum(-1).max() - 1)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
        rate = - (L.diagonal().min()) / (1-gamma) # L^* in sec 6.6 of the notes
        K = L / rate + torch.eye(num_classes)
    elif forward_kwargs['type'] == "bert_embed":
        embeds = BertModel.from_pretrained("bert-base-uncased").embeddings.word_embeddings.weight
        embeds = embeds.detach().cpu().numpy()
        
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds_normalized = embeds / norms
        
        print("Constructing nearest neighbor matrix...")
        
        k = 11
        index = faiss.IndexFlatIP(embeds.shape[1]) 
        index.add(embeds_normalized)
        distances, indices = index.search(embeds_normalized, k)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        row_indices = np.repeat(np.arange(embeds.shape[0]), k - 1) 
        col_indices = indices.flatten()
        data_rows = embeds[row_indices]
        data_cols = embeds[col_indices]
        dot_products = np.einsum('ij,ij->i', data_rows, data_cols)
        
        sparse_matrix = coo_matrix((dot_products, (row_indices, col_indices)), shape=(embeds.shape[0], embeds.shape[0]))
        sparse_matrix_csr = sparse_matrix.tocsr()
        
        print("Finished constructing nearest neighbor matrix...")
        
        L = sparse_matrix_csr
        rate = - (L.diagonal().min()) / (1-gamma) 
        K = L / rate + scipy.sparse.eye(L.shape[0])
    
    if "normalize" in forward_kwargs.keys() and forward_kwargs['normalize']:
        L = L / (- L.diagonal()[:, None])
        range_ = torch.arange(num_classes)
        L.diagonal().fill_(0)
        L[range_, range_] = -L.sum(-1)
        
        rate = - (L.diagonal().min()) / (1-gamma) 
        K = L / rate + torch.eye(num_classes)
        
    return L, K, rate
