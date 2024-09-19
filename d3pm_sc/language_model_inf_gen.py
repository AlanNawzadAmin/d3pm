import numpy as np
import torch
from transformers import BertModel, BertTokenizer
import faiss
import scipy

def get_knn(k, embeds, mask_pairs):
    range_ = np.arange(len(embeds))
    indices = np.empty([len(embeds), k])
    similarities = np.empty([len(embeds), k])
    for mask, search_mask in mask_pairs:
        if mask.sum()>0:
            index = faiss.IndexFlatIP(embeds.shape[1]) 
            index.add(embeds[search_mask])
            similarities_temp, indices_temp = index.search(embeds[mask], k+1)
            similarities[mask] = similarities_temp[:, 1:]
            indices[mask] = range_[search_mask][indices_temp[:, 1:]]
    return indices, similarities

def get_L_and_K(forward_kwargs, gamma):
    if forward_kwargs['type'] == "bert_embed":
        embeds = BertModel.from_pretrained("bert-base-uncased").embeddings.word_embeddings.weight
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        vocab = np.array(list(tokenizer.get_vocab().keys()))
        embeds = embeds.detach().cpu().numpy()
        
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds_normalized = embeds / norms
        
        print("Constructing nearest neighbor matrix...")
        k = forward_kwargs['knn']
        is_unused = 0 * np.array([t.startswith("[") and t.endswith("]") for t in vocab])
        is_suffix = 0 * np.array([t.startswith("##") for t in vocab])
        is_number = np.array([any([x in np.arange(10).astype(str) for x in t]) for t in vocab])
        is_normal = ~np.any([is_unused, is_suffix, is_number], axis=0)
        indices, similarities = get_knn(k, embeds_normalized,
            [(is_number, is_number+is_normal), (is_normal, is_normal)])

        print("Constructing sparse matrix...")
        row_indices = np.repeat(np.arange(embeds.shape[0]), k) 
        col_indices = indices.flatten()
        dot_products = similarities.flatten()
        # rates = distances.sum(-1)
        assert (dot_products > 0).all()
        assert (row_indices != col_indices).all()
        if forward_kwargs['make_sym']:
            row_indices, col_indices = np.r_[row_indices, col_indices], np.r_[col_indices, row_indices]
            dot_products = np.r_[dot_products, dot_products]
        dot_products = np.exp((dot_products - 1) / forward_kwargs['bandwidth'])

        sparse_matrix = scipy.sparse.coo_array((dot_products, (row_indices, col_indices)),
                                  shape=(embeds.shape[0], embeds.shape[0]))
        sparse_matrix.sum_duplicates()
        L_off_diag = sparse_matrix.tocsr()
        L_diag = - L_off_diag.sum(-1)
        if forward_kwargs['normalize']:
            L_off_diag = L_off_diag / (-L_diag)[:, None]
            L_diag = -1 + 0 * L_diag
        rate = - (L_diag.min()) / (1-gamma) 
        L_cpu = L_off_diag / rate + scipy.sparse.diags(L_diag / rate)
        K_cpu = L_off_diag / rate + scipy.sparse.diags(L_diag / rate + 1)
        
        L_cpu = L_cpu.tocoo()
        K_cpu = K_cpu.tocoo()
        K = torch.sparse_coo_tensor((K_cpu.row, K_cpu.col), K_cpu.data,
                                    size=(embeds.shape[0], embeds.shape[0])).float()
        K = K.to_sparse_csr()
        L = torch.sparse_coo_tensor((L_cpu.row, L_cpu.col), L_cpu.data,
                                    size=(embeds.shape[0], embeds.shape[0])).float()
        L = L.to_sparse_csr()
        print("Done!")
        return L, K, L_cpu, K_cpu, rate
    if forward_kwargs['type'] == "uniform":
        sparse_matrix = scipy.sparse.coo_array(([0], ([0], [0])),
                                  shape=(embeds.shape[0], embeds.shape[0]))
        L_cpu = sparse_matrix.tocsr()
        K_cpu = L_cpu
        L_cpu = L_cpu.tocoo()
        K_cpu = K_cpu.tocoo()
        K = torch.sparse_coo_tensor((K_cpu.row, K_cpu.col), K_cpu.data,
                                    size=(embeds.shape[0], embeds.shape[0])).float()
        K = K.to_sparse_csr()
        L = torch.sparse_coo_tensor((L_cpu.row, L_cpu.col), L_cpu.data,
                                    size=(embeds.shape[0], embeds.shape[0])).float()
        L = L.to_sparse_csr()
        return L, K, L_cpu, K_cpu, 0
