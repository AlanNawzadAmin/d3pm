import numpy as np
from sequence_models.layers import PositionFeedForward
from sequence_models.convolutional import ByteNetBlock
from torch import nn
import torch.nn.functional as F
from .dit_text import modulate_fused, TimestepEmbedder

class ByteNetLMTime(nn.Module):
    """Stacked residual blocks from ByteNet paper defined by n_layers

         Shape:
            Input: (N, L,)
            input_mask: (N, L, 1), optional
            Output: (N, L, d)
    """

    def __init__(self, n_tokens=31, d_embedding=128, d_model=1024, n_layer=16,
                 kernel_size=5, r=128, rank=None, n_frozen_embs=None,
                 padding_idx=None, causal=False, dropout=0.1, slim=True, activation='gelu',
                 schedule_conditioning=True):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :padding_idx: location of padding token in ordered alphabet
        :param causal: if True, chooses MaskedCausalConv1d() over MaskedConv1d()
        :param rank: rank of compressed weight matrices
        :param n_frozen_embs: number of frozen embeddings
        :param slim: if True, use half as many dimensions in the NLP as in the CNN
        :param activation: 'relu' or 'gelu'
        :param down_embed: if True, have lower dimension for initial embedding than in CNN layers
        """
        super().__init__()
        self.time_encoding = TimestepEmbedder(d_embedding) # Timestep encoding
        self.time_mod_layer = nn.Linear(d_embedding, d_model)
        if schedule_conditioning:
            self.s_embed_input = TimestepEmbedder(d_model)
            self.s_embed_block = TimestepEmbedder(d_embedding)
        self.embedder = nn.Embedding(n_tokens, d_model, padding_idx=padding_idx)
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layer)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        self.layers = nn.ModuleList([
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, causal=causal, rank=rank,
                         activation=activation)
            for d in dilations
        ])
        self.c_mod_layers = nn.ModuleList([nn.Linear(d_embedding, 2*d_model) for d in dilations])
        self.dropout = dropout
        self.decoder = PositionFeedForward(d_model, n_tokens)
        self.last_norm = nn.LayerNorm(d_model)

    def forward(self, x, t, S=None, input_mask=None):
        """
        :param x: (batch, length)
        :param y: (batch)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        x = self.embedder(x)
        c = F.silu(self.time_encoding(t))[:, None, :]

        x = x + self.time_mod_layer(c)
        if S is not None:
            bs, seq_len = S.shape[0], S.shape[1]
            S_out = F.silu(self.s_embed_input(S.reshape(-1))).reshape(bs, seq_len, -1)
            x = x + S_out
            
            # WIP, this is approximately correct but not thoroughly tested
            S_out = F.silu(self.s_embed_block(S.reshape(-1))).reshape(bs, seq_len, -1)
            c = c + S_out

        for layer, c_layer in zip(self.layers, self.c_mod_layers):
            x = layer(x, input_mask=input_mask)
            c_mod = c_layer(c)
            modulate_fused(x, *c_mod.chunk(2, dim=-1))
            if self.dropout > 0.0:
                x = F.dropout(x, self.dropout)
        return self.decoder(self.last_norm(x))
