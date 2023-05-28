import numpy as np
import torch.nn.functional as F
import math, copy, re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import torchtext
from typing import Optional
import torch
from torch import nn
from torch.nn import Linear, ReLU, GELU

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)
       
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self, key, query, value, mask=None):

        batch_size = key.size(0)
        seq_length = key.size(1)
        
        seq_length_query = query.size(1)
        
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
       
        k = self.key_matrix(key)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        k_adjusted = k.transpose(-1,-2) 
        product = torch.matmul(q, k_adjusted)  
      
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))
  
        product = product / math.sqrt(self.single_head_dim)

        scores = F.softmax(product, dim=-1)
      
        scores = torch.matmul(scores, v) 
      
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads) 
        
        output = self.out(concat)
       
        return output

class MLP(torch.nn.Module):
    
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.linear1 = Linear(n_in, n_hidden)
        self.linear2 = Linear(n_hidden, n_out)
        
    def forward(self, x):
        return self.linear2(ReLU()(self.linear1(x)))

class HyperMixerMLP(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out=None, mlps=1):
        super().__init__()
        if n_out is None:
            n_out = n_in

        self.original_in_size = n_in
        self.original_out_size = n_out

        assert n_in % mlps == 0
        assert n_out % mlps == 0
        assert n_hidden % mlps == 0
        n_in = n_in // mlps
        n_out = n_out // mlps
        n_hidden = n_hidden // mlps

        self.input_size = n_in
        self.output_size = n_out

        self.num_mlps = mlps

        self.fc1_weights = torch.nn.Parameter(torch.empty(mlps, n_hidden, n_in))
        self.fc1_biases = torch.nn.Parameter(torch.empty(mlps, n_hidden))
        self.fc2_weights = torch.nn.Parameter(torch.empty(mlps, n_out, n_hidden))
        self.fc2_biases = torch.nn.Parameter(torch.empty(mlps, n_out))

        torch.nn.init.xavier_uniform_(self.fc1_weights, gain=math.sqrt(2.0))
        torch.nn.init.xavier_uniform_(self.fc1_biases, gain=math.sqrt(2.0))
        torch.nn.init.xavier_uniform_(self.fc2_weights, gain=math.sqrt(2.0))
        torch.nn.init.xavier_uniform_(self.fc2_biases, gain=math.sqrt(2.0))

        self.activation = torch.nn.GELU()
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = x.reshape((batch_size, seq_len, self.num_mlps, self.output_size))
        x = torch.einsum("blmf,mhf->bmlh", x, self.fc1_weights) + self.fc1_biases.unsqueeze(0).unsqueeze(2)
        x = self.activation(x)
        x = torch.einsum("bmlh,mfh->bmlf", x, self.fc2_weights) + self.fc2_biases.unsqueeze(0).unsqueeze(2)

        return x
    
class HyperNetwork(torch.nn.Module):
    
    def __init__(self, d_in, d_hidden, n_in, n_hidden, mlps=1, tied=False):
        super().__init__()
        self.tied = tied
        self.w1_gen = HyperMixerMLP(n_in, n_hidden, n_in, mlps)
        if self.tied:
            self.w2_gen = self.w1_gen
        else:
            self.w2_gen = HyperMixerMLP(d_in, d_hidden, d_in, mlps)
        
    def forward(self, x):
        W1 = self.w1_gen(x.transpose(1, 2))
        if self.tied:
            W2 = W1
        else:
            W2 = self.w2_gen(x)
        return W1, W2

class HyperMixer(torch.nn.Module):
    def __init__(self, d_in, d_hidden, n_in, n_hidden, mlps=1, tied=False):
        super().__init__()
        self.input_dim = n_in
        self.output_dim = d_in
        self.hyper = HyperNetwork(d_in, d_hidden, n_in, n_hidden, mlps, tied)
        self.activation = torch.nn.GELU()
        self.layer_norm = torch.nn.LayerNorm(d_in)
        self.num_heads = mlps

    def forward(self, x):
        out = x
        batch_size = out.size(0)
        seq_len = out.size(1)
        hyp_input = out
        W1, W2 = self.hyper(hyp_input)

        out = out.transpose(1, 2)
        out = out.reshape((batch_size * self.num_heads, 
                           self.output_dim // self.num_heads, seq_len)) 
        W1 = W1.reshape((batch_size * self.num_heads, seq_len, -1))
        W2 = W2.reshape((batch_size * self.num_heads, seq_len, -1))

        out = self.activation(torch.bmm(out, W1))
        out = torch.bmm(out, W2.transpose(1, 2))

        out = out.reshape((batch_size, self.output_dim, seq_len))

        out = self.layer_norm(out.transpose(1, 2))

        return out

class MLP(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(MLP, self).__init__()
        self.linear1 = Linear(n_in, n_hidden)
        self.linear2 = Linear(n_hidden, n_out)
        
    def forward(self, X):
        X = F.relu(self.linear1(X))
        X = self.linear2(X)
        return X
    
class Encoder(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(Encoder, self).__init__()
        self.mlp = MLP(n_in, n_hidden, n_out)
        
    def forward(self, X):
        X = self.mlp(X)
        return X

class DeepSets(torch.nn.Module):
    """
      DeepSets model
    """
    def __init__(self, n_in, n_hidden_enc, n_out_enc, n_hidden_dec=16, n_out_dec=2):
        super(DeepSets, self).__init__()
        self.encoder = Encoder(n_in, n_hidden_enc, n_out_enc)
        self.decoder = MLP(n_out_enc, n_hidden_dec, n_out_dec)
        
    def forward(self, X):
        z_enc = self.encoder(X).mean(dim=1)
        z = self.decoder(z_enc)
        return z

class LightHyperClassifier(torch.nn.Module):
    """
      DeepSets with a HyperMixer block as token mixing layer
    """
    def __init__(self, n_in, n_hidden_enc, n_out_enc, 
                 n_hidden_dec=16, n_out_dec=2, input_size=1024):
        super(LightHyperClassifier, self).__init__()
        self.encoder = Encoder(n_in, n_hidden_enc, n_out_enc)
        self.token_mixing = HyperMixer(n_out_enc, n_hidden_enc, input_size, 
                                       n_hidden_enc, mlps=4)
        self.decoder = MLP(n_out_enc, n_hidden_dec, n_out_dec)
        
    def forward(self, X):
        z_enc = self.encoder(X)
        z_mixed = self.token_mixing(z_enc)
        z = self.decoder(z_enc)
        return z.mean(dim=1)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, max_length=512, n_heads=8,
                 use_hypernetwork=True):
        super(TransformerBlock, self).__init__()

        if use_hypernetwork is True:
            self.attention = HyperMixer(embed_dim, 2 * embed_dim, 
                                        max_length, 2 * max_length, n_heads)
        else:
            self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.using_hypernetwork = False
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.ReLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        
        if (self.using_hypernetwork):
            attention_out = self.attention(x)[0]
        else:
            attention_out = self.attention(x, x, x)[0]
        attention_residual_out = attention_out + x
        norm1_out = self.dropout1(self.norm1(attention_residual_out))

        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

        return norm2_out

class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, embed_dim, num_layers=2, 
                 expansion_factor=4, n_heads=8, hypernetwork=True):
        super(TransformerEncoder, self).__init__()
        
        vocab_size = 2 * n_heads
        self.encoder = MLP(embed_dim, 2 * embed_dim, vocab_size)
        self.layers = nn.ModuleList([TransformerBlock(vocab_size, 
                                                      expansion_factor, 
                                                      seq_len, 
                                                      n_heads, 
                                                      hypernetwork) for i in range(num_layers)])
    
    def forward(self, x):
        out = self.encoder(x)
        for layer in self.layers:
            out = layer(out)

        return out

class HeavyHyperClassifier(nn.Module):
    """
      Transformer's encoder with a HyperMixer as an attention layer with some
      additional mappings at the output
    """
    def __init__(self, seq_len, embed_dim, num_classes, 
                 num_layers=2, expansion_factor=4, n_heads=8):
        super(HeavyHyperClassifier, self).__init__()
        vocab_size = 2 * n_heads
        self.encoder = TransformerEncoder(seq_len, embed_dim, 
                                          num_layers, expansion_factor, n_heads)
        self.first_layer = nn.Sequential(
                           nn.Dropout(0.2),
                           nn.Linear(vocab_size, 5 * num_classes),
                           nn.ReLU()
        )

        self.output_layer = nn.Sequential(
                            nn.Dropout(0.2),
                            nn.Linear(5 * num_classes, num_classes)
        )
    
    def forward(self, x): 
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.first_layer(x)
        return self.output_layer(x)

class HeavyTransformerClassifier(nn.Module):
    """
      Transformer's encoder with classic attention layer with some
      additional mappings at the output
    """
    def __init__(self, seq_len, embed_dim, num_classes, 
                 num_layers=2, expansion_factor=4, n_heads=8):
        super(HeavyTransformerClassifier, self).__init__()
        vocab_size = 2 * n_heads
        self.encoder = TransformerEncoder(seq_len, embed_dim, 
                                          num_layers, expansion_factor, n_heads, hypernetwork=True)
        self.first_layer = nn.Sequential(
                           nn.Dropout(0.2),
                           nn.Linear(vocab_size, 5 * num_classes),
                           nn.ReLU()
        )

        self.output_layer = nn.Sequential(
                            nn.Dropout(0.2),
                            nn.Linear(5 * num_classes, num_classes)
        )
    
    def forward(self, x): 
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.first_layer(x)
        return self.output_layer(x)