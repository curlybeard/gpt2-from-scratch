import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

#use dataclass to skip initialization boilerplate
@dataclass
class GPTConfig:
    #max sequence length
    block_size: int = 128
    #number of tokens in vocab
    vocab_size: int = 50257
    #number of transformer blocks
    n_layer: int = 12
    #number of attention heads
    n_head: int = 12
    #embedding dimensions
    n_embed: int = 768

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        #porject data into 4x higher dimension (768 -> 3072)
        #gives model more workspace to process information
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        #non linearity to allow model to learn complex patterns
        #GPT 2 uses tanh
        self.gelu = nn.GELU(approximate='tanh')
        #project back down to original dimension
        #compression forces model to keep only most important information from its expanded processing
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        #verify that embedding dimension divides evenly by number of heads
        #multi head attention splits embedding dimension across heads
        #each head processes its own portion of the dimensions independently
        #ie. 768 / 12 = 64 dimensions per head
        assert config.n_embed % config.n_head == 0

        #GPT-2 uses a combined layer for query, key and value? more efficient apparently
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        #project result back to original dimension
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        #creates a 1024x1024 lower triangular matrix of ones (torch.tril)
        #tokens can only attend to previosu tokens, not previous ones
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        #B = batch size
        #T = sequence length
        #C = embedding dimension
        B, T, C = x.size()
        #[B, T, 3 * 768]
        qkv = self.c_attn(x)
        #separate into three separate tensors [B, T, 768]
        q, k, v = qkv.split(self.n_embed, dim=2)
        #split 678 dimensions into 12 heads of 64 dimensions each
        #[B, T, 12, 64] -> Transpose -> [B, 12, T, 64]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        #compute the attention score ie) how much it should attend to each key position
        #scaled down by 1/sqrt(64) to prevent values from getting too large
        att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
        #apply casual mask 'bias'
        #replace all 0s in mask with negative inf
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #soft max see negative inf and outputs 0 paying zero attention to this position
        att = F.softmax(att, dim=-1)
        #weighted average of value vectors based on attention probabilities
        y = att @ v
        #transpose to swap head and sequence dims back [B, 12, T, 64] -> [B, T, 12, 64]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        #project concatenated head output back to model dimension
        #allow heads to interact and be combined
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        #pre-norm architecture (apply layer norm before attention)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x):
        #normalize input, apply attention and add residual connection
        x = x + self.attn(self.ln_1(x))
        #normalize again, apply MLP and add another residual connection
        x = x + self.mlp(self.ln_2(x))
        return x

#transformer architecture based on https://arxiv.org/pdf/1706.03762
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            #convert each tokenID into a vector of size n_embed
            #gives each token in vocab a unique representation
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            #adds position information to each token
            wpe = nn.Embedding(config.block_size, config.n_embed),
            #list of n_layer transformer blocks
            #each block refines understand of text
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            #normalizes output for training stability
            ln_f = nn.LayerNorm(config.n_embed),
        ))
        #projects the final hdiden states back to vocab size to predict next token
        self.lm_head = self.transformer.wte.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        #create tensor of position indices from 0 to T-1
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        #add position emb and token embed together
        x = tok_emb + pos_emb

        #forward the transformer block and last normalization block to get logits
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = x @ self.lm_head.t()

        loss = None
        #reshape (B,T, vocab_size) into (B*T, vocab_size)
        #collapses all time steps and batche sinto one big matrix of predictions
        if targets is not None:
            #cross entropy compares prediction with true token, measuring loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss