
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self,Q,K,V,mask=None):
        scores= torch.matmul(Q,K.transpose(-2,-1))/sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention= F.softmax(scores, dim=-1)
        # After multiplying by V, each token gets a vector containing context-aware information
        output= torch.matmul(attention,V)
        return output


class MultiHeadAttention(nn.Module):
    # d_model: the dimension of the model, num_heads: the number of heads
    def __init__(self, d_model, num_heads):
          super().__init__()
          self.num_heads = num_heads
          self.d_k = d_model // num_heads
          self.W_Q = nn.Linear(d_model, d_model)
          self.W_K = nn.Linear(d_model, d_model)
          self.W_V = nn.Linear(d_model, d_model)
          self.W_O = nn.Linear(d_model, d_model)
          self.attention = Attention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        Q=self.W_Q(Q)
        K=self.W_K(K)
        V=self.W_V(V)
        batch = Q.shape[0]
        # Split into num_heads
        # → (batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
        # Q, K, V may have different sequence lengths: Q from decoder, K from encoder
        Q=Q.view(batch,-1,self.num_heads,self.d_k).transpose(1,2)
        K=K.view(batch,-1,self.num_heads,self.d_k).transpose(1,2)
        V=V.view(batch,-1,self.num_heads,self.d_k).transpose(1,2)
        # → (batch, n_heads, seq_len, d_k)
        attention=self.attention(Q,K,V,mask)
        # Concatenate heads: merge last two dims back into d_model
        attention=attention.transpose(1,2).contiguous().view(batch,-1,self.num_heads*self.d_k)
        # Output projection: combine information learned by all heads
        output=self.W_O(attention)
        return output
        



class FNN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

#   input: (batch, seq_len, d_model)
    def forward(self, x):
        output1= F.relu(self.linear1(x))
        output2= self.linear2(output1)
        return output2
    


class positionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # max_len: maximum sequence length, d_model: embedding dimension per token

        pe = torch.zeros(max_len, d_model)
        # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)        # (max_len, 1)
        div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position / div_term)  # odd dimensions
        # (1, max_len, d_model) - unsqueeze for batch broadcasting
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    # input: (batch, seq_len, d_model)
    def forward(self, x):
        # Add positional encoding up to the sequence length
        output1=x+ self.pe[:,:x.size(1), :]
        return output1



class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FNN(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        

    def forward(self, x, mask=None):
        # x: batch, seq_len, d_model
        attention_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attention_output)
        ffn_output = self.ffn(x)
        output = self.layernorm2(x + ffn_output)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        # Masked self-attention: prevents decoder from attending to future tokens
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        # Cross-attention: attends to encoder output to incorporate source information
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn= FNN(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention on decoder input
        x = self.layernorm1(x + self.mha1(x, x, x, tgt_mask))
        # Cross-attention with encoder output
        x = self.layernorm2(x + self.mha2(x, enc_output, enc_output, src_mask))
        output = self.layernorm3(x + self.ffn(x))
        return output


class Encoder(nn.Module):
    # d_model: model dimension, num_heads: number of attention heads, d_ff: feed-forward dimension, num_layers: number of encoder layers
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.positional_encoding = positionalEncoding(d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        x = self.token_embedding(x)  # batch, seq_len, d_model
        x = self.positional_encoding(x)  # batch, seq_len, d_model
        for layer in self.layers:
            x = layer(x, mask)
        return x  # batch, seq_len, d_model


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.positional_encoding = positionalEncoding(d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.token_embedding(x)  # batch, seq_len, d_model
        x = self.positional_encoding(x)  # batch, seq_len, d_model
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x  # batch, seq_len, d_model



class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, src_vocab_size)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, tgt_vocab_size)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

    # src: source input sequence, tgt_input: shifted target for autoregressive decoding
    def forward(self, src, tgt_input, src_mask=None, tgt_mask=None):
        # src, tgt: (batch, seq_len), masks: (batch, 1, seq_len, seq_len)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt_input, enc_output, src_mask, tgt_mask)
        output = self.output_linear(dec_output)  # batch, seq_len, vocab_size
        return output
    
