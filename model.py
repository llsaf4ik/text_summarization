import torch.nn as nn
import torch
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.W_O = nn.Parameter(
            nn.init.xavier_normal_(
                torch.randn(d_model, d_model)
            )
        )
        
        self.W_Q = nn.Parameter(
            nn.init.xavier_normal_(
                torch.randn(d_model, d_model)
            )
        )
       
        self.W_K = nn.Parameter(
            nn.init.xavier_normal_(
                torch.randn(d_model, d_model)
            )
        )
    
        self.W_V = nn.Parameter(
            nn.init.xavier_normal_(
                torch.randn(d_model, d_model)
            )
        )

        self.dropout = nn.Dropout(dropout_p)
        self.kv_cache = None
        
    def forward(self, Q, K, V, mask=None, cache=False):
        """
        Q:     [Batch, Seq_len1, d_model]
        K, V:  [Batch, Seq_len2, d_model]
        mask:  [Batch, Seq_len1, Seq_len2]
        """
            
        Q_heads = Q @ self.W_Q    # [Batch, Seq_len1, d_model]
        K_heads = K @ self.W_K    # [Batch, Seq_len2, d_model]
        V_heads = V @ self.W_V    # [Batch, Seq_len2, d_model]

        if cache:
            if self.kv_cache is not None:
                K_cached = torch.concat([self.kv_cache[0], K_heads], dim=1)
                V_cached = torch.concat([self.kv_cache[1], V_heads], dim=1)
            else:
                K_cached = K_heads
                V_cached = V_heads
                
            self.kv_cache = (K_cached, V_cached)
            K_heads = K_cached
            V_heads = V_cached
        
        # [Batch, Seq_len, d_model] -> [Batch, Seq_len, h, d_k] -> [Batch, h, Seq_len, d_k]
        Q_heads = Q_heads.view(Q_heads.shape[0], Q_heads.shape[1], self.h, self.d_k).transpose(1, 2)
        K_heads = K_heads.view(K_heads.shape[0], K_heads.shape[1], self.h, self.d_k).transpose(1, 2)
        V_heads = V_heads.view(V_heads.shape[0], V_heads.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # [Batch, h, Seq_len1, Seq_len2]
        attention_maps = torch.matmul(Q_heads, K_heads.transpose(-1, -2)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)                               # [Batch, 1, Seq_len1, Seq_len2]
            attention_maps.masked_fill_(mask == 0, -1e9)
            
        attention_weights = torch.softmax(attention_maps, dim=-1)  # [Batch, h, Seq_len1, Seq_len2]
        attention_weights = self.dropout(attention_weights)
        
        scores = torch.matmul(attention_weights, V_heads)          # [Batch, h, Seq_len1, d_k]
        scores = scores.transpose(1, 2).contiguous()               # [Batch, Seq_len1, h, d_k]
        scores = scores.view(scores.shape[0],                      # [Batch, Seq_len1, d_model]
                             scores.shape[1], 
                             self.d_model)
        
        result = scores @ self.W_O                                 # [Batch, Seq_len1, d_model]  
        
        return result

    def clear_cache(self):
        self.kv_cache = None
    
    
class LayerNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        
    def forward(self, X, eps=1e-6):
        """
        X: [Batch, Seq_len, size]
        """
        mean = X.mean(dim=-1, keepdim=True)              # [Batch, Seq_len, 1]  [1] в конце благодаря keepdim=True
        var = X.var(dim=-1, correction=0, keepdim=True)  # [Batch, Seq_len, 1]  который сохраняет размерность
        
        return self.gamma * (X - mean) / torch.sqrt(var + eps) + self.bias
    

class FeedForwardLayers(nn.Module):
    def __init__(self, d_model, dropout_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.layer1 = nn.Linear(d_model, 4 * d_model)
        self.layer2 = nn.Linear(4 * d_model, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, X):
        """
        X: [Batch, Seq_len, d_model]
        """
        out = self.act(self.layer1(X))
        out = self.dropout(out)
        out = self.layer2(out)
        
        return out
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len).unsqueeze(1).expand(max_seq_len, d_model // 2)  # [max_seq_len, d_model // 2]
        i = torch.arange(0, d_model//2).unsqueeze(0).expand(max_seq_len, d_model//2)       # [max_seq_len, d_model // 2]
        arg = pos / 10000**(2 * i / d_model)
        pe[:, ::2] = torch.sin(arg)
        pe[:, 1::2] = torch.cos(arg)
        
        pe = pe.unsqueeze(0)            # [1, max_seq_len, d_model]
        self.register_buffer("pe", pe)
        
    def forward(self, X):
        """
        X: [Batch, Seq_len, d_model] or int
        """
        if isinstance(X, int):
            out = self.pe[:, X, :]
        else:
            out = X + self.pe[:, :X.shape[1], :]
        return out


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, X):
        out = self.embedding(X) * math.sqrt(self.d_model)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout_p=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, h, dropout_p)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForwardLayers(d_model, dropout_p)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, X, padding_mask=None):
        """
        X:            [Batch, Seq_len, d_model]
        padding_mask: [Batch, 1, Seq_len]
        """
        out = self.norm1(X)                                # [Batch, Seq_len, d_model]
        out = self.mha(out, out, out, mask=padding_mask)   # [Batch, Seq_len, d_model]
        out = self.dropout(out)
        out = out + X

        norm_out = self.norm2(out)                         # [Batch, Seq_len, d_model]
        ff_out = self.ff(norm_out)                         # [Batch, Seq_len, d_model]
        ff_out = self.dropout(ff_out)
        out = ff_out + out      
        
        return out
    

class Encoder(nn.Module):
    def __init__(self, d_model, h, num_layers, dropout_p=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, h, dropout_p) for _ in range(num_layers)
        )
        self.norm = LayerNorm(d_model)
        
    def forward(self, X, padding_mask=None):
        """
        X:            [Batch, Seq_len, d_model]
        padding_mask: [Batch, 1, Seq_len]
        """
        for layer in self.layers:
            X = layer(X, padding_mask)
            
        return self.norm(X)
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, dropout_p=0.1):
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, h, dropout_p)
        self.norm1 = LayerNorm(d_model)
        self.cross_mha = MultiHeadAttention(d_model, h, dropout_p)
        self.norm2 = LayerNorm(d_model)
        self.ff = FeedForwardLayers(d_model, dropout_p)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, X, X_enc, enc_mask=None, dec_mask=None, cache=False):
        """
        X:        [Batch, Seq_len1, d_model]
        X_enc:    [Batch, Seq_len2, d_model]
        enc_mask: [Batch, 1, Seq_len2]
        dec_mask: [Batch, Seq_len1, Seq_len1]
        """
        norm_out = self.norm1(X)                                 # [Batch, Seq_len1, d_model]
        out = self.masked_mha(norm_out,                          # [Batch, Seq_len1, d_model]
                              norm_out, 
                              norm_out, dec_mask, cache=cache)       
        out = self.dropout(out)
        out = out + X

        norm_out = self.norm2(out)                               # [Batch, Seq_len1, d_model]
        cross_mha_out = self.cross_mha(norm_out,                 # [Batch, Seq_len1, d_model]
                                       X_enc, 
                                       X_enc, enc_mask, cache=False)    
        cross_mha_out = self.dropout(cross_mha_out)
        out = cross_mha_out + out

        norm_out = self.norm3(out)                               # [Batch, Seq_len1, d_model]
        ff_out = self.ff(norm_out)                               # [Batch, Seq_len1, d_model]
        ff_out = self.dropout(ff_out)
        out = ff_out + out
        
        return out

    def clear_cache(self):
        self.masked_mha.clear_cache()
    

class Decoder(nn.Module):
    def __init__(self, d_model, h, num_layers, dropout_p=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, h, dropout_p) for _ in range(num_layers)
        )
        self.norm = LayerNorm(d_model)
        
    def forward(self, X, X_enc, enc_mask=None, dec_mask=None, cache=False):
        """
        X:        [Batch, Seq_len1, d_model]
        X_enc:    [Batch, Seq_len2, d_model]
        enc_mask: [Batch, 1, Seq_len2]
        dec_mask: [Batch, Seq_len1, Seq_len1]
        """
        for layer in self.layers:
            X = layer(X, X_enc, enc_mask, dec_mask, cache=cache)
            
        return self.norm(X)

    def clear_cache(self):
        for layer in self.layers: 
            layer.clear_cache()
            
    
class Transformer(nn.Module):
    def __init__(self, d_model, h, enc_num_layers, dec_num_layers, vocab_size, max_seq_len, dropout_p=0.1):
        super().__init__()
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.encoder = Encoder(d_model, h, enc_num_layers, dropout_p)
        self.decoder = Decoder(d_model, h, dec_num_layers, dropout_p)
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = Embedding(vocab_size, d_model)
        nn.init.xavier_normal_(self.embedding.embedding.weight)
        self.projection_layer = nn.Linear(d_model, vocab_size)
        self.projection_layer.weight = self.embedding.embedding.weight
        
    def forward(self, X_enc, X_dec, enc_mask, dec_mask):
        """
        X_enc:      [Batch, Seq_len1]
        X_dec:      [Batch, Seq_len2]
        enc_mask:   [Batch, 1, Seq_len1]
        dec_mask:   [Batch, Seq_len2, Seq_len2]
        """
        X_enc = self.embedding(X_enc)                       # [Batch, Seq_len1, d_model]
        X_enc = self.pe(X_enc)                              # [Batch, Seq_len1, d_model]
        X_enc = self.dropout(X_enc)
        
        X_dec = self.embedding(X_dec)                       # [Batch, Seq_len2, d_model]
        X_dec = self.pe(X_dec)                              # [Batch, Seq_len2, d_model]
        X_dec = self.dropout(X_dec)
        
        out = self.encoder(X_enc, enc_mask)                 # [Batch, Seq_len1, d_model]
        out = self.decoder(X_dec, out, enc_mask, dec_mask)  # [Batch, Seq_len2, d_model]
        out = self.projection_layer(out)                    # [Batch, Seq_len2, vocab_size]
        
        return out

    def generate(self, X_enc, sos_id=1, eos_id=2, max_len=100, repetition_penalty=1.3, penalty_window=10):
        """
        X_enc: [Batch, Seq_len]
        """
        dec_id = torch.tensor([[sos_id]]).long().to(X_enc.device)
        
        X_enc_emb = self.embedding(X_enc)
        X_enc_emb = self.pe(X_enc_emb)
        enc_out = self.encoder(X_enc_emb)

        result = []
        
        i = 0
        while i < max_len:
            X_dec = self.embedding(dec_id)
            X_dec = X_dec + self.pe(i)
            out = self.decoder(X_dec, enc_out, cache=True) 
            logits = self.projection_layer(out)[0, -1, :] 
            
            # Определяем начало окна
            start_index = 0
            if penalty_window is not None and len(result) > penalty_window:
                start_index = len(result) - penalty_window
            
            # Берем только последние N токенов для проверки
            tokens_to_penalize = set(result[start_index:])
            
            for token in tokens_to_penalize:
                if logits[token] > 0:
                    logits[token] /= repetition_penalty
                else:
                    logits[token] *= repetition_penalty

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs).item()
            
            result.append(next_token)
            
            if next_token == eos_id:
                break
                
            dec_id = torch.tensor([[next_token]]).long().to(X_enc.device)
            i += 1

        self.decoder.clear_cache()
        return result