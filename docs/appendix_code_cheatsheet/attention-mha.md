# C.7 Self-Attention、MHA、MQA  GQA

 RL ，****，RL 。

---

## Scaled Dot-Product Attention

****：。

****：

- `Q`（query）：""， $[seq_q, d_k]$
- `K`（key）：""， $[seq_k, d_k]$
- `V`（value）：""， $[seq_k, d_v]$
- `d_k`：query/key ， $\sqrt{d_k}$  softmax 
- `mask`：causal mask "" $-\infty$，softmax 

### 

> **Q  K ，、、softmax、 V。**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

 $QK^T$  $[seq_q, d_k]\times[d_k, seq_k]=[seq_q, seq_k]$，softmax （key ）。

### 

```
scores = Q @ K^T / sqrt(d_k)      #  + ，[seq_q, seq_k]
if mask: scores = scores + mask    #  -inf
attn = softmax(scores, dim=-1)     #  key 
output = attn @ V                  # ，[seq_q, d_v]
```

### Python 

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [seq_q, d_k], K: [seq_k, d_k], V: [seq_k, d_v]
    mask: [seq_q, seq_k], 0=, -inf=
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)        # [seq_q, seq_k]

    if mask is not None:
        scores = scores + mask

    #  softmax，（key ）
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    return attn_weights @ V                # [seq_q, d_v]
```

### PyTorch 

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q/K/V: [B, heads, seq, d_k]
    mask: [B, 1, seq_q, seq_k], 1=, 0=
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


def causal_mask(seq_len):
    """ = 1（）， = 0（）。"""
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
```

---

## Multi-Head Attention (MHA)

****： attention 。MHA  $d_{model}$  $h$ ， attention，。

****：

- `d_model`：
- `n_heads` / $h$：， $d_k = d_{model}/h$
- `W_Q` / `W_K` / `W_V` / `W_O`：

### 

> ** h ， attention，concat  $W_O$。**

### 

```
Q, K, V = x @ W_Q, x @ W_K, x @ W_V      # [B, seq, d_model]
Q, K, V = split_heads(Q, K, V)           # [B, h, seq, d_k]
attn = scaled_dot_product_attention(Q, K, V, mask)
attn = merge_heads(attn)                  # [B, seq, d_model]
output = attn @ W_O
```

：**view  → transpose  → attention → transpose  → view **。

### PyTorch 

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, seq_len, d_model = x.shape

        #  + : [B, seq, d_model] -> [B, n_heads, seq, d_k]
        Q = self.W_Q(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_out = scaled_dot_product_attention(Q, K, V, mask)

        #  + 
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, d_model)
        return self.W_O(attn_out)
```

---

## MQA  GQA

****：MHA  K/V  Q ， KV cache 。MQA  Q  K/V，；GQA ——Q 、 K/V，。

****：

- `n_heads`：Q 
- `n_kv_heads`：K/V （MQA=1，GQA=$g$，$1<g<h$）
- `n_groups`：$= n_{heads}/n_{kv\_heads}$， K/V  Q 

### 

|  | Q  | K/V         | KV cache |            |
| ---- | ------ | --------------- | -------- | ------------------ |
| MHA  | $h$    | $h$             |      | GPT-2、BERT        |
| MQA  | $h$    | **1**           |      | PaLM、StarCoder    |
| GQA  | $h$    | **$g$** ($g<h$) |      | LLaMA 2/3、Mistral |

### 

> **MQA： Q  K/V（）；GQA： $g$ （）；MQA  $g=1$  GQA。**

### PyTorch （GQA）

```python
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        """
        n_heads: Q （ 32）
        n_kv_heads: K/V （ 8）， n_heads
        """
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, n_heads * self.d_k)
        self.W_K = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_O = nn.Linear(n_heads * self.d_k, d_model)

    def forward(self, x, mask=None):
        B, seq_len, _ = x.shape

        Q = self.W_Q(x).view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        #  head : [B, n_kv_heads, seq, d_k] -> [B, n_heads, seq, d_k]
        #  n_groups  Q  K/V
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, seq_len, -1)
        return self.W_O(attn_out)
```

---

## ：

|              |              |                    |
| -------------- | ------------------ | ---------------------- |
| Self-Attention | $O(n^2 \cdot d)$   | $n$ ，$d$  |
|        | $O(n \cdot d^2)$   |  token     |
| （MHA）    | $O(n^2 d + n d^2)$ |  $n^2$   |

---

## 

|                                     |                                                          |
| --------------------------------------- | ------------------------------------------------------------ |
|  $\sqrt{d_k}$  $\sqrt{d_{model}}$ |  $d_k = d_{model}/h$                           |
| softmax                       |  key ，（ query） 1                |
| causal mask                         | `tril`  = ， =                         |
| view  contiguous                      | `transpose` ，`.contiguous()`  `view`          |
| GQA  `repeat_interleave`              |  `repeat`； $n_{groups}$  Q  K/V |
| MQA  GQA                        | $n_{kv\_heads}=1$  GQA  MQA                          |
