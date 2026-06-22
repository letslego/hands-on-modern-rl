# C.5 Softmax  Cross-Entropy

。 DPO / PPO ， softmax 。

---

##  Softmax

****： logits （ 1）， $\exp(1000)=\text{inf}$ 。

****：

- `x` / `logits`：
- `m = max(x)`：，
- `axis`：，LLM （）

### 

> **， exp、、。**

### 

```
m = max(x)
exp_x = exp(x - m)
softmax = exp_x / sum(exp_x)
```

### Python 

```python
import numpy as np

def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
```

### PyTorch 

```python
import torch
import torch.nn.functional as F

# 
probs = F.softmax(logits, dim=-1)

# （）
def manual_softmax(x, dim=-1):
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x_shifted)
    return e_x / e_x.sum(dim=dim, keepdim=True)
```

---

## Log-Sum-Exp  Log-Softmax

****：LLM  log ，。 softmax  log ， 0  log  `-inf`。log-sum-exp  max  log ， log 。

****：

- `m = max(x)`： softmax，
- `lse = m + log(sum(exp(x - m)))`：logistic normalizer 
- ：`log_softmax(x)_i = x_i - m - log(sum(exp(x - m)))`

：

$$
\log\sum_j \exp(x_j) = m + \log\sum_j \exp(x_j - m), \quad m = \max(x)
$$

### 

> ** softmax  log——$\log\text{softmax}_i = x_i - \text{LSE}(x)$，LSE  max 。**

### Python 

```python
def log_softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    return x_shifted - np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))
```

### PyTorch 

```python
# ，
log_probs = F.log_softmax(logits, dim=-1)

# 
def manual_log_softmax(x, dim=-1):
    max_val = x.max(dim=dim, keepdim=True).values
    return x - max_val - torch.log(torch.sum(torch.exp(x - max_val), dim=dim, keepdim=True))
```

---

## Cross-Entropy Loss

****： / SFT """"。Cross-Entropy " log "——，loss 。

****：

- `logits`：， `[N, C]`，N ，C 
- `targets`：， `[N]`
- `ignore_index`：（ padding / prompt）， `-100`
- `log_probs`：log_softmax ，

 $p$  one-hot（ label  1），：

$$
H(p, q) = -\sum_i p_i \log q_i \;=\; -\log q_{\text{label}}
$$

### 

> **`-log_softmax(logits)[target].mean()`——。**

### 

```
log_probs = log_softmax(logits)
loss = -log_probs[target].mean()
```

### Python 

```python
def cross_entropy(logits, targets, ignore_index=-100):
    """
    logits:  [N, C]
    targets: [N] 
    """
    log_probs = log_softmax(logits, axis=-1)
    total, count = 0.0, 0
    for i in range(len(targets)):
        if targets[i] == ignore_index:
            continue
        total += -log_probs[i, targets[i]]
        count += 1
    return total / max(count, 1)
```

### PyTorch 

```python
def manual_cross_entropy(logits, targets, ignore_index=-100):
    """
    logits:  [B, C]
    targets: [B]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # gather  target  log 
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    # mask  ignore_index
    mask = targets != ignore_index
    return -target_log_probs[mask].mean()
```

---

## 

|                    |                                                                  |
| ---------------------- | -------------------------------------------------------------------- |
| softmax  max     |                                                  |
|  softmax  log      | ， `log_softmax`                                 |
| Cross-Entropy  |  softmax  log  CE， `F.cross_entropy(logits, targets)` |
| `ignore_index`         |  SFT loss ，padding token                      |
| temperature            | `logits / temperature`  softmax，T                     |
