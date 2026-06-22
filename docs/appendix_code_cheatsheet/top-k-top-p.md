# C.6 Top-k / Top-p Sampling  Temperature

 LLM ， RL （RLHF 、temperature ）。

 logits ，：

- **Temperature**： logits，。
- **Top-k**： token，。
- **Top-p**：，。

---

## Temperature

****：softmax """"，。

****：

- `logits`：
- `temperature`（$T$，$T>0$）：softmax ****
- `scaled_logits = logits / T`： softmax 

### 

> **softmax  T。T ，T 。**

### 

```
scaled_logits = logits / T        # softmax 
probs = softmax(scaled_logits)
sample from probs
```

### 

$$
p_i = \frac{\exp(x_i / T)}{\sum_j \exp(x_j / T)}
$$

- $T \to 0$： argmax（）
- $T = 1$：
- $T \to \infty$：

### PyTorch 

```python
def sample_with_temperature(logits, temperature=1.0):
    if temperature < 1e-8:
        return logits.argmax(dim=-1)  # T=0 
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## Top-k Sampling

****： token ，。

****：

- `k`： token （ 50）
- `threshold`： k  logit， `-inf`

### 

> ** k  logit， $-\infty$，softmax 。**

### 

```
threshold =  k  logit
logits[logits < threshold] = -inf
probs = softmax(logits)            # -inf 
sample from probs
```

### Python 

```python
import numpy as np

def top_k_filtering(logits, k):
    """logits: [vocab_size] ->  top-k  -inf"""
    if k >= len(logits):
        return logits
    threshold = np.sort(logits)[-k]  #  k （ k ）
    return np.where(logits >= threshold, logits, -np.inf)
```

### PyTorch 

```python
import torch

def top_k_filtering(logits, k):
    """logits: [B, vocab_size]  [vocab_size]"""
    if k <= 0:
        return logits
    top_k = min(k, logits.size(-1))
    threshold = torch.topk(logits, top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float('-inf'))

def top_k_sample(logits, k, temperature=1.0):
    logits = top_k_filtering(logits / temperature, k)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## Top-p (Nucleus) Sampling

****：Top-k  k ，—— 3 ， 50 。Top-p ** $\geq p$  token **（""），。

****：

- `p`：（ 0.9）
- `sorted_logits` / `sorted_indices`： logits 
- `cumulative_probs`：
- `nucleus_mask`：（ prob） p 

### 

> **， p ， $-\infty$。**

### 

```
sorted_logits, idx = sort_desc(logits)
sorted_probs = softmax(sorted_logits)
cumsum = cumsum(sorted_probs)
mask = cumsum - sorted_probs > p     #  prob 
sorted_logits[mask] = -inf
logits = scatter_back(sorted_logits, idx)  # 
probs = softmax(logits); sample
```

### 

|          | Top-k         | Top-p                         |
| -------- | ------------- | ----------------------------- |
|  |  k  |  p  |
|    |   |             |
|  | k=1 →     | p=0 → ，p=1 →       |

### Python 

```python
import numpy as np

def top_p_filtering(logits, p):
    """logits: [vocab_size] ->  -inf"""
    sorted_indices = np.argsort(logits)[::-1]           # 
    sorted_logits = logits[sorted_indices]
    sorted_probs = np.exp(sorted_logits - sorted_logits.max())
    sorted_probs /= sorted_probs.sum()
    cumulative_probs = np.cumsum(sorted_probs)

    # ： prob  p（ token）
    cutoff = cumulative_probs - sorted_probs > p
    sorted_logits[cutoff] = -np.inf

    result = np.full_like(logits, -np.inf)
    result[sorted_indices] = sorted_logits              # 
    return result
```

### PyTorch 

```python
import torch

def top_p_filtering(logits, p):
    """logits: [B, vocab_size] ->  -inf"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = (cumulative_probs - sorted_probs) > p  # 
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))

    return logits.scatter(1, sorted_indices, sorted_logits)  # 

def top_p_sample(logits, p, temperature=1.0):
    logits = top_p_filtering(logits / temperature, p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## 

 Temperature → Top-k → Top-p ：

```python
def generate_sample(logits, temperature=1.0, top_k=50, top_p=0.9):
    logits = logits / max(temperature, 1e-8)   # 1. Temperature（softmax ）
    logits = top_k_filtering(logits, top_k)    # 2. Top-k
    logits = top_p_filtering(logits, top_p)    # 3. Top-p
    probs = torch.softmax(logits, dim=-1)      # 4. 
    return torch.multinomial(probs, num_samples=1)
```

---

## 

|                  |                                                                                    |
| -------------------- | -------------------------------------------------------------------------------------- |
| Temperature      |  softmax **** T，                                              |
| Top-p  cumsum  | **** cumsum，                                                |
| Top-p    |  `cumsum - current_prob > p`  `cumsum > p`，（）token  |
| Top-k            |  `topk().values[..., -1]`  k ， sort                         |
| Top-p        |  `scatter` ，                                              |
|      |  `-inf`  softmax， token  1                          |
| `temperature=0`      |  argmax， 0                                                      |
