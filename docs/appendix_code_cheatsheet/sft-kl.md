# C.1 SFT Loss  KL 

## SFT Loss（）

****： token， loss。

****：

- `logits`：， `[B, seq_len, vocab_size]`， $t$  $t+1$
- `labels`： token ，prompt  `ignore_index=-100`
- `ignore_index`：（ `-100`）

### 

> **logits 、labels ： $t$  $t+1$；prompt  `-100`， loss。**

### 

```
logits = model(input_ids)                #  t  t+1
shift_logits = logits[:, :-1, :]         # ：""
shift_labels = labels[:, 1:]             # ：
loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

 $t$  $t+1$， logits  $t$  labels  $t+1$ 。

### Python 

```python
import numpy as np

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)  #  max，
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: [seq_len, vocab_size]
    labels: [seq_len]  ( shift)
    """
    shift_logits = logits[:-1]
    shift_labels = labels[1:]

    probs = softmax(shift_logits, axis=-1)
    total, count = 0.0, 0
    for t in range(len(shift_labels)):
        if shift_labels[t] == ignore_index:
            continue
        total += -np.log(probs[t, shift_labels[t]] + 1e-12)
        count += 1
    return total / max(count, 1)
```

### PyTorch 

```python
import torch
import torch.nn.functional as F

def sft_loss(logits, labels, ignore_index=-100):
    """
    logits: [B, seq_len, vocab_size]
    labels: [B, seq_len]
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
```

---

## KL 

****： $p$  $q$ ， PPO / GRPO  KL 。

****：

- `log_probs`： $p$  token  log 
- `ref_log_probs`： $q$（ SFT ） token  log 
- `log_ratio`：$\log(q/p)$，k3 

### 

> **k1：`mean(log_p − log_q)`，；k3：`mean(exp(Δ) − 1 − Δ)`，$\Delta=\log\frac{q}{p}$，。**

### 

```
# k1（PPO ）：，，
kl = (log_probs - ref_log_probs).mean()

# k3（GRPO / trl ）：，ratio  q/p
log_ratio = ref_log_probs - log_probs        # log(q/p)
kl = (exp(log_ratio) - 1 - log_ratio).mean()
```

### Python 

```python
import numpy as np

def kl_k1(log_p, log_q):
    """E_p[log p - log q]：，，"""
    return np.mean(log_p - log_q)

def kl_k3(log_p, log_q):
    """E_p[exp(log q - log p) - 1 - (log q - log p)]："""
    log_ratio = log_q - log_p
    return np.mean(np.exp(log_ratio) - 1 - log_ratio)
```

### PyTorch 

```python
import torch

def kl_penalty(log_probs, ref_log_probs, mode="k3"):
    """
    log_probs:     [B, seq_len]   p
    ref_log_probs: [B, seq_len]   q
    """
    if mode == "k1":
        return (log_probs - ref_log_probs).mean()

    log_ratio = ref_log_probs - log_probs   # log(q/p)
    return (torch.exp(log_ratio) - 1 - log_ratio).mean()
```

### 

 $p$， $\text{KL}(p \| q)$：

|  |                                                |                          |
| ------ | -------------------------------------------------- | ---------------------------- |
| k1     | $\mathbb{E}_p[\log \frac{p}{q}]$                   | ，， |
| k3     | $\mathbb{E}_p[\frac{q}{p} - 1 - \log \frac{q}{p}]$ | ， $\geq 0$，GRPO  |

::: warning 
k3  ratio  $q/p$（ref/current）。 $e^x - 1 - x \geq 0$  $x$ ，； $p/q$ ， $\text{KL}(p \| q)$。
:::

---

## 

|                 |                                                   |
| ------------------- | ----------------------------------------------------- |
| shift       | logits ****，labels ****： $t$  $t+1$ |
|  `ignore_index` | prompt  token  `-100`， loss              |
| k3 ratio      |  $q/p$（ref/current）；         |
| k1          | ，， bug              |
| softmax         |  `max(x)`  `exp`                                |
| `.contiguous()`     | PyTorch slice  `view` ， `.contiguous()`  |
