# C.3 DPO 

DPO Loss ****，。 DPO、IPO、KTO、SimPO 。

---

## DPO Loss

****： RLHF " reward model + PPO"， $(y_w, y_l)$ ， reward  Bradley-Terry ， critic。

****：

- `pi_chosen` / `pi_rejected`： $\pi_\theta$ / log 
- `ref_chosen` / `ref_rejected`： $\pi_{ref}$ / log （ `detach`）
- `log_ratio_w` / `log_ratio_l`：$\log\frac{\pi_\theta}{\pi_{ref}}$，（ $\beta$）
- `beta`（$\beta$）：， reference ；， 0.1~0.5

### 

> **4  logp（2  × 2 ）：" − "；、 β、 sigmoid、 log。**

$$\mathcal{L}_{DPO} = -\mathbb{E}\Big[\log\sigma\Big(\beta\Big(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big)\Big]$$

### 

```
log_ratio_w = log_pi_theta(y_w|x) - log_pi_ref(y_w|x)   # 
log_ratio_l = log_pi_theta(y_l|x) - log_pi_ref(y_l|x)   # 
logits      = beta * (log_ratio_w - log_ratio_l)        #  × 
loss        = -log_sigmoid(logits)                       #  →  BT 
```

### Python 

```python
import numpy as np

def log_sigmoid(x):
    return -np.logaddexp(0, -x)  # 

def dpo_loss(logp_chosen, logp_rejected,
             logp_ref_chosen, logp_ref_rejected,
             beta=0.1):
    """
    : scalar  [B]， loss
    """
    log_ratio_w = logp_chosen - logp_ref_chosen
    log_ratio_l = logp_rejected - logp_ref_rejected
    logits = beta * (log_ratio_w - log_ratio_l)
    return -log_sigmoid(logits).mean()
```

### PyTorch 

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1):
    """: [B]"""
    log_ratio_w = policy_chosen_logps - ref_chosen_logps
    log_ratio_l = policy_rejected_logps - ref_rejected_logps
    logits = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(logits).mean()
```

---

## IPO

****：DPO  $-\log\sigma$ ，。IPO ， $\frac{1}{2\beta}$，，。

****：

- `delta`：$\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$， β 
-  $\frac{1}{2\beta}$，

### 

> ** DPO  $-\log\sigma$  $(\Delta - \frac{1}{2\beta})^2$——，。**

$$\mathcal{L}_{IPO} = \Big(\Delta - \frac{1}{2\beta}\Big)^2, \quad \Delta = \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$$

### 

```
delta = log_ratio_w - log_ratio_l     #  β 
loss  = (delta - 1 / (2 * beta)) ** 2 # 
```

### PyTorch 

```python
def ipo_loss(log_ratio_w, log_ratio_l, beta=0.1):
    delta = log_ratio_w - log_ratio_l
    return ((delta - 1.0 / (2 * beta)) ** 2).mean()
```

---

## KTO

****：DPO/IPO  chosen-rejected ，"/"****。KTO ，，（）。

****：

- `log_ratio`： $\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$
- `z_ref`：， desirable  logit  EMA， $z_{ref} \approx \beta \mathbb{E}[\log\frac{\pi_\theta}{\pi_{ref}}]$， `detach`
- `lambda_D` / `lambda_U`：/， $\lambda_U > \lambda_D$（）

### 

> **： $\beta\log r$  $z_{ref}$， $z_{ref}$ ， $-\log\sigma$。**

### 

```
logit = beta * log_ratio                  #  × 
loss_desirable   = -log_sigmoid(logit - z_ref)        # : logit 
loss_undesirable = -log_sigmoid(z_ref - logit)        # : logit 
loss = lambda_D * loss_desirable + lambda_U * loss_undesirable
```

### PyTorch 

```python
import torch
import torch.nn.functional as F

def kto_loss(log_ratio, is_desirable, z_ref=0.0,
             beta=0.1, lambda_D=1.0, lambda_U=1.33):
    """
    log_ratio: [B] = log_pi(y|x) - log_ref(y|x)
    is_desirable: [B] bool; z_ref  β（TRL ）
    """
    logit = beta * log_ratio
    loss = torch.zeros_like(log_ratio)
    d = is_desirable
    u = ~is_desirable
    if d.any():
        loss[d] = lambda_D * -F.logsigmoid(logit[d] - z_ref)
    if u.any():
        loss[u] = lambda_U * -F.logsigmoid(z_ref - logit[u])
    return loss.mean()
```

---

## SimPO

****：DPO ，； log ，。SimPO ****， ref 。

****：

- `chosen_logps` / `rejected_logps`：/ log （）
- `chosen_lengths` / `rejected_lengths`：/ token 
- `beta`（$\beta$）：（SimPO  2.0， DPO ）
- `gamma`（$\gamma$）： margin，""

### 

> **DPO  ref：logp ， β  margin $\gamma$。**

### 

```
logp_w = log_pi(chosen)  / len(chosen)   # 
logp_l = log_pi(rejected) / len(rejected)
logits = beta * (logp_w - logp_l) - gamma #  margin
loss   = -log_sigmoid(logits)
```

### PyTorch 

```python
import torch.nn.functional as F

def simpo_loss(chosen_logps, rejected_logps,
               chosen_lengths, rejected_lengths,
               beta=2.0, gamma=0.5):
    logp_w = chosen_logps / chosen_lengths
    logp_l = rejected_logps / rejected_lengths
    logits = beta * (logp_w - logp_l) - gamma
    return -F.logsigmoid(logits).mean()
```

---

## DPO 

|   |  ref? | ?            |                                            |
| ----- | --------- | -------------------- | -------------------------------------------------- |
| DPO   |         |  (chosen/rejected) | $-\log\sigma(\beta\Delta)$，                 |
| IPO   |         |                    |  $(\Delta - \frac{1}{2\beta})^2$， |
| KTO   |         | （/）      |  $\pm$ sigmoid +  $z_{ref}$，    |
| SimPO | ****    |                    |  log-prob + margin $\gamma$              |

：$\Delta$  log-ratio 。

---

## 

|                    |                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
|  log_prob      | ：（chosen + rejected），；ref  `detach`                             |
| log-prob           | DPO/IPO/KTO  $\log\frac{\pi_\theta}{\pi_{ref}}$，**** raw log-prob                     |
| `log_sigmoid`  | PyTorch  `F.logsigmoid` ； `logaddexp`                                         |
| $\beta$            | $\beta$ ，；                                                 |
| IPO  sigmoid       | IPO  $\frac{1}{2\beta}$， sigmoid                                            |
| KTO  $z_{ref}$   | TRL  $z_{ref}$  $\beta$（ logit  EMA）； $\beta \cdot \text{log\_ratio} - z_{ref}$ |
| SimPO          |  SimPO ， log-prob                                             |
| chosen/rejected    | ：chosen                                                               |
