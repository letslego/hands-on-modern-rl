# Part 2:  — 

##  Part ？

。""， PPO 。， LLM 。

，：

- **MDP ** $(S, A, P, R, \gamma)$：""。
- ****：$V^\pi(s)$  $Q^\pi(s,a)$ """"。—— =  + 。
- **TD Error**：$\delta = r + \gamma V(s') - V(s)$，""， RL 。
- **DQN **：Q-Network（ Q ）、（）、（）。
- ****：$\nabla_\theta J = \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot G_t]$，，。
- **Actor-Critic**：Actor ，Critic ， $A(s,a) = Q(s,a) - V(s)$ 。
- **PPO **： $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$ ，。
- **GAE**：$\hat{A}_t = \sum_{k=0}^{\infty}(\gamma\lambda)^k \delta_{t+k}$，。

。

##  3 ：MDP——

### 

，""。****（Markov Decision Process, MDP）， $(S, A, P, R, \gamma)$ ：

- $S$ 。 CartPole ，$s = (\text{}, \text{}, \text{}, \text{}) \in \mathbb{R}^4$。
- $A$ 。 $\{a_1, a_2, \ldots\}$，。
- $P(s'|s,a)$ —— $s$  $a$  $s'$ 。""：，。，，。
- $R(s,a)$ —— $s$  $a$ 。
- $\gamma \in [0, 1)$ ——""""。$\gamma$  1 ， 0 。

 $\pi(a|s)$，，****：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

 $\gamma$？，，$\gamma < 1$  $G_t$ 。，，——" 100 "" 100 "。

### 

，：""？-""？。

**** $V^\pi(s)$  $s$ 、 $\pi$ ：

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t = s\right]$$

**** $Q^\pi(s, a)$  $s$  $a$、 $\pi$ ：

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t = s, a_t = a\right]$$

：$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$。，。

——""，：

$$V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

： = （）， =  + 。，。

### TD Error： RL 

， $P$  $R$（），。，——****（TD Error）：

$$\delta = r + \gamma V(s') - V(s)$$

TD Error ""。$V(s)$ ，$r + \gamma V(s')$ 。，$\delta = 0$；，$\delta > 0$， $V(s)$ 。

 RL 。 Q-Learning  DQN， REINFORCE  PPO， TD Error 。

###  GridWorld： Q-Learning

，" vs "：（），（）。 4×4 GridWorld ， Q-Learning ：

```python
Q = np.zeros((n_states, n_actions))  #  Q 

for episode in range(1000):
    state = env.reset()
    while not done:
        # ε-： ε ，
        action = epsilon_greedy(Q[state], epsilon)
        next_state, reward, done = env.step(action)
        #  TD Error  Q 
        td_target = reward + gamma * np.max(Q[next_state])
        Q[state, action] += alpha * (td_target - Q[state, action])
        state = next_state
```

 RL ： $Q(s,a)$ ， $\varepsilon$-， TD Error 。（ 16  Atari  $210 \times 160$ ），—— DQN 。

##  4 ：DQN——

### 

CartPole ， 4 。，Atari  $84 \times 84 \times 4$ ， $256^{28224}$ ——。 Q 。

DQN ** Q **。 $s$， Q  $Q(s, a_1), Q(s, a_2), \ldots$。 TD Error ：

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

 $\theta$ ，$\theta^-$ 。： $Q(s, a)$ " +  Q "。

### DQN 

。，（），。DQN ：

**（Experience Replay）**  $(s, a, r, s')$ ， batch。，。

```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

**（Target Network）** 。，。 TD Target ，""。，，；，。

**$\varepsilon$-** ，。

### DQN 

 DQN  2015  Atari ，。**Double DQN** """"： $\theta$  $\arg\max_{a'} Q(s', a'; \theta)$， $\theta^-$ 。 DQN  Q ——。

**Dueling DQN**  Q  $V(s)$  $A(s,a)$ ：

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a')$$

，$A(s,a) \approx 0$， $V(s)$。。

##  5 ：——

###  Value-Based  Policy-Based

DQN ： $Q(s,a)$， $\arg\max$ 。——**、**。 $[-10, 10]^6$， Q 。—— token 。

：，**** $\pi_\theta(a|s)$， $\theta$ 。， $J(\theta) = \mathbb{E}_{\pi_\theta}[G_t]$ ：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

：$\nabla_\theta \log \pi_\theta(a_t|s_t)$ " $a_t$  $s_t$ "，$G_t$ 。 $G_t > 0$，； $G_t < 0$，。 REINFORCE 。

```python
def reinforce_update(policy, optimizer, states, actions, returns):
    log_probs = []
    for s, a in zip(states, actions):
        dist = Categorical(policy(s))
        log_probs.append(dist.log_prob(a))

    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * G  # ： = 
    loss /= len(returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 

REINFORCE ：****。 episode  $G_t$  100 ，""，。。

**** $b(s)$，：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$

 $b(s)$  $a$，（ $\mathbb{E}_{a \sim \pi}[\nabla \log \pi(a|s)] = 0$），。 $V(s)$—— Critic。

##  6 ：Actor-Critic—— Critic 

### Actor-Critic 

 Actor（） Critic（）， Actor-Critic 。Actor ，Critic ""。""****：

$$A(s, a) = Q(s, a) - V(s)$$

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)
```

，Actor  $A = r + \gamma V(s') - V(s)$ ，Critic  TD Error ：

```python
# 
_, next_value = model(next_state)
td_target = reward + gamma * next_value * (1 - done)
td_error = td_target - value

actor_loss = -log_prob * td_error.detach()  # Actor：
critic_loss = td_error ** 2                 # Critic： TD Error 
loss = actor_loss + critic_loss
```

##  7 ：PPO——

### 

Actor-Critic  REINFORCE ，——，，。

PPO（Proximal Policy Optimization）。****：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$$

 $r_t = 1$ ，。$r_t > 1$ ，$r_t < 1$ 。PPO ：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t,\;\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right)\right]$$

 $\varepsilon$  0.2。 $\hat{A}_t > 0$（），PPO  $r_t$  $1+\varepsilon$，； $\hat{A}_t < 0$（）， $r_t$  $1-\varepsilon$。""****。

 PPO ：

$$L(\theta) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 \mathcal{H}[\pi_\theta]$$

 $L^{\text{VF}}$  Critic （MSE），$\mathcal{H}$ ，，。

### GAE：

 $\hat{A}_t$  PPO 。： TD Error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$——； $G_t - V(s_t)$——。**GAE**（Generalized Advantage Estimation）：

$$\hat{A}_t^{\text{GAE}} = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k}$$

 $\lambda \in [0, 1]$ 。$\lambda = 0$  TD（，），$\lambda = 1$ （，）。 $\lambda = 0.95$。

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages
```

###  LunarLander  LLM

PPO  Gym  LunarLander  RL ，。 RLHF ，PPO ：Actor（）、Critic（）、Reference（， KL ）、Reward Model（）。 Bradley-Terry  RM ：

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

 8 —— DPO ， RM 。

## 

Part 2  RL ：MDP  →  → DQN  Q ， → ， → Actor-Critic  Critic  → PPO  GAE 。

， LLM 。 PPO ， GRPO ； Actor-Critic ， RLHF 。

> ****：[Part 3: LLM ](/chapter08_rlhf/intro)
