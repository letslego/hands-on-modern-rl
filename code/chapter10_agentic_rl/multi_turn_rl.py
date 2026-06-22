"""
12： RL —— ORM  PRM 
==========================================================

 Agent（ 3~5 ），
：

  1. ORM（Outcome Reward Model）：
     （1.0  0.0），

  2. PRM（Process Reward Model）：
     （0.0~1.0），

：
  - （Credit Assignment）：？
  - ：G_t = r_t + γ * G_{t+1}，γ 
  - γ  → ；γ  → 

：
    python multi_turn_rl.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)

# ，
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ：
# ==========================================
# Agent ：
#   - calculator: 
#   - search:      
#   - code_executor: 
# 

def tool_calculator(query):
    """
    
    ，（）
    """
    # ：
    if "123 + 456" in query:
        return {"result": "579", "correct": True}
    elif "25 * 4" in query:
        return {"result": "100", "correct": True}
    elif "sqrt(144)" in query or "144" in query:
        return {"result": "12", "correct": True}
    else:
        # ， 70% 
        correct = np.random.random() < 0.7
        return {"result": "", "correct": correct}


def tool_search(query):
    """
    
    
    """
    # 
    if "Python" in query or "python" in query:
        return {"result": "Python ...", "correct": True}
    elif "RL" in query or "" in query:
        return {"result": "...", "correct": True}
    else:
        correct = np.random.random() < 0.7
        return {"result": "...", "correct": correct}


def tool_code_executor(code):
    """
    
    
    """
    # 
    if "print" in code or "def " in code:
        return {"result": "", "correct": True}
    elif "import" in code:
        return {"result": "", "correct": True}
    else:
        correct = np.random.random() < 0.6
        return {"result": "", "correct": correct}


# ： → (, )
TOOLS = {
    "calculator": (tool_calculator, "，"),
    "search": (tool_search, "，"),
    "code_executor": (tool_code_executor, "，"),
}


# ==========================================
# ：
# ==========================================
# ， 3~5 

SCENARIOS = [
    {
        "task": " 123 + 456 ， Python ，",
        "turns": [
            {"tool": "calculator",  "query": "123 + 456",   "description": "："},
            {"tool": "search",      "query": "Python ",  "description": "： Python "},
            {"tool": "code_executor","query": "print(579)",  "description": "："},
        ],
    },
    {
        "task": "， 25*4，",
        "turns": [
            {"tool": "search",       "query": "",       "description": "："},
            {"tool": "calculator",   "query": "25 * 4",            "description": "："},
            {"tool": "code_executor","query": "def hello(): pass",  "description": "："},
        ],
    },
    {
        "task": "，，，",
        "turns": [
            {"tool": "calculator",    "query": "sqrt(144)",         "description": "："},
            {"tool": "search",        "query": "",       "description": "："},
            {"tool": "code_executor", "query": "import numpy",       "description": "："},
            {"tool": "search",        "query": "",       "description": "："},
        ],
    },
    {
        "task": "，，",
        "turns": [
            {"tool": "search",        "query": "",       "description": "："},
            {"tool": "code_executor", "query": "import math",        "description": "："},
            {"tool": "calculator",    "query": "",         "description": "："},
        ],
    },
    {
        "task": " RL ，，， PPO",
        "turns": [
            {"tool": "search",        "query": "RL ",        "description": "："},
            {"tool": "code_executor", "query": "def train(): pass",   "description": "："},
            {"tool": "calculator",    "query": "",        "description": "："},
            {"tool": "search",        "query": "PPO ",       "description": "： PPO"},
            {"tool": "code_executor", "query": "print('done')",       "description": "："},
        ],
    },
]


# ==========================================
# ：ORM  PRM 
# ==========================================

def compute_orm_rewards(turns):
    """
    ORM（Outcome Reward Model）：

    ， 1.0；
    ， 0.0。

    ——。
    """
    n = len(turns)
    rewards = [0.0] * n  # ： 0

    # ：
    all_correct = all(turn.get("correct", False) for turn in turns)
    final_reward = 1.0 if all_correct else 0.0

    # 
    rewards[-1] = final_reward

    return rewards


def compute_prm_rewards(turns):
    """
    PRM（Process Reward Model）：

    ， 0.0~1.0 。
    ——。

    ：
    - ： + 
    - ：
    """
    rewards = []
    for turn in turns:
        correct = turn.get("correct", False)

        if correct:
            # ：
            # 
            base_reward = 0.7 + np.random.uniform(0, 0.3)  # 0.7 ~ 1.0
        else:
            # ：（）
            base_reward = np.random.uniform(0.0, 0.3)  # 0.0 ~ 0.3

        rewards.append(round(base_reward, 3))

    return rewards


def compute_discounted_returns(rewards, gamma=0.99):
    """
    ：G_t = r_t + γ * G_{t+1}

    。
    γ（）：
      - γ  1.0：（）
      - γ  0.0：（）

    ：
        rewards: 
        gamma:   
    ：
        returns: 
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ==========================================
# ：
# ==========================================
print("=" * 70)
print("  12： RL —— ORM vs PRM ")
print("=" * 70)

np.random.seed(42)  # ，

# 
all_orm_rewards = []   # ORM 
all_prm_rewards = []   # PRM 
all_orm_returns = {}   # ORM （ gamma）
all_prm_returns = {}   # PRM （ gamma）
gamma_values = [0.5, 0.9, 0.99]  # 

for gamma in gamma_values:
    all_orm_returns[gamma] = []
    all_prm_returns[gamma] = []

# 
for idx, scenario in enumerate(SCENARIOS):
    print(f"\n{'─' * 70}")
    print(f"   {idx + 1}：{scenario['task']}")
    print(f"   {len(scenario['turns'])} ")
    print(f"{'─' * 70}")

    # 
    turns = scenario["turns"]
    for t, turn in enumerate(turns):
        tool_name = turn["tool"]
        query = turn["query"]

        # 
        tool_func, _ = TOOLS[tool_name]
        result = tool_func(query)
        turn["correct"] = result["correct"]

        status = "" if result["correct"] else ""
        print(f"   {t+1} ：{turn['description']}")
        print(f"          {tool_name}({query}) → {status}")

    # ---- ORM  ----
    orm_rewards = compute_orm_rewards(turns)
    all_orm_rewards.append(orm_rewards)

    print(f"\n  [ORM ] :")
    for t, r in enumerate(orm_rewards):
        bar = "█" * int(r * 20)
        print(f"     {t+1} : {r:.1f}  {bar}")

    # ---- PRM  ----
    prm_rewards = compute_prm_rewards(turns)
    all_prm_rewards.append(prm_rewards)

    print(f"\n  [PRM ] :")
    for t, r in enumerate(prm_rewards):
        bar = "█" * int(r * 20)
        print(f"     {t+1} : {r:.3f}  {bar}")

    # ---- （ gamma ）----
    for gamma in gamma_values:
        orm_returns = compute_discounted_returns(orm_rewards, gamma=gamma)
        prm_returns = compute_discounted_returns(prm_rewards, gamma=gamma)
        all_orm_returns[gamma].append(orm_returns)
        all_prm_returns[gamma].append(prm_returns)

    #  gamma=0.99 
    gamma_demo = 0.99
    orm_ret_demo = compute_discounted_returns(orm_rewards, gamma=gamma_demo)
    prm_ret_demo = compute_discounted_returns(prm_rewards, gamma=gamma_demo)

    print(f"\n  （γ = {gamma_demo}）:")
    print(f"    {'':<6} {'':<12} {' G_t':<16} {''}")
    print(f"    {'─' * 60}")

    # ORM 
    print(f"    [ORM ]")
    G = 0.0
    for t in reversed(range(len(orm_rewards))):
        old_G = G
        G = orm_rewards[t] + gamma_demo * old_G
        formula = f"G_{t} = {orm_rewards[t]:.1f} + {gamma_demo} * {old_G:.4f} = {G:.4f}"
        print(f"     {t+1}   r={orm_rewards[t]:<8.1f}  G={G:<12.4f}  {formula}")

    # PRM 
    print(f"    [PRM ]")
    G = 0.0
    for t in reversed(range(len(prm_rewards))):
        old_G = G
        G = prm_rewards[t] + gamma_demo * old_G
        formula = f"G_{t} = {prm_rewards[t]:.3f} + {gamma_demo} * {old_G:.4f} = {G:.4f}"
        print(f"     {t+1}   r={prm_rewards[t]:<8.3f}  G={G:<12.4f}  {formula}")


# ==========================================
# ：ORM vs PRM 
# ==========================================
print("\n" + "=" * 70)
print("  ORM vs PRM ")
print("=" * 70)

print("\n  【】")
for idx in range(len(SCENARIOS)):
    n_turns = len(SCENARIOS[idx]["turns"])
    orm_nonzero = sum(1 for r in all_orm_rewards[idx] if r > 0)
    prm_nonzero = sum(1 for r in all_prm_rewards[idx] if r > 0)
    print(f"     {idx+1}（{n_turns} ）:"
          f" ORM  = {orm_nonzero}/{n_turns},"
          f" PRM  = {prm_nonzero}/{n_turns}")

print(f"\n  :")
print(f"    - ORM ：，")
print(f"    - PRM ：，")
print(f"    -  Agent，PRM ")

print("\n  【 γ 】")
for gamma in gamma_values:
    print(f"\n    γ = {gamma}:")
    for idx in range(len(SCENARIOS)):
        orm_ret = all_orm_returns[gamma][idx]
        prm_ret = all_prm_returns[gamma][idx]
        n = len(orm_ret)
        print(f"       {idx+1}（{n} ）:")
        print(f"        ORM : {[f'{v:.4f}' for v in orm_ret]}")
        print(f"        PRM : {[f'{v:.4f}' for v in prm_ret]}")

print(f"\n  γ :")
print(f"    - γ=0.5: ，")
print(f"    - γ=0.9: ，")
print(f"    - γ=0.99: ，")


# ==========================================
# ：
# ==========================================
print("\n...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle(" RL —— ORM vs PRM ", fontsize=18, fontweight="bold")

# ---- 1：Turn  ----
ax1 = axes[0, 0]

# 
max_turns = max(len(r) for r in all_orm_rewards)
n_scenarios = len(SCENARIOS)

heatmap_orm = np.zeros((n_scenarios, max_turns))
heatmap_prm = np.zeros((n_scenarios, max_turns))

for i in range(n_scenarios):
    for j in range(len(all_orm_rewards[i])):
        heatmap_orm[i, j] = all_orm_rewards[i][j]
        heatmap_prm[i, j] = all_prm_rewards[i][j]

#  PRM （）
im = ax1.imshow(heatmap_prm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax1.set_xticks(range(max_turns))
ax1.set_xticklabels([f" {i+1} " for i in range(max_turns)])
ax1.set_yticks(range(n_scenarios))
ax1.set_yticklabels([f" {i+1}" for i in range(n_scenarios)])
ax1.set_title("PRM ", fontsize=14, fontweight="bold")
ax1.set_xlabel("", fontsize=12)
ax1.set_ylabel("", fontsize=12)

# 
for i in range(n_scenarios):
    for j in range(len(all_prm_rewards[i])):
        ax1.text(j, i, f"{heatmap_prm[i, j]:.2f}",
                 ha="center", va="center", fontsize=10, fontweight="bold")

fig.colorbar(im, ax=ax1, label="")

# ---- 2：ORM vs PRM （gamma=0.99）----
ax2 = axes[0, 1]

gamma_plot = 0.99
colors_orm = plt.cm.Blues(np.linspace(0.4, 0.9, n_scenarios))
colors_prm = plt.cm.Reds(np.linspace(0.4, 0.9, n_scenarios))

for i in range(n_scenarios):
    n = len(all_orm_returns[gamma_plot][i])
    x = np.arange(n)

    # ORM ，PRM 
    ax2.plot(x, all_orm_returns[gamma_plot][i],
             marker="o", linestyle="--", linewidth=2, markersize=6,
             color=colors_orm[i],
             label=f"{i+1} ORM" if i == 0 else None)
    ax2.plot(x, all_prm_returns[gamma_plot][i],
             marker="s", linestyle="-", linewidth=2, markersize=6,
             color=colors_prm[i],
             label=f"{i+1} PRM" if i == 0 else None)

# （）
ax2.plot([], [], marker="o", linestyle="--", color="steelblue", linewidth=2, label="ORM ")
ax2.plot([], [], marker="s", linestyle="-", color="crimson", linewidth=2, label="PRM ")

ax2.set_title(f"ORM vs PRM （γ={gamma_plot}）", fontsize=14, fontweight="bold")
ax2.set_xlabel("", fontsize=12)
ax2.set_ylabel(" G_t", fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 
ax2.annotate("ORM: \n→  ≈ 0",
             xy=(0.02, 0.95), xycoords="axes fraction",
             fontsize=10, color="steelblue", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
ax2.annotate("PRM: \n→ ",
             xy=(0.02, 0.75), xycoords="axes fraction",
             fontsize=10, color="crimson", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.3))

# ---- 3： gamma （5，PRM）----
ax3 = axes[1, 0]

# （55）
demo_idx = 4  # 5（4）
colors_gamma = ["#E91E63", "#FF9800", "#4CAF50"]

for gi, gamma in enumerate(gamma_values):
    ret = all_prm_returns[gamma][demo_idx]
    x = np.arange(len(ret))
    ax3.plot(x, ret, marker="o", linewidth=2.5, markersize=8,
             color=colors_gamma[gi], label=f"γ = {gamma}")

ax3.set_title(f" γ （5，PRM）", fontsize=14, fontweight="bold")
ax3.set_xlabel("", fontsize=12)
ax3.set_ylabel(" G_t", fontsize=12)
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(len(all_prm_returns[0.99][demo_idx])))
ax3.set_xticklabels([f"{i+1}" for i in range(len(all_prm_returns[0.99][demo_idx]))])

# ---- 4：ORM vs PRM  ----
ax4 = axes[1, 1]

x_pos = np.arange(n_scenarios)
bar_width = 0.35

# 
orm_density = []
prm_density = []
for i in range(n_scenarios):
    n = len(all_orm_rewards[i])
    orm_density.append(sum(1 for r in all_orm_rewards[i] if r > 0) / n * 100)
    prm_density.append(sum(1 for r in all_prm_rewards[i] if r > 0) / n * 100)

bars1 = ax4.bar(x_pos - bar_width/2, orm_density, bar_width,
                label='ORM（）', color='steelblue', alpha=0.8)
bars2 = ax4.bar(x_pos + bar_width/2, prm_density, bar_width,
                label='PRM（）', color='crimson', alpha=0.8)

ax4.set_title("（）", fontsize=14, fontweight="bold")
ax4.set_xlabel("", fontsize=12)
ax4.set_ylabel(" (%)", fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f"{i+1}" for i in range(n_scenarios)])
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim(0, 110)

# 
for bar in bars1:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight="bold")
for bar in bars2:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("output/multi_turn_orm_vs_prm.png", dpi=150, bbox_inches="tight")
print(": output/multi_turn_orm_vs_prm.png")
plt.show()


# ==========================================
# ：
# ==========================================
print("\n" + "=" * 70)
print("  ")
print("=" * 70)
print("""
  1.  Agent RL 
     - Agent 
     - /？

  2. ORM ：
     ✓ ，
     ✗ ，""
     ✗ ，

  3. PRM ：
     ✓ ，
     ✓ """"
     ✗ ，
     ✗ 

  4.  γ ：
     - γ ，（）
     - γ ，（）
     -  γ ≥ 0.9

  5. ：
     - ：ORM （）
     - ：PRM （、）
     - ：PRM  + ORM 
""")
print("=" * 70)
