"""
12： Agent 
==========================================================

 Agent， REINFORCE 
。

：
  - 
  - Agent  3 ：
      1. search(query)      —— 
      2. calculate(expr)    —— 
      3. run_code(code)     —— 
  -  →  (+1)
  -  →  (-0.1)

：
  - ：（softmax ）
  - ：REINFORCE（）
  -  50  episode，

：
    python tool_use_agent.py
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

# 
TOOL_NAMES = ["search", "calculate", "run_code"]
TOOL_DESCRIPTIONS = {
    "search": "，",
    "calculate": "，",
    "run_code": "，",
}

# ：
# ：（ 1/3），
TRAINING_QUERIES = [
    # ---- search  ----
    {"query": "？",                     "correct_tool": "search"},
    {"query": "Python ？",               "correct_tool": "search"},
    {"query": "？",                           "correct_tool": "search"},
    {"query": "？",                      "correct_tool": "search"},
    {"query": "？",                        "correct_tool": "search"},
    {"query": "？",             "correct_tool": "search"},
    {"query": "？",                "correct_tool": "search"},
    {"query": "？",                          "correct_tool": "search"},
    # ---- calculate  ----
    {"query": " 123 + 456",                       "correct_tool": "calculate"},
    {"query": "25  37 ？",                   "correct_tool": "calculate"},
    {"query": "1024  8 ？",                   "correct_tool": "calculate"},
    {"query": " 17 ",                          "correct_tool": "calculate"},
    {"query": "， 5",                  "correct_tool": "calculate"},
    {"query": "3  10 ？",                    "correct_tool": "calculate"},
    {"query": "99  7×8 ？",               "correct_tool": "calculate"},
    {"query": " 1024 ",                    "correct_tool": "calculate"},
    # ---- run_code  ----
    {"query": "",                      "correct_tool": "run_code"},
    {"query": " Python ",              "correct_tool": "run_code"},
    {"query": "",              "correct_tool": "run_code"},
    {"query": "",              "correct_tool": "run_code"},
    {"query": "",                  "correct_tool": "run_code"},
    {"query": " REST API",                 "correct_tool": "run_code"},
    {"query": "",                    "correct_tool": "run_code"},
    {"query": "",                      "correct_tool": "run_code"},
]


def simulate_tool_result(tool_name, query, correct_tool):
    """
    

    ：
        tool_name:     
        query:         
        correct_tool:  
    ：
        result_dict:   
    """
    if tool_name == correct_tool:
        # 
        return {"success": True, "message": f" {tool_name} "}
    else:
        # 
        return {"success": False, "message": f" {tool_name} "}


# ==========================================
# ：
# ==========================================

class ToolPolicy:
    """
    ：

    ：
      -  logits（）
      -  softmax  logits 
      -  logits 

    ：
      - ， logit 
      - ， logit 
      - softmax  1

    ：
        n_tools: 
        learning_rate: 
    """

    def __init__(self, n_tools=3, learning_rate=0.05):
        self.n_tools = n_tools
        self.learning_rate = learning_rate

        #  logits： → （1/3, 1/3, 1/3）
        self.query_type_logits = {
            "search":    np.zeros(n_tools),
            "calculate": np.zeros(n_tools),
            "run_code":  np.zeros(n_tools),
        }

    def get_query_type(self, query):
        """
        （）

        ， NLP 。
        ：
          -  → calculate
          -  → run_code
          -  → search
        """
        calc_keywords = ["", "", "", "", "", "", "",
                         "", "", "", "", ""]
        code_keywords = ["", "", "", "", "", "", "API",
                         "", "", "", "", ""]

        for kw in calc_keywords:
            if kw in query:
                return "calculate"
        for kw in code_keywords:
            if kw in query:
                return "run_code"
        return "search"

    def get_probabilities(self, query_type):
        """
        

         softmax  logits ：
          π(a|s) = exp(logit_a) / Σ exp(logit_i)
        """
        logits = self.query_type_logits[query_type]
        #  softmax
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def sample_action(self, query_type):
        """
        （）

         argmax！。
        ，。
        """
        probs = self.get_probabilities(query_type)
        action = np.random.choice(self.n_tools, p=probs)
        return action, probs[action]

    def update(self, query_type, action, reward):
        """
        REINFORCE 

        ：
          θ ← θ + α * ∇log π(a|s) * G

         softmax ，：
          ∇log π(a_k|s) = e_k - π(s)
           e_k  one-hot ，π(s) 

        ：
          -  reward > 0：
          -  reward < 0：
        """
        probs = self.get_probabilities(query_type)

        # 
        # ∇log π(a_k|s) = e_k - π(s)
        grad = -probs.copy()       # -π(s)
        grad[action] += 1.0        # +e_k

        # ：θ ← θ + α * grad * reward
        self.query_type_logits[query_type] += self.learning_rate * grad * reward


# ==========================================
# ：
# ==========================================

def train(policy, n_episodes=50, queries_per_episode=8):
    """
    REINFORCE 

     episode：
      1. 
      2. Agent 
      3. （ +1.0， -0.1）
      4. （）

    ：
        policy:             
        n_episodes:         
        queries_per_episode: 
    ：
        history: 
    """
    # 
    history = {
        "episode_rewards": [],      # 
        "episode_accuracy": [],     # 
        "tool_probs_history": {     # 
            "search": [],
            "calculate": [],
            "run_code": [],
        },
    }

    for episode in range(n_episodes):
        episode_reward = 0.0
        correct_count = 0

        # 
        indices = np.random.choice(len(TRAINING_QUERIES),
                                   size=min(queries_per_episode, len(TRAINING_QUERIES)),
                                   replace=False)

        for idx in indices:
            query_data = TRAINING_QUERIES[idx]
            query = query_data["query"]
            correct_tool = query_data["correct_tool"]

            # ：
            query_type = policy.get_query_type(query)

            # ：
            action, prob = policy.sample_action(query_type)
            chosen_tool = TOOL_NAMES[action]

            # ：，
            result = simulate_tool_result(chosen_tool, query, correct_tool)

            if result["success"]:
                reward = 1.0    # ：
                correct_count += 1
            else:
                reward = -0.1   # ：（）

            # ：
            policy.update(query_type, action, reward)
            episode_reward += reward

        # 
        avg_reward = episode_reward / queries_per_episode
        accuracy = correct_count / queries_per_episode
        history["episode_rewards"].append(avg_reward)
        history["episode_accuracy"].append(accuracy)

        # 
        for qt in ["search", "calculate", "run_code"]:
            probs = policy.get_probabilities(qt)
            history["tool_probs_history"][qt].append(probs.copy())

        #  10 
        if (episode + 1) % 10 == 0:
            print(f"   {episode+1:3d}/{n_episodes} | "
                  f": {avg_reward:+.3f} | "
                  f": {accuracy:.1%}")

    return history


# ==========================================
# ：
# ==========================================
print("=" * 70)
print("  12： Agent ")
print("=" * 70)

np.random.seed(42)  # 

# 
policy = ToolPolicy(n_tools=3, learning_rate=0.05)

# ----  ----
print("\n【】（，）:")
print(f"  {'':<12} {'search':<12} {'calculate':<12} {'run_code':<12}")
print(f"  {'─' * 48}")
for qt in ["search", "calculate", "run_code"]:
    probs = policy.get_probabilities(qt)
    print(f"  {qt:<12} {probs[0]:<12.4f} {probs[1]:<12.4f} {probs[2]:<12.4f}")

# 
correct_before = 0
total_before = len(TRAINING_QUERIES)
for qd in TRAINING_QUERIES:
    qt = policy.get_query_type(qd["query"])
    action, _ = policy.sample_action(qt)
    if TOOL_NAMES[action] == qd["correct_tool"]:
        correct_before += 1
accuracy_before = correct_before / total_before
print(f"\n  : {correct_before}/{total_before} = {accuracy_before:.1%}")

# ----  ----
print("\n" + "─" * 70)
print("   REINFORCE （50 ）")
print("─" * 70)

history = train(policy, n_episodes=50, queries_per_episode=8)

# ----  ----
print("\n" + "─" * 70)
print("  ！")
print("─" * 70)

print("\n【】（）:")
print(f"  {'':<12} {'search':<12} {'calculate':<12} {'run_code':<12}  {''}")
print(f"  {'─' * 64}")
for qt, optimal in [("search", "search"), ("calculate", "calculate"), ("run_code", "run_code")]:
    probs = policy.get_probabilities(qt)
    print(f"  {qt:<12} {probs[0]:<12.4f} {probs[1]:<12.4f} {probs[2]:<12.4f}  {optimal}")

# （：）
correct_after = 0
for qd in TRAINING_QUERIES:
    qt = policy.get_query_type(qd["query"])
    probs = policy.get_probabilities(qt)
    best_action = np.argmax(probs)
    if TOOL_NAMES[best_action] == qd["correct_tool"]:
        correct_after += 1
accuracy_after = correct_after / total_before
print(f"\n  （）: {correct_after}/{total_before} = {accuracy_after:.1%}")
print(f"  : {accuracy_after - accuracy_before:+.1%}")

# ----  ----
print("\n【】:")
print(f"  {'':<30} {'':<10} {'':<10} {''}")
print(f"  {'─' * 65}")
for qd in TRAINING_QUERIES:
    qt = policy.get_query_type(qd["query"])
    probs = policy.get_probabilities(qt)
    best_action = np.argmax(probs)
    predicted_tool = TOOL_NAMES[best_action]
    correct = predicted_tool == qd["correct_tool"]
    mark = "" if correct else ""
    query_short = qd["query"][:28] + ".." if len(qd["query"]) > 28 else qd["query"]
    print(f"  {query_short:<30} {qd['correct_tool']:<10} {predicted_tool:<10} {mark}")


# ==========================================
# ：
# ==========================================
print("\n...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(" Agent —— REINFORCE ", fontsize=18, fontweight="bold")

# ---- 1： ----
ax1 = axes[0]

episodes = np.arange(1, len(history["episode_accuracy"]) + 1)

# 
for qt_idx, (qt, color, marker) in enumerate([
    ("search",    "#2196F3", "o"),
    ("calculate", "#FF9800", "s"),
    ("run_code",  "#4CAF50", "^"),
]):
    probs_history = np.array(history["tool_probs_history"][qt])
    # ，
    ax1.plot(episodes, probs_history[:, qt_idx],
             marker=marker, linewidth=2.5, markersize=6,
             color=color, label=f"{qt}  →  {TOOL_NAMES[qt_idx]}")

ax1.axhline(y=1/3, color="gray", linestyle="--", alpha=0.5, label=" (1/3)")
ax1.set_title("", fontsize=14, fontweight="bold")
ax1.set_xlabel("", fontsize=12)
ax1.set_ylabel("", fontsize=12)
ax1.legend(fontsize=10, loc="center right")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)

# 
ax1.annotate("： (~33%)",
             xy=(1, 1/3), xytext=(8, 0.15),
             fontsize=10, color="gray",
             arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
ax1.annotate("：",
             xy=(50, 0.9), xytext=(30, 0.95),
             fontsize=10, color="green", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="green", lw=1.5))

# ---- 2： ----
ax2 = axes[1]

ax2.plot(episodes, history["episode_accuracy"],
         linewidth=1.5, alpha=0.4, color="steelblue", label="（）")

# 
window = 5
if len(history["episode_accuracy"]) >= window:
    moving_avg = []
    for i in range(len(history["episode_accuracy"])):
        start = max(0, i - window + 1)
        moving_avg.append(np.mean(history["episode_accuracy"][start:i+1]))
    ax2.plot(episodes, moving_avg, color="crimson", linewidth=2.5,
             label=f"（={window}）")

ax2.axhline(y=1/3, color="gray", linestyle="--", alpha=0.5, label=" (33.3%)")
ax2.axhline(y=accuracy_after, color="green", linestyle=":", alpha=0.7,
            label=f" ({accuracy_after:.1%})")

ax2.set_title("", fontsize=14, fontweight="bold")
ax2.set_xlabel("", fontsize=12)
ax2.set_ylabel("", fontsize=12)
ax2.legend(fontsize=10, loc="lower right")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("output/tool_use_agent_training.png", dpi=150, bbox_inches="tight")
print(": output/tool_use_agent_training.png")
plt.show()


# ==========================================
# ：
# ==========================================
print("\n" + "=" * 70)
print("  ")
print("=" * 70)
print(f"""
  1.  Agentic RL 
     - Agent 
     - """"

  2. REINFORCE 
     - ：， {accuracy_before:.1%}
     - ：， {accuracy_after:.1%}
     -  50  episode 

  3. ：
     - （ Transformer）
     - （）
     - （PRM）
     - （，）

  4. ：
     - ：
     - ：Agent 
     -  multi_turn_rl.py 
""")
print("=" * 70)
