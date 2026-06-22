"""
3：


：
1.  V^π(s)
2. （Value Iteration） V^π(s)  V*(s)
3. ，
4. 

：
    python bellman_equation_verify.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)


# ==========================================
# ： 3  MDP
# ==========================================
#  MDP，。
#
# ：S = {s0, s1, s2}
# ：A = {a0, a1}（）
#
#  P(s'|s,a)  R(s,a) ：
#
#  s0 ：
#    a0 →  s1（ 1.0）， = 1
#    a1 →  s2（ 1.0）， = 2
#
#  s1 ：
#    a0 →  s0（ 0.5）， = -1
#                  s2（ 0.5）， = -1
#    a1 →  s1（ 0.8）， = 0
#                  s2（ 0.2）， = 0
#
#  s2 ：
#    a0 →  s1（ 1.0）， = 3
#    a1 →  s2（ 1.0）， = 1（）

# 
N_STATES = 3
N_ACTIONS = 2

# ：P[s][a] = {: }
P = {
    0: {  # s0
        0: {1: 1.0},    # a0 → s1  1.0
        1: {2: 1.0},    # a1 → s2  1.0
    },
    1: {  # s1
        0: {0: 0.5, 2: 0.5},  # a0 → s0  0.5, s2  0.5
        1: {1: 0.8, 2: 0.2},  # a1 → s1  0.8, s2  0.2
    },
    2: {  # s2
        0: {1: 1.0},    # a0 → s1  1.0
        1: {2: 1.0},    # a1 → s2  1.0（）
    },
}

# ：R[s][a] = 
R = {
    0: {0: 1, 1: 2},     # s0: a0  1, a1  2
    1: {0: -1, 1: 0},    # s1: a0  -1, a1  0
    2: {0: 3, 1: 1},     # s2: a0  3, a1  1
}

GAMMA = 0.9  # 


# ==========================================
# ： —— 
# ==========================================
def manual_bellman_expectation():
    """
     V^π(s)

     π，：
        V^π(s) = Σ_a π(a|s) * [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V^π(s')]

    ：
        π(a|s) = 0.5（）

    （ V^π(s0) = v0, V^π(s1) = v1, V^π(s2) = v2）：

    V^π(s0) = 0.5 * [R(s0,a0) + γ * V^π(s1)] + 0.5 * [R(s0,a1) + γ * V^π(s2)]
            = 0.5 * [1 + 0.9 * v1] + 0.5 * [2 + 0.9 * v2]
            = 0.5 + 0.45*v1 + 1 + 0.45*v2
            = 1.5 + 0.45*v1 + 0.45*v2  ........................ (1)

    V^π(s1) = 0.5 * [R(s1,a0) + γ * (0.5*V^π(s0) + 0.5*V^π(s2))]
            + 0.5 * [R(s1,a1) + γ * (0.8*V^π(s1) + 0.2*V^π(s2))]
            = 0.5 * [-1 + 0.9*(0.5*v0 + 0.5*v2)]
            + 0.5 * [0 + 0.9*(0.8*v1 + 0.2*v2)]
            = -0.5 + 0.225*v0 + 0.225*v2 + 0.36*v1 + 0.09*v2
            = -0.5 + 0.225*v0 + 0.36*v1 + 0.315*v2  ............ (2)

    V^π(s2) = 0.5 * [R(s2,a0) + γ * V^π(s1)] + 0.5 * [R(s2,a1) + γ * V^π(s2)]
            = 0.5 * [3 + 0.9 * v1] + 0.5 * [1 + 0.9 * v2]
            = 1.5 + 0.45*v1 + 0.5 + 0.45*v2
            = 2.0 + 0.45*v1 + 0.45*v2  ........................ (3)
    """
    print("=" * 60)
    print("   —— ")
    print("=" * 60)
    print()
    print("： π(a|s) = 0.5")
    print(f"：γ = {GAMMA}")
    print()
    print("（v0 = V^π(s0), v1 = V^π(s1), v2 = V^π(s2)）：")
    print("  v0 = 1.5   + 0.45*v1 + 0.45*v2  ...... (1)")
    print("  v1 = -0.5  + 0.225*v0 + 0.36*v1 + 0.315*v2  (2)")
    print("  v2 = 2.0   + 0.45*v1 + 0.45*v2  ...... (3)")
    print()

    #  A * v = b
    # 1: v0 - 0.45*v1 - 0.45*v2 = 1.5
    # 2: -0.225*v0 + (1-0.36)*v1 - 0.315*v2 = -0.5
    # 3: -0.45*v1 + (1-0.45)*v2 = 2.0

    A = np.array([
        [1.0,   -0.45,   -0.45],
        [-0.225, 0.64,   -0.315],
        [0.0,   -0.45,    0.55],
    ])
    b = np.array([1.5, -0.5, 2.0])

    manual_V = np.linalg.solve(A, b)

    print("：")
    for i in range(N_STATES):
        print(f"  V^π(s{i}) = {manual_V[i]:.6f}")
    print()
    return manual_V


# ==========================================
# ： —— 
# ==========================================
def policy_evaluation(policy, max_iter=1000, tol=1e-8):
    """
    ：

    （）：
        V(s) ← Σ_a π(a|s) * [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')]

     V(s) ， V(s)  V^π(s)。

    ：
        policy:  π(a|s)， (N_STATES, N_ACTIONS)
        max_iter: 
        tol: 
    ：
        V: 
        history:  V （）
    """
    V = np.zeros(N_STATES)
    history = [V.copy()]

    for iteration in range(max_iter):
        V_new = np.zeros(N_STATES)

        for s in range(N_STATES):
            # ：
            for a in range(N_ACTIONS):
                # π(a|s) * [R(s,a) + γ * Σ P(s'|s,a) * V(s')]
                action_value = R[s][a]
                for next_s, prob in P[s][a].items():
                    action_value += GAMMA * prob * V[next_s]
                V_new[s] += policy[s][a] * action_value

        # 
        delta = np.max(np.abs(V_new - V))
        history.append(V_new.copy())
        V = V_new

        if delta < tol:
            break

    return V, history


# ==========================================
# ： —— 
# ==========================================
def value_iteration(max_iter=1000, tol=1e-8):
    """
    ：， V*(s)

    （）：
        V(s) ← max_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V(s')]

    ：
    - ： π， V^π(s)
    - ：， V*(s)

    V*(s) ：
        V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a) + γ * V*(s')]

    ：
        max_iter: 
        tol: 
    ：
        V_star: 
        optimal_policy: 
        history: 
    """
    V = np.zeros(N_STATES)
    history = [V.copy()]

    for iteration in range(max_iter):
        V_new = np.zeros(N_STATES)

        for s in range(N_STATES):
            #  Q(s, a)
            q_values = []
            for a in range(N_ACTIONS):
                q = R[s][a]
                for next_s, prob in P[s][a].items():
                    q += GAMMA * prob * V[next_s]
                q_values.append(q)

            # ：
            V_new[s] = max(q_values)

        delta = np.max(np.abs(V_new - V))
        history.append(V_new.copy())
        V = V_new

        if delta < tol:
            break

    #  V* 
    optimal_policy = extract_optimal_policy(V)

    return V, optimal_policy, history


def extract_optimal_policy(V):
    """
     V*  π*

    π*(s) = argmax_a [R(s,a) + γ * Σ_{s'} P(s'|s,a) * V*(s')]
    """
    policy = np.zeros((N_STATES, N_ACTIONS))

    for s in range(N_STATES):
        q_values = []
        for a in range(N_ACTIONS):
            q = R[s][a]
            for next_s, prob in P[s][a].items():
                q += GAMMA * prob * V[next_s]
            q_values.append(q)

        best_action = np.argmax(q_values)
        policy[s][best_action] = 1.0  # 

    return policy


# ==========================================
# ：
# ==========================================
def verify_results():
    """"""

    # 
    uniform_policy = np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS

    print("=" * 60)
    print("  ")
    print("=" * 60)
    print()

    # ------------------------------------------
    # 1： vs （）
    # ------------------------------------------
    print("-" * 60)
    print("  1： V^π(s) ")
    print("-" * 60)

    # 
    manual_V = manual_bellman_expectation()

    # 
    iter_V, iter_history = policy_evaluation(uniform_policy)
    print("（）：")
    for i in range(N_STATES):
        print(f"  V^π(s{i}) = {iter_V[i]:.6f}")
    print()

    # 
    print(">>> ：")
    print(f"  {'':<8s} {'':<15s} {'':<15s} {'':<15s}")
    for i in range(N_STATES):
        error = abs(manual_V[i] - iter_V[i])
        print(f"  s{i:<6d} {manual_V[i]:<15.8f} {iter_V[i]:<15.8f} {error:<15.2e}")

    all_match = np.allclose(manual_V, iter_V, atol=1e-6)
    print(f"\n  ：{' ✓' if all_match else ' ✗'}")
    print(f"  （ = ，！）")
    print()

    # ------------------------------------------
    # 2： vs 
    # ------------------------------------------
    print("-" * 60)
    print("  2：V^π(s) vs V*(s)")
    print("-" * 60)
    print()

    V_star, optimal_policy, vi_history = value_iteration()

    print(" V^π(s)（）：")
    for i in range(N_STATES):
        print(f"  V^π(s{i}) = {iter_V[i]:.6f}")

    print()
    print(" V*(s)（）：")
    for i in range(N_STATES):
        print(f"  V*(s{i}) = {V_star[i]:.6f}")

    print()
    print(" π*：")
    action_names = ['a0', 'a1']
    for s in range(N_STATES):
        best = np.argmax(optimal_policy[s])
        print(f"  π*(s{s}) = {action_names[best]}")

    print()
    print(f"  {'':<8s} {'V^π(s) ':<20s} {'V*(s) ':<20s} {'':<10s}")
    for i in range(N_STATES):
        improvement = V_star[i] - iter_V[i]
        print(f"  s{i:<6d} {iter_V[i]:<20.6f} {V_star[i]:<20.6f} {improvement:>+10.6f}")

    print()
    print("  ：V*(s) ≥ V^π(s) （）")

    # ------------------------------------------
    # ：
    # ------------------------------------------
    visualize_convergence(iter_history, vi_history)


def visualize_convergence(expectation_history, optimal_history):
    """
    

    ：（）
    ：（）
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ------------------------------------------
    # ：
    # ------------------------------------------
    ax1 = axes[0]
    history_arr = np.array(expectation_history)
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    state_labels = ['V^π(s0)', 'V^π(s1)', 'V^π(s2)']

    for s in range(N_STATES):
        ax1.plot(history_arr[:, s], color=colors[s], label=state_labels[s], linewidth=2)

    ax1.set_xlabel('', fontsize=11)
    ax1.set_ylabel(' V(s)', fontsize=11)
    ax1.set_title('\n（）', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    #  30 （）
    n_show = min(30, len(expectation_history))
    ax1.set_xlim(0, n_show)

    # ------------------------------------------
    # ：
    # ------------------------------------------
    ax2 = axes[1]
    history_arr2 = np.array(optimal_history)

    state_labels_star = ['V*(s0)', 'V*(s1)', 'V*(s2)']
    for s in range(N_STATES):
        ax2.plot(history_arr2[:, s], color=colors[s], label=state_labels_star[s], linewidth=2)

    ax2.set_xlabel('', fontsize=11)
    ax2.set_ylabel(' V(s)', fontsize=11)
    ax2.set_title('\n（）', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    n_show2 = min(30, len(optimal_history))
    ax2.set_xlim(0, n_show2)

    plt.tight_layout()
    plt.savefig('output/bellman_equation_verify_results.png', dpi=150, bbox_inches='tight')
    print("\n output/bellman_equation_verify_results.png")
    plt.show()


# ==========================================
# ：
# ==========================================
def print_value_iteration_steps(n_steps=10):
    """
     n_steps ，

    ：
    -  V(s) = 0（）
    - ，V(s) ，
    - ，V(s) 
    """
    print()
    print("=" * 60)
    print("  ")
    print("=" * 60)
    print()
    print("    |  V*(s0)    V*(s1)    V*(s2)   |  ")
    print("  " + "-" * 55)

    V = np.zeros(N_STATES)

    for iteration in range(n_steps):
        V_new = np.zeros(N_STATES)

        for s in range(N_STATES):
            q_values = []
            for a in range(N_ACTIONS):
                q = R[s][a]
                for next_s, prob in P[s][a].items():
                    q += GAMMA * prob * V[next_s]
                q_values.append(q)
            V_new[s] = max(q_values)

        delta = np.max(np.abs(V_new - V))

        print(f"  {iteration + 1:4d}  |"
              f"  {V_new[0]:>8.4f}  {V_new[1]:>8.4f}  {V_new[2]:>8.4f} |"
              f"  {delta:>8.6f}")

        V = V_new

        if delta < 1e-8:
            print(f"\n   {iteration + 1} ！")
            break

    print("  " + "-" * 55)
    print(f"\n  ：")
    for i in range(N_STATES):
        print(f"    V*(s{i}) = {V[i]:.6f}")

    # 
    print(f"\n  ：")
    action_names = ['a0', 'a1']
    for s in range(N_STATES):
        q_values = []
        for a in range(N_ACTIONS):
            q = R[s][a]
            for next_s, prob in P[s][a].items():
                q += GAMMA * prob * V[next_s]
            q_values.append(q)
        best = np.argmax(q_values)
        print(f"    π*(s{s}) = {action_names[best]}  "
              f"(Q(s{s},a0)={q_values[0]:.4f}, Q(s{s},a1)={q_values[1]:.4f})")


# ==========================================
# 
# ==========================================
def main():
    """："""

    # 1. 
    verify_results()

    # 2. 
    print_value_iteration_steps(n_steps=20)


if __name__ == "__main__":
    main()
