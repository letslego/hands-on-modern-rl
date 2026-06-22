# 6.4 ：AlphaGo 

 Actor-Critic ，：****（——：[ $\pi_\theta(a|s)$](../chapter05_policy_gradient/reinforce)）****（——：[Critic $V(s)$](./critic-training)）。2016 ，DeepMind  AlphaGo （MCTS），—— RL 。

 AlphaGo ： 6×6  AI。

::: tip  6×6？
 19×19， $2 \times 10^{170}$，。6×6 ，"、、"。AlphaGo ——、、MCTS—— 6×6 。
:::

## AlphaGo 

AlphaGo ：

|            |                        |                                               |
| -------------- | -------------------------- | --------------------------------------------------------- |
|        |  | [ 5 ](../chapter05_policy_gradient/reinforce) |
|        |          | [ 6.2  Critic ](./critic-training)                |
|  | ， |                                               |

：MCTS ""，""（），""（）。

## 6×6 

，：、、（）。

```python
import numpy as np

BOARD_SIZE = 6
EMPTY, BLACK, WHITE = 0, 1, -1

class MiniGo:
    """6×6 """

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = BLACK
        self.ko_point = None  # 
        self.passes = 0       #  pass 
        self.history = []     # 

    def copy(self):
        env = MiniGo()
        env.board = self.board.copy()
        env.current_player = self.current_player
        env.ko_point = self.ko_point
        env.passes = self.passes
        env.history = list(self.history)
        return env

    def get_opponent(self, player):
        return -player

    def on_board(self, r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def get_neighbors(self, r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if self.on_board(nr, nc):
                yield nr, nc

    def get_group(self, r, c):
        """ (r,c) """
        color = self.board[r, c]
        if color == EMPTY:
            return set(), 0
        visited = set()
        liberties = set()
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            for nr, nc in self.get_neighbors(cr, cc):
                if self.board[nr, nc] == EMPTY:
                    liberties.add((nr, nc))
                elif self.board[nr, nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
        return visited, len(liberties)

    def remove_group(self, group):
        for r, c in group:
            self.board[r, c] = EMPTY

    def play(self, r, c):
        """ (r,c) ，"""
        if not self.on_board(r, c) or self.board[r, c] != EMPTY:
            return False
        if (r, c) == self.ko_point:
            return False

        self.board[r, c] = self.current_player
        opponent = self.get_opponent(self.current_player)
        captured = []

        for nr, nc in self.get_neighbors(r, c):
            if self.board[nr, nc] == opponent:
                group, liberties = self.get_group(nr, nc)
                if liberties == 0:
                    captured.extend(group)
                    self.remove_group(group)

        # ：
        if len(captured) == 1:
            _, my_liberties = self.get_group(r, c)
            if my_liberties == 1:
                self.ko_point = captured[0]
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # 
        _, my_liberties = self.get_group(r, c)
        if my_liberties == 0:
            self.board[r, c] = EMPTY
            return False

        self.passes = 0
        self.current_player = opponent
        return True

    def pass_turn(self):
        """"""
        self.ko_point = None
        self.passes += 1
        self.current_player = self.get_opponent(self.current_player)

    def is_game_over(self):
        return self.passes >= 2

    def get_legal_moves(self):
        """"""
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c] == EMPTY and (r, c) != self.ko_point:
                    # 
                    env_copy = self.copy()
                    if env_copy.play(r, c):
                        moves.append((r, c))
        return moves

    def compute_score(self):
        """： + """
        score = {BLACK: 0, WHITE: 0}
        visited = set()

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c] != EMPTY:
                    score[self.board[r, c]] += 1
                elif (r, c) not in visited:
                    # BFS 
                    region = set()
                    borders = set()
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) in region:
                            continue
                        if self.board[cr, cc] == EMPTY:
                            region.add((cr, cc))
                            visited.add((cr, cc))
                            for nr, nc in self.get_neighbors(cr, cc):
                                if self.board[nr, nc] == EMPTY:
                                    stack.append((nr, nc))
                                else:
                                    borders.add(self.board[nr, nc])
                    # ，
                    if len(borders) == 1:
                        score[list(borders)[0]] += len(region)

        #  3.75（6×6  3.75 ）
        score[WHITE] += 3.75
        return score

    def get_winner(self):
        """：BLACK  WHITE"""
        score = self.compute_score()
        return BLACK if score[BLACK] > score[WHITE] else WHITE
```

，：、、、、。

## 

AlphaGo ，，：

- ****：（Actor）
- ****：，（Critic）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """：Conv3x3 + BatchNorm + ReLU"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class AlphaGoNet(nn.Module):
    """AlphaGo """

    def __init__(self, board_size=BOARD_SIZE, num_blocks=4, channels=64):
        super().__init__()
        self.board_size = board_size

        # ：2 （、）
        self.input_conv = nn.Conv2d(2, channels, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(channels)

        # 
        self.blocks = nn.ModuleList([ConvBlock(channels) for _ in range(num_blocks)])

        # ： board_size × board_size  logits
        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size,
                                   board_size * board_size)

        # ：
        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board, current_player):
        """
        Args:
            board: (B, board_size, board_size) 
            current_player: (B,)  (1=, -1=)
        Returns:
            policy_logits: (B, board_size * board_size)
            value: (B, 1)  [-1, 1]
        """
        # ：2 （、）
        player_mask = current_player.view(-1, 1, 1).unsqueeze(1)  # (B,1,1,1)
        own = (board.unsqueeze(1) == player_mask).float()         # (B,1,H,W)
        opp = (board.unsqueeze(1) == -player_mask).float()        # (B,1,H,W)
        x = torch.cat([own, opp], dim=1)                          # (B,2,H,W)

        # 
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            x = x + block(x)  # 

        # 
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # 
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value
```

[ 6.3  Actor-Critic](./actor-critic) ——，，。

## 

MCTS  AlphaGo ""。，。：

1. **（Select）**：， UCB ""
2. **（Expand）**：，
3. **（Evaluate）**：（）
4. **（Backpropagate）**：

```python
import math

class MCTSNode:
    """MCTS """

    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {}       # action -> MCTSNode
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior       # 

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct=1.5):
        """PUCT ：Q + U（ bonus）"""
        if self.visit_count == 0:
            return float('inf')
        u = c_puct * self.prior * math.sqrt(self.parent.visit_count) \
            / (1 + self.visit_count)
        return self.q_value + u

    def select_child(self):
        """ UCB """
        return max(self.children.items(),
                   key=lambda item: item[1].ucb_score())

    def expand(self, action_priors):
        """"""
        for action, prior in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior=prior)

    def backpropagate(self, value):
        """（）"""
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            # ，
            self.parent.backpropagate(-value)


class MCTS:
    """"""

    def __init__(self, model, c_puct=1.5, num_simulations=100):
        self.model = model
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def run(self, env):
        """ MCTS，"""
        root = MCTSNode()

        # 
        for _ in range(self.num_simulations):
            node = root
            sim_env = env.copy()

            # 1. ：
            while node.children:
                action, node = node.select_child()
                sim_env.play(*action)

            # 2. ：
            board_tensor = torch.tensor(sim_env.board, dtype=torch.float32).unsqueeze(0)
            player_tensor = torch.tensor([sim_env.current_player], dtype=torch.float32)

            with torch.no_grad():
                policy_logits, value = self.model(board_tensor, player_tensor)

            # 3. ：
            legal_moves = sim_env.get_legal_moves()
            if legal_moves:
                #  logits  -inf
                mask = torch.full((BOARD_SIZE * BOARD_SIZE,), float('-inf'))
                for r, c in legal_moves:
                    mask[r * BOARD_SIZE + c] = policy_logits[0, r * BOARD_SIZE + c]
                probs = torch.softmax(mask, dim=0)

                action_priors = [
                    ((r, c), probs[r * BOARD_SIZE + c].item())
                    for r, c in legal_moves
                ]
                node.expand(action_priors)
            else:
                # ，pass
                pass

            # 4. 
            node.backpropagate(value.item())

        # 
        visit_counts = {}
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        return visit_counts
```

 `backpropagate`  `-value`——：，。。

## 

AlphaGo **（Self-Play）**： AI ，。，。——""""。

```python
def self_play_game(model, mcts, temperature=1.0):
    """ MCTS ， (states, policies, winner)"""
    env = MiniGo()
    states, players_list, policies = [], [], []
    max_moves = BOARD_SIZE * BOARD_SIZE * 2  # 

    for _ in range(max_moves):
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            env.pass_turn()
            if env.is_game_over():
                break
            continue

        # MCTS 
        visit_counts = mcts.run(env)
        total_visits = sum(visit_counts.values())

        # （）
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for (r, c), visits in visit_counts.items():
            policy[r * BOARD_SIZE + c] = visits / total_visits

        # 
        if temperature > 0:
            noisy_policy = policy ** (1.0 / temperature)
            noisy_policy /= noisy_policy.sum() + 1e-8
            action_idx = np.random.choice(len(policy), p=noisy_policy)
        else:
            action_idx = policy.argmax()

        r, c = divmod(action_idx, BOARD_SIZE)

        states.append(env.board.copy())
        players_list.append(env.current_player)
        policies.append(policy)

        env.play(r, c)
        if env.is_game_over():
            break

    # 
    winner = env.get_winner()

    #  +1/-1 
    values = []
    for player in players_list:
        values.append(1.0 if player == winner else -1.0)

    return states, players_list, policies, values


def train_alphago(num_iterations=20, games_per_iter=10, num_epochs=5):
    """AlphaGo """
    model = AlphaGoNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mcts = MCTS(model, num_simulations=50)  # 

    replay_buffer = []  # (state, player, policy, value)

    for iteration in range(num_iterations):
        #  1：
        new_data = []
        for _ in range(games_per_iter):
            # ，
            temp = 1.0 if iteration < num_iterations // 2 else 0.5
            states, players, policies, values = self_play_game(model, mcts, temp)
            for s, p, pi, v in zip(states, players, policies, values):
                new_data.append((s, p, pi, v))

        replay_buffer.extend(new_data)
        #  5000 
        if len(replay_buffer) > 5000:
            replay_buffer = replay_buffer[-5000:]

        #  2：
        model.train()
        for epoch in range(num_epochs):
            #  mini-batch
            indices = np.random.choice(len(replay_buffer),
                                       size=min(64, len(replay_buffer)),
                                       replace=False)

            boards = torch.stack([
                torch.tensor(replay_buffer[i][0], dtype=torch.float32)
                for i in indices
            ])
            players = torch.tensor(
                [replay_buffer[i][1] for i in indices], dtype=torch.float32
            )
            target_policies = torch.stack([
                torch.tensor(replay_buffer[i][2], dtype=torch.float32)
                for i in indices
            ])
            target_values = torch.tensor(
                [replay_buffer[i][3] for i in indices], dtype=torch.float32
            ).unsqueeze(1)

            # 
            policy_logits, pred_values = model(boards, players)

            # ：（MCTS ）
            policy_loss = F.cross_entropy(policy_logits, target_policies)

            # ：
            value_loss = F.mse_loss(pred_values, target_values)

            # 
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (iteration + 1) % 5 == 0:
            print(f"Iteration {iteration+1}/{num_iterations} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Buffer: {len(replay_buffer)}")

    return model
```

：

1. ****： + MCTS ， (, MCTS, ) 
2. ****： MCTS ，

 5 [](../chapter05_policy_gradient/reinforce)：AlphaGo （ REINFORCE），** MCTS **。MCTS ——。

## 

```python
def human_vs_ai(model, mcts, human_color=BLACK):
    """ vs AI """
    env = MiniGo()
    print(f"{'(X)' if human_color == BLACK else '(O)'}")
    print("： （ '2 3'）， 'pass' \n")

    while not env.is_game_over():
        print(env_to_string(env.board))

        if env.current_player == human_color:
            # 
            legal = env.get_legal_moves()
            print(f": {legal}")
            cmd = input(": ").strip()
            if cmd == 'pass':
                env.pass_turn()
            else:
                r, c = map(int, cmd.split())
                if not env.play(r, c):
                    print("，")
                    continue
        else:
            # AI 
            visit_counts = mcts.run(env)
            if visit_counts:
                best_action = max(visit_counts, key=visit_counts.get)
                print(f"AI : {best_action} "
                      f"(: {visit_counts[best_action]})")
                env.play(*best_action)
            else:
                print("AI: pass")
                env.pass_turn()
        print()

    # 
    score = env.compute_score()
    print(env_to_string(env.board))
    print(f": {score[BLACK]:.1f}  | : {score[WHITE]:.1f} ")
    winner = "" if score[BLACK] > score[WHITE] else ""
    print(f"{winner} ！")


def env_to_string(board):
    symbols = {EMPTY: '·', BLACK: 'X', WHITE: 'O'}
    lines = ["   " + " ".join(str(i) for i in range(BOARD_SIZE))]
    for r in range(BOARD_SIZE):
        line = f"{r}: " + " ".join(symbols[board[r, c]] for c in range(BOARD_SIZE))
        lines.append(line)
    return "\n".join(lines)
```

## AlphaGo 

 AlphaGo ：

| AlphaGo   |                  |                                                    |
| ------------- | ------------------------ | ------------------------------------------------------ |
|       | Actor，      | [](../chapter05_policy_gradient/reinforce) |
|       | Critic，     | [Actor-Critic ](./actor-critic)                    |
| MCTS  | "" | [](../chapter05_policy_gradient/pg-improvements)   |
|       |  +       | REINFORCE                                    |
| $-v$      |          | [](./advantage-function)             |

：AlphaGo  Actor-Critic + MCTS 。（Actor），（Critic），MCTS 。"Actor  + Critic  + "， AlphaZero ， RL 。

## 

。，：

### 

|                                                                   |                                                  |                              |
| --------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------ |
| [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) | PyTorch，， Othello/Gomoku/TicTacToe | ****，， |
| [michaelnny/alpha_zero](https://github.com/michaelnny/alpha_zero)     | PyTorch，9×9  + 15×15                      |  9×9           |
| [KataGo](https://github.com/lightvector/KataGo)                       | C++/Python， 7×7  19×19，          | ****       |
| [Leela Zero](https://github.com/leela-zero/leela-zero)                | C++，AlphaGo Zero                          |  AlphaGo Zero          |
| [MiniZero](https://github.com/rlglab/minizero)                        | C++/Python， AlphaZero/MuZero/Gumbel         |  MCTS                |

****： [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)  Othello（，）， pipeline ，。

### 

|                                                             |               |                                                             |
| ----------------------------------------------------------------- | ----------------- | --------------------------------------------------------------- |
| [JGDB](https://pjreddie.com/projects/jgdb/)                       | 53.5 ，194 MB | ****， train/val/test，， YOLO  |
| [featurecat/go-dataset](https://github.com/featurecat/go-dataset) | 2110          | ， Fox ， 18k  9p                         |
| [CWI ](https://homepages.cwi.nl/~aeb/go/games/games/) | 8.8 ，45 MB   | ，                                          |
| [KGS ](https://www.gokgs.com/)                            |         | KGS ，                                  |

 JGDB ： SGF →  (, )  →  → 。 AlphaGo 。

## 

1. ****： [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)  9×9 ，。
2. **AlphaZero **：，。 6×6 ？
3. **MCTS **： 10 、50 、200 。？
4. ** JGDB **： [JGDB ](https://pjreddie.com/projects/jgdb/)， SGF ，""，—— AlphaGo pipeline。

## 

[^1]: Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. _Nature_, 529(7587), 484-489. [DOI](https://doi.org/10.1038/nature16961)

[^2]: Silver, D., et al. (2017). Mastering the game of Go without human knowledge. _Nature_, 550(7676), 354-359. [DOI](https://doi.org/10.1038/nature24270)

[^3]: Silver, D., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. _Science_, 362(6419), 1140-1144. [DOI](https://doi.org/10.1126/science.aar6404)
