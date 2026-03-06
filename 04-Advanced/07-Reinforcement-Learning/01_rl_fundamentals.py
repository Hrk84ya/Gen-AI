"""
Reinforcement Learning Fundamentals — Tabular Methods
=====================================================
Covers: MDPs, Q-Learning, SARSA, Monte Carlo, exploration strategies.
All implementations are self-contained using only NumPy.
"""

import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, List


# ---------------------------------------------------------------------------
# 1. Simple Grid World Environment
# ---------------------------------------------------------------------------

class GridWorld:
    """
    A simple 5x5 grid world MDP.
    
    Layout:
        S . . . .
        . # . # .
        . . . . .
        . # . # .
        . . . . G
    
    S = start (0,0), G = goal (4,4), # = walls
    Actions: 0=up, 1=right, 2=down, 3=left
    Reward: -1 per step, +10 at goal, -5 for hitting a wall
    """

    ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}

    def __init__(self, size: int = 5):
        self.size = size
        self.walls = {(1, 1), (1, 3), (3, 1), (3, 3)}
        self.goal = (size - 1, size - 1)
        self.state = (0, 0)

    def reset(self) -> Tuple[int, int]:
        self.state = (0, 0)
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        dr, dc = self.ACTIONS[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc

        # Check boundaries and walls
        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.walls:
            self.state = (nr, nc)
        else:
            return self.state, -5.0, False  # penalty for invalid move

        if self.state == self.goal:
            return self.state, 10.0, True

        return self.state, -1.0, False

    @property
    def n_states(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4


# ---------------------------------------------------------------------------
# 2. Exploration Strategies
# ---------------------------------------------------------------------------

class EpsilonGreedy:
    """ε-greedy exploration with optional decay."""

    def __init__(self, epsilon: float = 1.0, min_epsilon: float = 0.01, decay: float = 0.995):
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def select_action(self, q_values: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(len(q_values))
        return int(np.argmax(q_values))

    def step(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)


class BoltzmannExploration:
    """Softmax / Boltzmann exploration."""

    def __init__(self, temperature: float = 1.0, min_temp: float = 0.1, decay: float = 0.995):
        self.temperature = temperature
        self.min_temp = min_temp
        self.decay = decay

    def select_action(self, q_values: np.ndarray) -> int:
        scaled = (q_values - q_values.max()) / self.temperature
        probs = np.exp(scaled) / np.exp(scaled).sum()
        return int(np.random.choice(len(q_values), p=probs))

    def step(self):
        self.temperature = max(self.min_temp, self.temperature * self.decay)


class UCBExploration:
    """Upper Confidence Bound exploration."""

    def __init__(self, c: float = 2.0):
        self.c = c
        self.counts = None

    def select_action(self, q_values: np.ndarray, total_steps: int) -> int:
        if self.counts is None:
            self.counts = np.zeros(len(q_values))
        # Select unvisited actions first
        unvisited = np.where(self.counts == 0)[0]
        if len(unvisited) > 0:
            action = int(np.random.choice(unvisited))
        else:
            ucb = q_values + self.c * np.sqrt(np.log(total_steps + 1) / self.counts)
            action = int(np.argmax(ucb))
        self.counts[action] += 1
        return action


# ---------------------------------------------------------------------------
# 3. Q-Learning (Off-Policy TD Control)
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-Learning.
    
    Update rule:
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(self, n_actions: int, lr: float = 0.1, gamma: float = 0.99):
        self.q_table: Dict = defaultdict(lambda: np.zeros(n_actions))
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.explorer = EpsilonGreedy()

    def act(self, state) -> int:
        return self.explorer.select_action(self.q_table[state])

    def update(self, state, action: int, reward: float, next_state, done: bool):
        best_next = np.max(self.q_table[next_state]) if not done else 0.0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def train(self, env: GridWorld, episodes: int = 500) -> List[float]:
        rewards_per_episode = []
        for ep in range(episodes):
            state = env.reset()
            total_reward = 0.0
            for _ in range(200):
                action = self.act(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            self.explorer.step()
            rewards_per_episode.append(total_reward)
        return rewards_per_episode


# ---------------------------------------------------------------------------
# 4. SARSA (On-Policy TD Control)
# ---------------------------------------------------------------------------

class SARSAAgent:
    """
    Tabular SARSA.
    
    Update rule:
        Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]
    
    Key difference from Q-Learning: uses the *actual* next action a'
    chosen by the policy, not the greedy max.
    """

    def __init__(self, n_actions: int, lr: float = 0.1, gamma: float = 0.99):
        self.q_table: Dict = defaultdict(lambda: np.zeros(n_actions))
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.explorer = EpsilonGreedy()

    def act(self, state) -> int:
        return self.explorer.select_action(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action, done):
        next_q = self.q_table[next_state][next_action] if not done else 0.0
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def train(self, env: GridWorld, episodes: int = 500) -> List[float]:
        rewards_per_episode = []
        for ep in range(episodes):
            state = env.reset()
            action = self.act(state)
            total_reward = 0.0
            for _ in range(200):
                next_state, reward, done = env.step(action)
                next_action = self.act(next_state)
                self.update(state, action, reward, next_state, next_action, done)
                state, action = next_state, next_action
                total_reward += reward
                if done:
                    break
            self.explorer.step()
            rewards_per_episode.append(total_reward)
        return rewards_per_episode


# ---------------------------------------------------------------------------
# 5. Monte Carlo Control (First-Visit)
# ---------------------------------------------------------------------------

class MonteCarloAgent:
    """
    First-visit Monte Carlo control with ε-greedy policy.
    
    Collects full episodes, then updates Q-values using the
    actual discounted return G_t from each first visit to (s, a).
    """

    def __init__(self, n_actions: int, gamma: float = 0.99):
        self.q_table: Dict = defaultdict(lambda: np.zeros(n_actions))
        self.returns: Dict = defaultdict(list)
        self.n_actions = n_actions
        self.gamma = gamma
        self.explorer = EpsilonGreedy()

    def act(self, state) -> int:
        return self.explorer.select_action(self.q_table[state])

    def train(self, env: GridWorld, episodes: int = 500) -> List[float]:
        rewards_per_episode = []
        for ep in range(episodes):
            # Generate episode
            episode = []
            state = env.reset()
            for _ in range(200):
                action = self.act(state)
                next_state, reward, done = env.step(action)
                episode.append((state, action, reward))
                state = next_state
                if done:
                    break

            # Compute returns and update Q-values (first-visit)
            G = 0.0
            visited = set()
            for state, action, reward in reversed(episode):
                G = reward + self.gamma * G
                if (state, action) not in visited:
                    visited.add((state, action))
                    self.returns[(state, action)].append(G)
                    self.q_table[state][action] = np.mean(self.returns[(state, action)])

            self.explorer.step()
            rewards_per_episode.append(sum(r for _, _, r in episode))
        return rewards_per_episode


# ---------------------------------------------------------------------------
# 6. Demo / Comparison
# ---------------------------------------------------------------------------

def compare_agents(episodes: int = 500):
    """Train Q-Learning, SARSA, and Monte Carlo on GridWorld and compare."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = GridWorld()
    agents = {
        "Q-Learning": QLearningAgent(env.n_actions),
        "SARSA": SARSAAgent(env.n_actions),
        "Monte Carlo": MonteCarloAgent(env.n_actions),
    }

    results = {}
    for name, agent in agents.items():
        env_copy = GridWorld()
        rewards = agent.train(env_copy, episodes=episodes)
        # Smooth with rolling average
        window = 20
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        results[name] = smoothed
        print(f"{name}: avg reward (last 50 eps) = {np.mean(rewards[-50:]):.2f}")

    plt.figure(figsize=(10, 5))
    for name, smoothed in results.items():
        plt.plot(smoothed, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Reward (smoothed)")
    plt.title("Tabular RL Methods on GridWorld")
    plt.legend()
    plt.tight_layout()
    plt.savefig("04-Advanced/07-Reinforcement-Learning/tabular_rl_comparison.png", dpi=100)
    plt.close()
    print("Saved comparison plot.")


if __name__ == "__main__":
    compare_agents()
