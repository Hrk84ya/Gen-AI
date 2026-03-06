"""
Advanced RL Topics
==================
Covers: Model-based RL (world models), reward shaping, curriculum learning,
multi-agent RL, and offline RL concepts.
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# 1. World Model (Model-Based RL)
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """
    A learned dynamics model: predicts next state and reward given (state, action).
    
    In model-based RL, we learn a model of the environment and use it
    for planning (e.g., generating imagined rollouts to train a policy
    without real environment interaction).
    
    s_{t+1}, r_t = f_θ(s_t, a_t)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action_onehot: torch.Tensor):
        x = torch.cat([state, action_onehot], dim=-1)
        next_state = state + self.state_net(x)  # residual prediction
        reward = self.reward_net(x).squeeze(-1)
        return next_state, reward


class ModelBasedAgent:
    """
    Dyna-style agent: learns a world model alongside Q-values,
    then does additional "imagined" updates using the model.
    
    Real step → update model + Q
    Imagined steps → sample from model → update Q
    """

    def __init__(self, state_dim: int, action_dim: int, n_imagined: int = 5,
                 lr_model: float = 1e-3, lr_q: float = 1e-3, gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_imagined = n_imagined
        self.gamma = gamma

        self.world_model = WorldModel(state_dim, action_dim)
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.opt_model = optim.Adam(self.world_model.parameters(), lr=lr_model)
        self.opt_q = optim.Adam(self.q_net.parameters(), lr=lr_q)
        self.buffer = deque(maxlen=50_000)
        self.epsilon = 1.0

    def act(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state))
            return int(q.argmax().item())

    def _action_onehot(self, action: int) -> torch.Tensor:
        oh = torch.zeros(self.action_dim)
        oh[action] = 1.0
        return oh

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def update(self, state, action, reward, next_state, done):
        self.store(state, action, reward, next_state, done)

        state_t = torch.FloatTensor(state)
        next_state_t = torch.FloatTensor(next_state)
        action_oh = self._action_onehot(action)

        # --- Update world model ---
        pred_next, pred_reward = self.world_model(state_t.unsqueeze(0), action_oh.unsqueeze(0))
        model_loss = F.mse_loss(pred_next.squeeze(), next_state_t) + F.mse_loss(pred_reward.squeeze(), torch.tensor(reward))
        self.opt_model.zero_grad()
        model_loss.backward()
        self.opt_model.step()

        # --- Update Q from real experience ---
        self._update_q(state_t, action, reward, next_state_t, done)

        # --- Imagined rollouts ---
        if len(self.buffer) > 100:
            for _ in range(self.n_imagined):
                idx = np.random.randint(len(self.buffer))
                s, a, _, _, _ = self.buffer[idx]
                s_t = torch.FloatTensor(s)
                a_oh = self._action_onehot(a)
                with torch.no_grad():
                    ns_t, r_t = self.world_model(s_t.unsqueeze(0), a_oh.unsqueeze(0))
                self._update_q(s_t, a, r_t.item(), ns_t.squeeze(), False)

    def _update_q(self, state_t, action, reward, next_state_t, done):
        q_values = self.q_net(state_t.unsqueeze(0))
        q_sa = q_values[0, action]
        with torch.no_grad():
            next_q = self.q_net(next_state_t.unsqueeze(0)).max().item() if not done else 0.0
        target = reward + self.gamma * next_q
        loss = F.mse_loss(q_sa, torch.tensor(target))
        self.opt_q.zero_grad()
        loss.backward()
        self.opt_q.step()


# ---------------------------------------------------------------------------
# 2. Reward Shaping
# ---------------------------------------------------------------------------

class PotentialBasedRewardShaper:
    """
    Potential-based reward shaping (Ng et al., 1999).
    
    Adds a shaping reward F(s, s') = γΦ(s') - Φ(s) to the environment reward.
    This preserves the optimal policy while accelerating learning.
    
    Φ(s) is a potential function — domain knowledge encoded as a hint.
    """

    def __init__(self, potential_fn, gamma: float = 0.99):
        self.potential_fn = potential_fn
        self.gamma = gamma

    def shape(self, state, next_state, reward: float) -> float:
        shaping = self.gamma * self.potential_fn(next_state) - self.potential_fn(state)
        return reward + shaping


class CurriculumScheduler:
    """
    Simple curriculum learning scheduler for RL.
    
    Starts the agent on easier tasks and gradually increases difficulty
    as performance improves.
    """

    def __init__(self, difficulty_levels: List[Dict], performance_threshold: float = 0.8):
        self.levels = difficulty_levels
        self.current_level = 0
        self.threshold = performance_threshold
        self.performance_history = deque(maxlen=50)

    @property
    def current_config(self) -> Dict:
        return self.levels[min(self.current_level, len(self.levels) - 1)]

    def report_performance(self, score: float):
        self.performance_history.append(score)
        if len(self.performance_history) >= 50:
            avg = np.mean(self.performance_history)
            if avg >= self.threshold and self.current_level < len(self.levels) - 1:
                self.current_level += 1
                self.performance_history.clear()
                print(f"Curriculum: advancing to level {self.current_level} "
                      f"({self.current_config})")


# ---------------------------------------------------------------------------
# 3. Multi-Agent RL — Independent Learners
# ---------------------------------------------------------------------------

class IndependentQLearner:
    """
    Independent Q-Learning for multi-agent settings.
    
    Each agent maintains its own Q-table and treats other agents
    as part of the environment. Simple but surprisingly effective
    in many cooperative/competitive settings.
    """

    def __init__(self, agent_id: int, n_actions: int, lr: float = 0.1, gamma: float = 0.99):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.q_table: Dict = {}
        self.epsilon = 1.0

    def _get_q(self, state) -> np.ndarray:
        key = str(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions)
        return self.q_table[key]

    def act(self, state) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._get_q(state)))

    def update(self, state, action, reward, next_state, done):
        q = self._get_q(state)
        next_q_max = np.max(self._get_q(next_state)) if not done else 0.0
        q[action] += self.lr * (reward + self.gamma * next_q_max - q[action])


class PredatorPreyEnv:
    """
    Simple 2-agent predator-prey grid world.
    
    - Predator tries to catch the prey (reward +10 for predator, -10 for prey)
    - Prey tries to escape (reward +1 per step survived)
    - Both agents move simultaneously on a 7x7 grid
    """

    def __init__(self, size: int = 7, max_steps: int = 50):
        self.size = size
        self.max_steps = max_steps
        self.actions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (0, 0)}
        self.reset()

    def reset(self):
        self.predator = (0, 0)
        self.prey = (self.size - 1, self.size - 1)
        self.steps = 0
        return self._obs()

    def _obs(self):
        return {"predator": self.predator, "prey": self.prey}

    def _move(self, pos, action):
        dr, dc = self.actions[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        nr = max(0, min(self.size - 1, nr))
        nc = max(0, min(self.size - 1, nc))
        return (nr, nc)

    def step(self, predator_action: int, prey_action: int):
        self.predator = self._move(self.predator, predator_action)
        self.prey = self._move(self.prey, prey_action)
        self.steps += 1

        caught = self.predator == self.prey
        timeout = self.steps >= self.max_steps
        done = caught or timeout

        rewards = {
            "predator": 10.0 if caught else -0.1,
            "prey": -10.0 if caught else 1.0,
        }
        return self._obs(), rewards, done


def train_predator_prey(episodes: int = 1000):
    """Train two independent Q-learners in predator-prey."""
    env = PredatorPreyEnv()
    predator = IndependentQLearner(0, n_actions=5)
    prey = IndependentQLearner(1, n_actions=5)

    catch_rate = deque(maxlen=100)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            pa = predator.act(obs["predator"])
            ya = prey.act(obs["prey"])
            next_obs, rewards, done = env.step(pa, ya)
            predator.update(obs["predator"], pa, rewards["predator"], next_obs["predator"], done)
            prey.update(obs["prey"], ya, rewards["prey"], next_obs["prey"], done)
            obs = next_obs

        caught = obs["predator"] == obs["prey"]
        catch_rate.append(float(caught))

        # Decay exploration
        predator.epsilon = max(0.05, predator.epsilon * 0.998)
        prey.epsilon = max(0.05, prey.epsilon * 0.998)

        if (ep + 1) % 200 == 0:
            print(f"Episode {ep+1} | Catch rate (100): {np.mean(catch_rate):.2%}")


# ---------------------------------------------------------------------------
# 4. Offline RL — Conservative Q-Learning (CQL) Concept
# ---------------------------------------------------------------------------

class ConservativeQLoss:
    """
    Conservative Q-Learning (CQL) loss component.
    
    Offline RL learns from a fixed dataset without environment interaction.
    The key challenge is overestimation of Q-values for out-of-distribution
    actions. CQL adds a regularizer that penalizes high Q-values for
    actions not in the dataset.
    
    L_CQL = α * (E_{a~π}[Q(s,a)] - E_{a~D}[Q(s,a)]) + standard_td_loss
    
    This pushes down Q-values for unseen actions and pushes up Q-values
    for actions in the dataset.
    """

    def __init__(self, alpha: float = 1.0, n_actions: int = 4):
        self.alpha = alpha
        self.n_actions = n_actions

    def compute(self, q_net: nn.Module, states: torch.Tensor,
                dataset_actions: torch.Tensor, td_loss: torch.Tensor) -> torch.Tensor:
        # Q-values for all actions (policy distribution)
        all_q = q_net(states)  # (batch, n_actions)
        logsumexp_q = torch.logsumexp(all_q, dim=1).mean()

        # Q-values for dataset actions
        dataset_q = all_q.gather(1, dataset_actions.unsqueeze(1)).squeeze(1).mean()

        # CQL regularizer: penalize high Q for non-dataset actions
        cql_loss = self.alpha * (logsumexp_q - dataset_q)

        return td_loss + cql_loss


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Multi-Agent: Predator-Prey ===")
    train_predator_prey(episodes=500)

    print("\n=== Reward Shaping Example ===")
    # Example: grid world where potential = negative Manhattan distance to goal
    goal = (4, 4)
    shaper = PotentialBasedRewardShaper(
        potential_fn=lambda s: -(abs(s[0] - goal[0]) + abs(s[1] - goal[1])),
        gamma=0.99,
    )
    shaped_r = shaper.shape(state=(0, 0), next_state=(0, 1), reward=-1.0)
    print(f"Original reward: -1.0, Shaped reward: {shaped_r:.2f}")

    print("\n=== Curriculum Learning Example ===")
    curriculum = CurriculumScheduler(
        difficulty_levels=[
            {"grid_size": 3, "n_walls": 0},
            {"grid_size": 5, "n_walls": 2},
            {"grid_size": 7, "n_walls": 4},
            {"grid_size": 10, "n_walls": 8},
        ],
        performance_threshold=0.8,
    )
    for i in range(200):
        curriculum.report_performance(np.random.random() * (0.5 + i / 200))
