"""
Deep RL — Value-Based Methods
==============================
Covers: DQN, Double DQN, Dueling DQN, Prioritized Experience Replay.
Uses PyTorch. Designed to work with Gymnasium (CartPole).
"""

import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# ---------------------------------------------------------------------------
# 1. Replay Buffers
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Standard uniform experience replay."""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Simplified proportional prioritized experience replay.
    
    Transitions with higher TD-error are sampled more frequently,
    so the agent focuses on surprising experiences.
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 1e-4):
        self.capacity = capacity
        self.alpha = alpha  # prioritization exponent
        self.beta = beta    # importance-sampling exponent
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, *args):
        max_priority = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        n = len(self.buffer)
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 2. Network Architectures
# ---------------------------------------------------------------------------

class DQNNet(nn.Module):
    """Standard DQN: state → Q-values for each action."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingDQNNet(nn.Module):
    """
    Dueling DQN: separates state-value V(s) and advantage A(s,a).
    Q(s,a) = V(s) + A(s,a) - mean(A)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Subtract mean advantage for identifiability
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# 3. DQN Agent (supports Double DQN + Dueling + PER)
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Configurable DQN agent.
    
    Args:
        state_dim: Observation space dimension
        action_dim: Number of discrete actions
        double: Use Double DQN (decouple selection and evaluation)
        dueling: Use Dueling architecture
        prioritized: Use Prioritized Experience Replay
    """

    def __init__(self, state_dim: int, action_dim: int, *,
                 double: bool = True, dueling: bool = False, prioritized: bool = False,
                 lr: float = 1e-3, gamma: float = 0.99, batch_size: int = 64,
                 target_update_freq: int = 100, buffer_size: int = 100_000):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double = double
        self.prioritized = prioritized
        self.step_count = 0

        NetClass = DuelingDQNNet if dueling else DQNNet
        self.policy_net = NetClass(state_dim, action_dim)
        self.target_net = NetClass(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        if prioritized:
            self.buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.buffer = ReplayBuffer(buffer_size)

        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        if self.prioritized:
            transitions, indices, is_weights = self.buffer.sample(self.batch_size)
        else:
            transitions = self.buffer.sample(self.batch_size)
            is_weights = torch.ones(self.batch_size)

        batch = Transition(*zip(*transitions))
        states = torch.FloatTensor(np.array(batch.state))
        actions = torch.LongTensor(batch.action).unsqueeze(1)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones = torch.FloatTensor(batch.done)

        # Current Q-values
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if self.double:
                # Double DQN: select action with policy net, evaluate with target net
                best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        # Weighted loss (weights matter for PER, uniform otherwise)
        td_errors = target - q_values
        loss = (is_weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        if self.prioritized:
            self.buffer.update_priorities(indices, td_errors.detach().numpy())

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------------------------

def train_dqn(env_name: str = "CartPole-v1", episodes: int = 300, **agent_kwargs):
    """
    Train a DQN agent on a Gymnasium environment.
    
    Usage:
        train_dqn()                                    # Standard DQN
        train_dqn(double=True, dueling=True)           # Double Dueling DQN
        train_dqn(double=True, prioritized=True)       # Double DQN + PER
    """
    try:
        import gymnasium as gym
    except ImportError:
        print("Install gymnasium: pip install gymnasium")
        return

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, **agent_kwargs)
    reward_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(500):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(state, action, reward, next_state, float(done))
            agent.learn()
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        reward_history.append(total_reward)

        if (ep + 1) % 50 == 0:
            avg = np.mean(reward_history[-50:])
            print(f"Episode {ep+1}/{episodes} | Avg Reward (50): {avg:.1f} | ε: {agent.epsilon:.3f}")

    env.close()
    return reward_history


if __name__ == "__main__":
    print("=== Double DQN ===")
    train_dqn(double=True, dueling=False, episodes=200)
    print("\n=== Double Dueling DQN ===")
    train_dqn(double=True, dueling=True, episodes=200)
