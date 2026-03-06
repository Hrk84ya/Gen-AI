"""
Deep RL — Policy Gradient Methods
===================================
Covers: REINFORCE with baseline, A2C, PPO, GAE.
Uses PyTorch + Gymnasium.
"""

import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# 1. Shared Network Components
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    """Simple policy network: state → action probabilities."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> Categorical:
        logits = self.net(x)
        return Categorical(logits=logits)


class ValueNet(nn.Module):
    """State-value network: state → V(s)."""

    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ActorCriticNet(nn.Module):
    """Shared-backbone actor-critic network."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        features = self.shared(x)
        dist = Categorical(logits=self.policy_head(features))
        value = self.value_head(features).squeeze(-1)
        return dist, value


# ---------------------------------------------------------------------------
# 2. REINFORCE with Baseline
# ---------------------------------------------------------------------------

class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo policy gradient) with a learned value baseline.
    
    Policy gradient theorem:
        ∇J(θ) = E[ Σ_t ∇log π(a_t|s_t) * (G_t - V(s_t)) ]
    
    The baseline V(s_t) reduces variance without introducing bias.
    """

    def __init__(self, state_dim: int, action_dim: int, lr_policy: float = 1e-3,
                 lr_value: float = 1e-3, gamma: float = 0.99):
        self.gamma = gamma
        self.policy = PolicyNet(state_dim, action_dim)
        self.value = ValueNet(state_dim)
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.opt_value = optim.Adam(self.value.parameters(), lr=lr_value)

    def act(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        dist = self.policy(state_t)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, log_probs: List[torch.Tensor], rewards: List[float],
               states: List[np.ndarray]):
        # Compute discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        states_t = torch.FloatTensor(np.array(states))
        values = self.value(states_t).detach()

        # Advantage = return - baseline
        advantages = returns - values

        # Policy loss
        policy_loss = -torch.stack(log_probs) * advantages
        policy_loss = policy_loss.mean()

        self.opt_policy.zero_grad()
        policy_loss.backward()
        self.opt_policy.step()

        # Value loss
        values = self.value(states_t)
        value_loss = F.mse_loss(values, returns)

        self.opt_value.zero_grad()
        value_loss.backward()
        self.opt_value.step()

        return policy_loss.item(), value_loss.item()


# ---------------------------------------------------------------------------
# 3. Advantage Actor-Critic (A2C)
# ---------------------------------------------------------------------------

class A2CAgent:
    """
    Synchronous Advantage Actor-Critic.
    
    Unlike REINFORCE, A2C updates at every step using the TD advantage:
        A(s,a) = r + γV(s') - V(s)
    
    This gives lower variance than Monte Carlo returns.
    """

    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, entropy_coef: float = 0.01, value_coef: float = 0.5):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.net = ActorCriticNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def act(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        dist, value = self.net(state_t)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def update(self, log_probs, values, rewards, dones, next_value):
        """Update using collected trajectory."""
        returns = []
        R = next_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)

        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()
        advantages = returns - values.detach()

        # Actor loss
        policy_loss = -(log_probs * advantages).mean()

        # Critic loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus (encourages exploration)
        # Recompute distributions for entropy — simplified here
        entropy_loss = -log_probs.mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()


# ---------------------------------------------------------------------------
# 4. Proximal Policy Optimization (PPO)
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    PPO-Clip — the workhorse of modern deep RL.
    
    Key idea: constrain policy updates so the new policy doesn't
    deviate too far from the old one, using a clipped surrogate objective:
    
        L^CLIP = E[ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t) ]
    
    where r_t = π_new(a|s) / π_old(a|s)
    """

    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95, clip_eps: float = 0.2,
                 epochs_per_update: int = 4, entropy_coef: float = 0.01,
                 value_coef: float = 0.5):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs_per_update = epochs_per_update
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.net = ActorCriticNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def act(self, state: np.ndarray):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        dist, value = self.net(state_t)
        action = dist.sample()
        return action.item(), dist.log_prob(action).detach(), value.detach()

    def compute_gae(self, rewards, values, dones, next_value) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation (GAE-λ)."""
        values = torch.cat([torch.stack(values).squeeze(), next_value.unsqueeze(0)])
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self, states, actions, old_log_probs, rewards, dones, next_value):
        """PPO update with multiple epochs over the collected batch."""
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        old_log_probs_t = torch.stack(old_log_probs)

        advantages, returns = self.compute_gae(rewards, [self.net(torch.FloatTensor(s).unsqueeze(0))[1] for s in states], dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs_per_update):
            dist, values = self.net(states_t)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            # Probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs_t)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

        return policy_loss.item(), value_loss.item()


# ---------------------------------------------------------------------------
# 5. Training Loops
# ---------------------------------------------------------------------------

def train_reinforce(env_name="CartPole-v1", episodes=500):
    try:
        import gymnasium as gym
    except ImportError:
        print("pip install gymnasium")
        return
    env = gym.make(env_name)
    agent = REINFORCEAgent(env.observation_space.shape[0], env.action_space.n)
    reward_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        log_probs, rewards, states = [], [], []
        for _ in range(500):
            action, lp = agent.act(state)
            states.append(state)
            log_probs.append(lp)
            next_state, reward, term, trunc, _ = env.step(action)
            rewards.append(reward)
            state = next_state
            if term or trunc:
                break
        agent.update(log_probs, rewards, states)
        reward_history.append(sum(rewards))
        if (ep + 1) % 50 == 0:
            print(f"REINFORCE ep {ep+1} | avg(50): {np.mean(reward_history[-50:]):.1f}")
    env.close()
    return reward_history


def train_ppo(env_name="CartPole-v1", episodes=500, rollout_len=128):
    try:
        import gymnasium as gym
    except ImportError:
        print("pip install gymnasium")
        return
    env = gym.make(env_name)
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
    reward_history = []
    state, _ = env.reset()
    ep_reward = 0.0

    for step in range(episodes * rollout_len):
        states, actions, log_probs, rewards, dones = [], [], [], [], []

        for _ in range(rollout_len):
            action, lp, _ = agent.act(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            states.append(state)
            actions.append(action)
            log_probs.append(lp)
            rewards.append(reward)
            dones.append(float(done))
            ep_reward += reward

            if done:
                reward_history.append(ep_reward)
                ep_reward = 0.0
                state, _ = env.reset()
                if len(reward_history) % 50 == 0:
                    print(f"PPO ep {len(reward_history)} | avg(50): {np.mean(reward_history[-50:]):.1f}")
                if len(reward_history) >= episodes:
                    env.close()
                    return reward_history
            else:
                state = next_state

        # Compute next value for GAE
        with torch.no_grad():
            _, next_val = agent.net(torch.FloatTensor(state).unsqueeze(0))
        agent.update(states, actions, log_probs, rewards, dones, next_val)

    env.close()
    return reward_history


if __name__ == "__main__":
    print("=== REINFORCE with Baseline ===")
    train_reinforce(episodes=200)
    print("\n=== PPO ===")
    train_ppo(episodes=200)
