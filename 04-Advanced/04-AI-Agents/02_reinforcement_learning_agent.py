"""
Reinforcement Learning Agent Implementation

This module implements various RL agents including Q-Learning, Deep Q-Network (DQN),
Policy Gradient methods, and Actor-Critic algorithms for different environments.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import gym
from typing import Tuple, List, Dict, Any, Optional
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', 
                                   ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.buffer.append(self.Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List:
        """Sample a batch of transitions"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Deep Q-Network for value function approximation"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 128]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DuelingDQN(nn.Module):
    """Dueling DQN architecture that separates value and advantage streams"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DuelingDQN, self).__init__()
        
        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class DQNAgent:
    """Deep Q-Network Agent with experience replay and target network"""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        use_dueling: bool = False
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Neural networks
        if use_dueling:
            self.q_network = DuelingDQN(state_size, action_size)
            self.target_network = DuelingDQN(state_size, action_size)
        else:
            self.q_network = DQNNetwork(state_size, action_size)
            self.target_network = DQNNetwork(state_size, action_size)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.losses = []
        self.episode_rewards = []
        self.step_count = 0
        
        # Copy weights to target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state)
        action_batch = torch.LongTensor(batch.action)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(batch.next_state)
        done_batch = torch.BoolTensor(batch.done)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        self.losses.append(loss.item())
    
    def train(self, env, episodes: int = 1000, max_steps: int = 500):
        """Train the agent in the environment"""
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                self.replay()
                
                if done:
                    break
            
            self.episode_rewards.append(total_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")


class PolicyNetwork(nn.Module):
    """Policy network for policy gradient methods"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)


class REINFORCEAgent:
    """REINFORCE (Monte Carlo Policy Gradient) Agent"""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Training metrics
        self.episode_returns = []
    
    def act(self, state: np.ndarray) -> int:
        """Choose action according to policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor)
        
        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item()
    
    def remember(self, state, action, reward):
        """Store experience for current episode"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm"""
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Normalize returns
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for state, action, G in zip(self.episode_states, self.episode_actions, returns):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.policy_network(state_tensor)
            
            log_prob = torch.log(action_probs[0, action])
            policy_loss.append(-log_prob * G)
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        return policy_loss.item()
    
    def train(self, env, episodes: int = 1000, max_steps: int = 500):
        """Train the agent using REINFORCE"""
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            # Collect episode
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                
                self.remember(state, action, reward)
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Update policy
            loss = self.update_policy()
            self.episode_returns.append(total_reward)
            
            if episode % 100 == 0:
                avg_return = np.mean(self.episode_returns[-100:])
                print(f"Episode {episode}, Average Return: {avg_return:.2f}")


class ActorCriticAgent:
    """Actor-Critic Agent with separate policy and value networks"""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # Actor network (policy)
        self.actor = PolicyNetwork(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Training metrics
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
    
    def act(self, state: np.ndarray) -> int:
        """Choose action according to policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item()
    
    def update(self, state, action, reward, next_state, done):
        """Update actor and critic networks"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Critic update
        current_value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor) if not done else torch.tensor([[0.0]])
        target_value = reward + self.gamma * next_value
        
        critic_loss = F.mse_loss(current_value, target_value.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        advantage = (target_value - current_value).detach()
        
        action_probs = self.actor(state_tensor)
        log_prob = torch.log(action_probs[0, action])
        actor_loss = -log_prob * advantage
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
    
    def train(self, env, episodes: int = 1000, max_steps: int = 500):
        """Train the agent using Actor-Critic"""
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            self.episode_rewards.append(total_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")


class MultiAgentEnvironment:
    """Simple multi-agent environment for testing coordination"""
    
    def __init__(self, n_agents: int = 2, grid_size: int = 5):
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        self.agent_positions = []
        for _ in range(self.n_agents):
            pos = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            self.agent_positions.append(pos)
        
        # Place target
        self.target_pos = (np.random.randint(0, self.grid_size),
                          np.random.randint(0, self.grid_size))
        
        return self.get_observations()
    
    def get_observations(self):
        """Get observations for all agents"""
        observations = []
        for i, pos in enumerate(self.agent_positions):
            # Observation includes agent position, other agents, and target
            obs = np.zeros(self.grid_size * self.grid_size * 3)  # self, others, target
            
            # Self position
            self_idx = pos[0] * self.grid_size + pos[1]
            obs[self_idx] = 1
            
            # Other agents
            for j, other_pos in enumerate(self.agent_positions):
                if i != j:
                    other_idx = other_pos[0] * self.grid_size + other_pos[1]
                    obs[self.grid_size * self.grid_size + other_idx] = 1
            
            # Target
            target_idx = self.target_pos[0] * self.grid_size + self.target_pos[1]
            obs[2 * self.grid_size * self.grid_size + target_idx] = 1
            
            observations.append(obs)
        
        return observations
    
    def step(self, actions):
        """Execute actions for all agents"""
        rewards = []
        
        for i, action in enumerate(actions):
            # Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
            x, y = self.agent_positions[i]
            
            if action == 0 and x > 0:  # up
                x -= 1
            elif action == 1 and x < self.grid_size - 1:  # down
                x += 1
            elif action == 2 and y > 0:  # left
                y -= 1
            elif action == 3 and y < self.grid_size - 1:  # right
                y += 1
            # action == 4 is stay (no movement)
            
            self.agent_positions[i] = (x, y)
            
            # Calculate reward
            distance_to_target = abs(x - self.target_pos[0]) + abs(y - self.target_pos[1])
            reward = -distance_to_target * 0.1
            
            # Bonus for reaching target
            if (x, y) == self.target_pos:
                reward += 10
            
            # Penalty for collision with other agents
            for j, other_pos in enumerate(self.agent_positions):
                if i != j and (x, y) == other_pos:
                    reward -= 5
            
            rewards.append(reward)
        
        # Check if any agent reached target
        done = any(pos == self.target_pos for pos in self.agent_positions)
        
        return self.get_observations(), rewards, done, {}


# Example usage and testing
if __name__ == "__main__":
    # Test DQN Agent on CartPole
    print("=== Testing DQN Agent ===")
    
    try:
        env = gym.make('CartPole-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        dqn_agent = DQNAgent(state_size, action_size, use_dueling=True)
        
        # Train for a few episodes (reduce for testing)
        print("Training DQN agent...")
        dqn_agent.train(env, episodes=100, max_steps=200)
        
        # Test trained agent
        state = env.reset()
        total_reward = 0
        for _ in range(200):
            action = dqn_agent.act(state, training=False)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        
        print(f"DQN Test Episode Reward: {total_reward}")
        
    except Exception as e:
        print(f"DQN test failed: {e}")
    
    # Test REINFORCE Agent
    print("\n=== Testing REINFORCE Agent ===")
    
    try:
        env = gym.make('CartPole-v1')
        reinforce_agent = REINFORCEAgent(state_size, action_size)
        
        print("Training REINFORCE agent...")
        reinforce_agent.train(env, episodes=100, max_steps=200)
        
        print(f"REINFORCE final average return: {np.mean(reinforce_agent.episode_returns[-10:]):.2f}")
        
    except Exception as e:
        print(f"REINFORCE test failed: {e}")
    
    # Test Actor-Critic Agent
    print("\n=== Testing Actor-Critic Agent ===")
    
    try:
        env = gym.make('CartPole-v1')
        ac_agent = ActorCriticAgent(state_size, action_size)
        
        print("Training Actor-Critic agent...")
        ac_agent.train(env, episodes=100, max_steps=200)
        
        print(f"Actor-Critic final average reward: {np.mean(ac_agent.episode_rewards[-10:]):.2f}")
        
    except Exception as e:
        print(f"Actor-Critic test failed: {e}")
    
    # Test Multi-Agent Environment
    print("\n=== Testing Multi-Agent Environment ===")
    
    multi_env = MultiAgentEnvironment(n_agents=2, grid_size=5)
    
    # Simple random policy test
    observations = multi_env.reset()
    print(f"Initial observations shape: {[obs.shape for obs in observations]}")
    
    for step in range(10):
        actions = [np.random.randint(0, 5) for _ in range(multi_env.n_agents)]
        observations, rewards, done, _ = multi_env.step(actions)
        print(f"Step {step}: Rewards = {rewards}, Done = {done}")
        
        if done:
            print("Target reached!")
            break
    
    print("\n✅ All RL agent tests completed!")