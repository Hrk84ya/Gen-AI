# Reinforcement Learning

## 🎯 Learning Objectives
By the end of this module, you will:
- Understand the RL framework: agents, environments, rewards, and policies
- Implement value-based methods (Q-Learning, DQN, Double DQN)
- Implement policy gradient methods (PPO, A2C)
- Understand model-based RL and planning
- Apply RL to continuous control and game environments
- Know when RL is (and isn't) the right approach

## 🧠 What is Reinforcement Learning?

Reinforcement Learning is a paradigm where an agent learns to make decisions by interacting with an environment. Unlike supervised learning, there are no labeled examples — the agent discovers good behavior through trial-and-error, guided by reward signals.

### Core Components
- **Agent**: The learner and decision-maker
- **Environment**: Everything the agent interacts with
- **State (s)**: A representation of the current situation
- **Action (a)**: A choice the agent can make
- **Reward (r)**: A scalar feedback signal
- **Policy (π)**: A mapping from states to actions
- **Value Function (V/Q)**: Expected cumulative reward

### The RL Loop
```
Agent observes state s_t
  → selects action a_t via policy π
  → environment transitions to s_{t+1}
  → agent receives reward r_t
  → agent updates its policy
  → repeat
```

## 📚 Module Contents

### 1. [RL Fundamentals & Tabular Methods](./01_rl_fundamentals.py)
- Markov Decision Processes (MDPs)
- Bellman equations
- Q-Learning and SARSA (tabular)
- Monte Carlo methods
- Exploration vs exploitation (ε-greedy, UCB, Boltzmann)

### 2. [Deep RL: Value-Based Methods](./02_deep_rl_value.py)
- Deep Q-Networks (DQN) with target networks
- Double DQN, Dueling DQN, Prioritized Experience Replay
- Rainbow DQN components

### 3. [Deep RL: Policy Gradient Methods](./03_deep_rl_policy.py)
- REINFORCE with baseline
- Advantage Actor-Critic (A2C)
- Proximal Policy Optimization (PPO)
- Generalized Advantage Estimation (GAE)

### 4. [Advanced Topics](./04_advanced_rl.py)
- Model-based RL and world models
- Multi-agent RL basics
- Reward shaping and curriculum learning
- Offline RL / batch RL overview

## 🔗 Relationship to Other Modules
- The [AI Agents](../04-AI-Agents/) module covers a basic DQN/REINFORCE/Actor-Critic implementation. This module goes deeper with PPO, model-based methods, and advanced techniques.

## 📚 Additional Resources

### Books
- "Reinforcement Learning: An Introduction" by Sutton & Barto (the bible of RL)
- "Deep Reinforcement Learning Hands-On" by Maxim Lapan

### Papers
- "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- "Mastering the Game of Go with Deep Neural Networks" (Silver et al., 2016)

### Online
- [Spinning Up in Deep RL](https://spinningup.openai.com/) (OpenAI)
- [David Silver's RL Course](https://www.davidsilver.uk/teaching/) (UCL)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---
**Next Module**: [Graph Neural Networks](../08-Graph-Neural-Networks/) →
