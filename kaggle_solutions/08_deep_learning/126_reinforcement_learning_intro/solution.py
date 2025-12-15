"""
Reinforcement Learning Introduction - CartPole

Learn the fundamentals of reinforcement learning using the classic
CartPole balancing task with Q-Learning and DQN.

Dataset: https://www.kaggle.com/datasets/balajibaskar/openai-gym-cartpole
Difficulty: ⭐⭐⭐ Advanced Level
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

# Simulated environment (no gym dependency)
class CartPoleEnv:
    """Simplified CartPole environment simulation."""

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        # Thresholds
        self.theta_threshold = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        self.state = None
        self.steps = 0

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return new state, reward, done."""
        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
        )

        reward = 1.0 if not done else 0.0

        return self.state.copy(), reward, done


class QLearningAgent:
    """Q-Learning agent with discretized state space."""

    def __init__(self, n_bins: int = 20, learning_rate: float = 0.1,
                 discount_factor: float = 0.99, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.n_bins = n_bins
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # State space bounds
        self.state_bounds = [
            (-2.4, 2.4),      # x
            (-3.0, 3.0),      # x_dot
            (-0.21, 0.21),    # theta
            (-3.0, 3.0)       # theta_dot
        ]

        # Initialize Q-table
        self.q_table = np.zeros([n_bins] * 4 + [2])

    def discretize_state(self, state: np.ndarray) -> Tuple:
        """Convert continuous state to discrete indices."""
        discrete = []
        for i, val in enumerate(state):
            low, high = self.state_bounds[i]
            val = np.clip(val, low, high)
            bin_idx = int((val - low) / (high - low) * (self.n_bins - 1))
            discrete.append(bin_idx)
        return tuple(discrete)

    def choose_action(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        discrete_state = self.discretize_state(state)
        return np.argmax(self.q_table[discrete_state])

    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        """Update Q-value using Q-learning update rule."""
        s = self.discretize_state(state)
        s_next = self.discretize_state(next_state)

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[s_next])

        self.q_table[s + (action,)] += self.lr * (target - self.q_table[s + (action,)])

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DQNAgent:
    """Deep Q-Network agent (simplified numpy implementation)."""

    def __init__(self, state_size: int = 4, action_size: int = 2,
                 hidden_size: int = 64, learning_rate: float = 0.001,
                 discount_factor: float = 0.99, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 memory_size: int = 10000, batch_size: int = 64):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        # Experience replay
        self.memory = deque(maxlen=memory_size)

        # Initialize network weights
        self._init_network()

    def _init_network(self) -> None:
        """Initialize neural network weights."""
        scale = 0.1
        self.W1 = np.random.randn(self.state_size, self.hidden_size) * scale
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.hidden_size) * scale
        self.b2 = np.zeros(self.hidden_size)
        self.W3 = np.random.randn(self.hidden_size, self.action_size) * scale
        self.b3 = np.zeros(self.action_size)

        # Target network
        self.W1_target = self.W1.copy()
        self.b1_target = self.b1.copy()
        self.W2_target = self.W2.copy()
        self.b2_target = self.b2.copy()
        self.W3_target = self.W3.copy()
        self.b3_target = self.b3.copy()

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    def _forward(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """Forward pass through the network."""
        if use_target:
            W1, b1, W2, b2, W3, b3 = (self.W1_target, self.b1_target,
                                       self.W2_target, self.b2_target,
                                       self.W3_target, self.b3_target)
        else:
            W1, b1, W2, b2, W3, b3 = (self.W1, self.b1, self.W2, self.b2,
                                       self.W3, self.b3)

        h1 = self._relu(np.dot(state, W1) + b1)
        h2 = self._relu(np.dot(h1, W2) + b2)
        return np.dot(h2, W3) + b3

    def choose_action(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self._forward(state.reshape(1, -1))
        return np.argmax(q_values[0])

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> float:
        """Train on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        # Compute targets
        current_q = self._forward(states)
        next_q = self._forward(next_states, use_target=True)

        targets = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

        # Simple gradient descent update
        loss = self._train_step(states, targets)
        return loss

    def _train_step(self, states: np.ndarray, targets: np.ndarray) -> float:
        """Perform one training step."""
        # Forward pass with intermediate values
        h1 = self._relu(np.dot(states, self.W1) + self.b1)
        h2 = self._relu(np.dot(h1, self.W2) + self.b2)
        output = np.dot(h2, self.W3) + self.b3

        # Compute loss
        loss = np.mean((output - targets) ** 2)

        # Backward pass (simplified)
        d_output = 2 * (output - targets) / self.batch_size

        d_W3 = np.dot(h2.T, d_output)
        d_b3 = np.sum(d_output, axis=0)

        d_h2 = np.dot(d_output, self.W3.T)
        d_h2[h2 <= 0] = 0

        d_W2 = np.dot(h1.T, d_h2)
        d_b2 = np.sum(d_h2, axis=0)

        d_h1 = np.dot(d_h2, self.W2.T)
        d_h1[h1 <= 0] = 0

        d_W1 = np.dot(states.T, d_h1)
        d_b1 = np.sum(d_h1, axis=0)

        # Update weights
        self.W3 -= self.lr * d_W3
        self.b3 -= self.lr * d_b3
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1

        return loss

    def update_target_network(self) -> None:
        """Copy weights to target network."""
        self.W1_target = self.W1.copy()
        self.b1_target = self.b1.copy()
        self.W2_target = self.W2.copy()
        self.b2_target = self.b2.copy()
        self.W3_target = self.W3.copy()
        self.b3_target = self.b3.copy()

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class RLTrainer:
    """Reinforcement Learning Trainer."""

    def __init__(self):
        self.env = CartPoleEnv()
        self.results: Dict[str, List] = {}

    def train_q_learning(self, n_episodes: int = 500) -> List[float]:
        """Train Q-Learning agent."""
        print("\nTraining Q-Learning Agent...")
        agent = QLearningAgent()
        rewards = []

        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done and total_reward < 500:
                action = agent.choose_action(state)
                next_state, reward, done = self.env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            agent.decay_epsilon()
            rewards.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.1f}, "
                      f"Epsilon = {agent.epsilon:.3f}")

        self.results['Q-Learning'] = rewards
        return rewards

    def train_dqn(self, n_episodes: int = 300) -> List[float]:
        """Train DQN agent."""
        print("\nTraining DQN Agent...")
        agent = DQNAgent()
        rewards = []
        target_update_freq = 10

        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done and total_reward < 500:
                action = agent.choose_action(state)
                next_state, reward, done = self.env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state
                total_reward += reward

            agent.decay_epsilon()

            if (episode + 1) % target_update_freq == 0:
                agent.update_target_network()

            rewards.append(total_reward)

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(rewards[-50:])
                print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.1f}, "
                      f"Epsilon = {agent.epsilon:.3f}")

        self.results['DQN'] = rewards
        return rewards

    def plot_results(self, output_dir: str = '.') -> None:
        """Visualize training results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reinforcement Learning Results - CartPole', fontsize=16)

        colors = {'Q-Learning': 'blue', 'DQN': 'green'}

        # Learning curves
        for name, rewards in self.results.items():
            axes[0, 0].plot(rewards, alpha=0.3, color=colors[name])
            # Moving average
            window = 50
            if len(rewards) >= window:
                ma = pd.Series(rewards).rolling(window=window).mean()
                axes[0, 0].plot(ma, label=f'{name} (MA-{window})',
                               color=colors[name], linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Learning Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Reward distribution
        for i, (name, rewards) in enumerate(self.results.items()):
            axes[0, 1].hist(rewards, bins=30, alpha=0.5, label=name,
                           color=colors[name])
        axes[0, 1].set_xlabel('Total Reward')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Reward Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Final performance comparison
        final_100 = {name: np.mean(rewards[-100:])
                     for name, rewards in self.results.items()}
        bars = axes[1, 0].bar(final_100.keys(), final_100.values(),
                              color=[colors[k] for k in final_100.keys()])
        axes[1, 0].set_ylabel('Average Reward (Last 100)')
        axes[1, 0].set_title('Final Performance')
        for bar, val in zip(bars, final_100.values()):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                           f'{val:.1f}', ha='center')
        axes[1, 0].grid(True, alpha=0.3)

        # Success rate over time
        threshold = 195  # CartPole solved threshold
        for name, rewards in self.results.items():
            success_rate = []
            window = 50
            for i in range(len(rewards)):
                start = max(0, i - window + 1)
                rate = sum(1 for r in rewards[start:i+1] if r >= threshold) / (i - start + 1)
                success_rate.append(rate * 100)
            axes[1, 1].plot(success_rate, label=name, color=colors[name])
        axes[1, 1].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='Solved')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_title(f'Success Rate (Reward >= {threshold})')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/rl_results.png', dpi=300, bbox_inches='tight')
        print(f"Results saved to {output_dir}/rl_results.png")
        plt.close()

    def get_summary(self) -> pd.DataFrame:
        """Get training summary."""
        summary = []
        for name, rewards in self.results.items():
            summary.append({
                'Algorithm': name,
                'Max Reward': max(rewards),
                'Avg Reward (Last 100)': np.mean(rewards[-100:]),
                'Std Reward (Last 100)': np.std(rewards[-100:]),
                'Episodes to 100+': next((i for i, r in enumerate(rewards) if r >= 100), -1),
                'Success Rate (%)': sum(1 for r in rewards if r >= 195) / len(rewards) * 100
            })
        return pd.DataFrame(summary)


def main():
    """Main execution."""
    print("=" * 70)
    print("REINFORCEMENT LEARNING - CARTPOLE")
    print("=" * 70)

    trainer = RLTrainer()

    # Train agents
    trainer.train_q_learning(n_episodes=500)
    trainer.train_dqn(n_episodes=300)

    # Results
    summary = trainer.get_summary()
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    # Visualize
    trainer.plot_results()

    print("\n" + "=" * 70)
    best = summary.loc[summary['Avg Reward (Last 100)'].idxmax()]
    print(f"Best Algorithm: {best['Algorithm']}")
    print(f"Best Avg Reward: {best['Avg Reward (Last 100)']:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
