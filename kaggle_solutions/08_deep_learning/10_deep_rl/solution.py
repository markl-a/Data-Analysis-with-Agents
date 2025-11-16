"""
Deep Q-Network (DQN) for Reinforcement Learning - Kaggle Solution Example
==========================================================================

This example demonstrates Deep Q-Learning, where a neural network learns
to play a simple grid-world game by maximizing cumulative rewards.

Problem: Train an agent to navigate a grid world to reach goals while avoiding obstacles

Approach:
1. Define a simple grid-world environment
2. Implement Deep Q-Network (DQN) with experience replay
3. Train agent using epsilon-greedy exploration
4. Visualize learning progress and agent behavior
5. Compare with random baseline

Author: Kaggle Competition Team
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
random.seed(42)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class GridWorld:
    """
    Simple grid-world environment for RL.

    Agent starts at top-left, goal at bottom-right.
    Obstacles scattered throughout.
    """

    def __init__(self, size=10, n_obstacles=15):
        """
        Initialize grid world.

        Args:
            size: Grid size (size x size)
            n_obstacles: Number of obstacle cells
        """
        self.size = size
        self.n_obstacles = n_obstacles
        self.action_space = 4  # Up, Right, Down, Left
        self.state_space = size * size

        # Action mappings
        self.actions = {
            0: (-1, 0),  # Up
            1: (0, 1),   # Right
            2: (1, 0),   # Down
            3: (0, -1)   # Left
        }

        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        # Create grid
        self.grid = np.zeros((self.size, self.size))

        # Place obstacles randomly
        self.obstacles = set()
        while len(self.obstacles) < self.n_obstacles:
            pos = (np.random.randint(1, self.size-1),
                   np.random.randint(1, self.size-1))
            if pos != (0, 0):  # Not at start
                self.obstacles.add(pos)
                self.grid[pos] = -1

        # Start and goal positions
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.grid[self.goal] = 1

        # Agent position
        self.agent_pos = self.start
        self.steps = 0
        self.max_steps = self.size * self.size * 2

        return self.get_state()

    def get_state(self):
        """Get current state as one-hot encoded position."""
        state = np.zeros(self.state_space)
        idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        state[idx] = 1
        return state

    def step(self, action):
        """
        Take action and return next state, reward, done.

        Args:
            action: Integer action (0-3)

        Returns:
            next_state, reward, done, info
        """
        self.steps += 1

        # Calculate new position
        delta = self.actions[action]
        new_pos = (self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1])

        # Check boundaries
        if (new_pos[0] < 0 or new_pos[0] >= self.size or
            new_pos[1] < 0 or new_pos[1] >= self.size):
            reward = -0.5  # Wall penalty
            done = False
        # Check obstacles
        elif new_pos in self.obstacles:
            reward = -1.0  # Obstacle penalty
            done = False
        # Check goal
        elif new_pos == self.goal:
            reward = 10.0  # Goal reward
            done = True
            self.agent_pos = new_pos
        # Valid move
        else:
            reward = -0.1  # Step penalty to encourage efficiency
            done = False
            self.agent_pos = new_pos

        # Check max steps
        if self.steps >= self.max_steps:
            done = True

        return self.get_state(), reward, done, {}

    def render(self):
        """Render the grid world."""
        grid_visual = self.grid.copy()
        grid_visual[self.agent_pos] = 2  # Agent marker
        return grid_visual


class DQN:
    """
    Deep Q-Network implementation.

    Uses a neural network to approximate Q-values and experience replay
    for stable training.
    """

    def __init__(self, state_size, action_size, hidden_sizes=[128, 64]):
        """
        Initialize DQN.

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        self.state_size = state_size
        self.action_size = action_size

        # Q-network
        self.weights = []
        self.biases = []

        # Build network: state -> hidden layers -> Q-values
        layer_sizes = [state_size] + hidden_sizes + [action_size]

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

    def relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ReLU derivative."""
        return (x > 0).astype(float)

    def forward(self, state):
        """Forward pass through network."""
        self.activations = [state]
        self.z_values = []

        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)

        # Output layer (linear)
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(z)

        return self.activations[-1]

    def predict(self, state):
        """Predict Q-values for state."""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        return self.forward(state)

    def act(self, state, explore=True):
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state
            explore: Whether to use exploration

        Returns:
            Action index
        """
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        q_values = self.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append(Experience(state, action, reward, next_state, done))

    def replay(self):
        """Train on batch of experiences from replay buffer."""
        if len(self.memory) < self.batch_size:
            return 0

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)

        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])

        # Predict Q-values
        current_q = self.predict(states)
        next_q = self.predict(next_states)

        # Compute targets
        targets = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

        # Backpropagation
        loss = self.backward(states, targets)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def backward(self, states, targets):
        """Backpropagation to update weights."""
        m = states.shape[0]

        # Forward pass
        self.forward(states)

        # Compute loss (MSE)
        predictions = self.activations[-1]
        loss = np.mean((predictions - targets) ** 2)

        # Backward pass
        delta = 2 * (predictions - targets) / m

        # Update weights
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(self.activations[i].T, delta)
            grad_b = np.sum(delta, axis=0, keepdims=True)

            self.weights[i] -= self.learning_rate * grad_w
            self.biases[i] -= self.learning_rate * grad_b

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])

        return loss


def train_dqn(env, agent, n_episodes=500, verbose=True):
    """
    Train DQN agent.

    Args:
        env: Environment instance
        agent: DQN agent
        n_episodes: Number of training episodes
        verbose: Print progress

    Returns:
        Training history
    """
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'epsilon': [],
        'losses': []
    }

    if verbose:
        print(f"🎮 Training DQN for {n_episodes} episodes...")

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_losses = []

        done = False
        while not done:
            # Choose action
            action = agent.act(state, explore=True)

            # Take step
            next_state, reward, done, _ = env.step(action)

            # Remember
            agent.remember(state, action, reward, next_state, done)

            # Train
            loss = agent.replay()
            if loss > 0:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward
            steps += 1

        # Record history
        history['episode_rewards'].append(total_reward)
        history['episode_lengths'].append(steps)
        history['epsilon'].append(agent.epsilon)
        history['losses'].append(np.mean(episode_losses) if episode_losses else 0)

        # Print progress
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(history['episode_rewards'][-50:])
            avg_length = np.mean(history['episode_lengths'][-50:])
            print(f"  Episode {episode+1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    return history


def evaluate_agent(env, agent, n_episodes=100):
    """Evaluate trained agent without exploration."""
    rewards = []
    lengths = []
    success_count = 0

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.act(state, explore=False)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

            if reward == 10.0:  # Reached goal
                success_count += 1

        rewards.append(total_reward)
        lengths.append(steps)

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'success_rate': success_count / n_episodes
    }


def visualize_results(history, eval_results, env, agent):
    """Create comprehensive visualizations."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('Deep Q-Network (DQN) Training Results', fontsize=16, fontweight='bold')

    # 1. Episode rewards
    ax = fig.add_subplot(gs[0, 0])
    episodes = range(1, len(history['episode_rewards']) + 1)
    ax.plot(episodes, history['episode_rewards'], alpha=0.3, color='blue')

    # Moving average
    window = 20
    moving_avg = np.convolve(history['episode_rewards'],
                            np.ones(window)/window, mode='valid')
    ax.plot(range(window, len(episodes)+1), moving_avg,
           color='red', linewidth=2, label=f'{window}-Episode MA')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Rewards', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Episode lengths
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(episodes, history['episode_lengths'], alpha=0.3, color='green')

    moving_avg_len = np.convolve(history['episode_lengths'],
                                 np.ones(window)/window, mode='valid')
    ax.plot(range(window, len(episodes)+1), moving_avg_len,
           color='darkgreen', linewidth=2, label=f'{window}-Episode MA')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length (Efficiency)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Epsilon decay
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(episodes, history['epsilon'], color='purple', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Training loss
    ax = fig.add_subplot(gs[1, 0])
    valid_losses = [l for l in history['losses'] if l > 0]
    ax.plot(valid_losses, alpha=0.7, color='orange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (MSE)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 5. Environment visualization
    ax = fig.add_subplot(gs[1, 1])
    env_visual = env.render()

    cmap = plt.cm.colors.ListedColormap(['white', 'red', 'green', 'blue'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(env_visual, cmap=cmap, norm=norm)
    ax.set_title('Environment State\n(Red=Obstacle, Green=Goal, Blue=Agent)',
                fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # 6. Q-value heatmap
    ax = fig.add_subplot(gs[1, 2])
    q_map = np.zeros((env.size, env.size))

    for i in range(env.size):
        for j in range(env.size):
            state = np.zeros(env.state_space)
            state[i * env.size + j] = 1
            q_values = agent.predict(state)
            q_map[i, j] = np.max(q_values)

    im = ax.imshow(q_map, cmap='viridis')
    ax.set_title('Learned Q-Values (Max)', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)

    # 7. Performance comparison
    ax = fig.add_subplot(gs[2, 0])

    metrics = ['Mean\nReward', 'Success\nRate']
    dqn_values = [eval_results['mean_reward'], eval_results['success_rate'] * 10]
    random_values = [-2.0, 0.05 * 10]  # Approximate random baseline

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, dqn_values, width, label='DQN', color='green', alpha=0.7)
    ax.bar(x + width/2, random_values, width, label='Random', color='red', alpha=0.7)

    ax.set_ylabel('Score')
    ax.set_title('Agent Performance Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 8. Learning progress
    ax = fig.add_subplot(gs[2, 1])

    segment_size = len(history['episode_rewards']) // 5
    segments = ['Early', 'Q1', 'Mid', 'Q3', 'Final']
    segment_rewards = []

    for i in range(5):
        start = i * segment_size
        end = start + segment_size if i < 4 else len(history['episode_rewards'])
        segment_rewards.append(np.mean(history['episode_rewards'][start:end]))

    ax.plot(segments, segment_rewards, 'o-', linewidth=2, markersize=8, color='blue')
    ax.set_ylabel('Average Reward')
    ax.set_title('Learning Progress Over Time', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 9. Summary statistics
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    summary = f"""
    TRAINING SUMMARY
    ══════════════════════

    Episodes: {len(history['episode_rewards'])}

    Final Performance:
    • Mean Reward: {eval_results['mean_reward']:.2f}
    • Success Rate: {eval_results['success_rate']*100:.1f}%
    • Avg Steps: {eval_results['mean_length']:.1f}

    Best Episode:
    • Reward: {max(history['episode_rewards']):.2f}

    Final Epsilon: {history['epsilon'][-1]:.4f}

    Convergence:
    • First 100: {np.mean(history['episode_rewards'][:100]):.2f}
    • Last 100: {np.mean(history['episode_rewards'][-100:]):.2f}
    • Improvement: {np.mean(history['episode_rewards'][-100:]) - np.mean(history['episode_rewards'][:100]):.2f}
    """

    ax.text(0.1, 0.5, summary, fontsize=10, fontfamily='monospace',
           verticalalignment='center')

    plt.savefig('/tmp/dqn_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Visualization saved to /tmp/dqn_results.png")
    plt.show()


def main():
    """Main execution function."""
    print("=" * 70)
    print("DEEP Q-NETWORK (DQN) - KAGGLE SOLUTION")
    print("=" * 70)

    # Create environment
    print("\n🎮 Initializing Grid World Environment...")
    env = GridWorld(size=10, n_obstacles=15)

    print(f"  Grid Size: {env.size}x{env.size}")
    print(f"  Obstacles: {env.n_obstacles}")
    print(f"  State Space: {env.state_space}")
    print(f"  Action Space: {env.action_space}")

    # Create agent
    print("\n🤖 Initializing DQN Agent...")
    agent = DQN(
        state_size=env.state_space,
        action_size=env.action_space,
        hidden_sizes=[128, 64]
    )

    print(f"  Network: {env.state_space} -> 128 -> 64 -> {env.action_space}")
    print(f"  Gamma: {agent.gamma}")
    print(f"  Initial Epsilon: {agent.epsilon}")

    # Train agent
    print("\n" + "=" * 70)
    history = train_dqn(env, agent, n_episodes=500, verbose=True)

    # Evaluate agent
    print("\n" + "=" * 70)
    print("📊 Evaluating Trained Agent...")
    eval_results = evaluate_agent(env, agent, n_episodes=100)

    print(f"\n✅ Evaluation Results (100 episodes):")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Mean Steps: {eval_results['mean_length']:.1f}")
    print(f"  Success Rate: {eval_results['success_rate']*100:.1f}%")

    # Visualize
    print("\n📊 Generating visualizations...")
    visualize_results(history, eval_results, env, agent)

    print("\n" + "=" * 70)
    print("✅ DEEP Q-NETWORK TRAINING COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
