"""
Multi-Armed Bandits for Recommendations
========================================

This solution demonstrates multi-armed bandit algorithms for recommendation systems,
including ε-greedy, UCB, Thompson Sampling, and contextual bandits (LinUCB).

Author: Kaggle Solutions Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from scipy.stats import beta
import warnings
warnings.filterwarnings('ignore')


class EpsilonGreedy:
    """Epsilon-Greedy bandit algorithm"""

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        """
        Initialize Epsilon-Greedy

        Args:
            n_arms: Number of arms (items)
            epsilon: Exploration rate
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self) -> int:
        """
        Select an arm to pull

        Returns:
            Selected arm index
        """
        if np.random.random() < self.epsilon:
            # Explore: random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: best arm
            return np.argmax(self.values)

    def update(self, arm: int, reward: float):
        """
        Update arm statistics

        Args:
            arm: Arm that was pulled
            reward: Reward received
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward


class UCB:
    """Upper Confidence Bound bandit algorithm"""

    def __init__(self, n_arms: int, c: float = 2.0):
        """
        Initialize UCB

        Args:
            n_arms: Number of arms
            c: Exploration parameter
        """
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0

    def select_arm(self) -> int:
        """
        Select an arm using UCB

        Returns:
            Selected arm index
        """
        # Pull each arm once initially
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            bonus = np.sqrt((self.c * np.log(self.total_counts)) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus

        return np.argmax(ucb_values)

    def update(self, arm: int, reward: float):
        """Update arm statistics"""
        self.counts[arm] += 1
        self.total_counts += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward


class ThompsonSampling:
    """Thompson Sampling bandit algorithm"""

    def __init__(self, n_arms: int):
        """
        Initialize Thompson Sampling

        Args:
            n_arms: Number of arms
        """
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Successes + 1
        self.beta_param = np.ones(n_arms)  # Failures + 1

    def select_arm(self) -> int:
        """
        Select an arm using Thompson Sampling

        Returns:
            Selected arm index
        """
        # Sample from beta distribution for each arm
        samples = np.random.beta(self.alpha, self.beta_param)
        return np.argmax(samples)

    def update(self, arm: int, reward: float):
        """
        Update arm statistics

        Args:
            arm: Arm that was pulled
            reward: Reward received (0 or 1 for binary)
        """
        # For continuous rewards, we approximate
        if reward > 0.5:  # Treat as success
            self.alpha[arm] += 1
        else:  # Treat as failure
            self.beta_param[arm] += 1


class LinUCB:
    """Linear UCB for contextual bandits"""

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        """
        Initialize LinUCB

        Args:
            n_arms: Number of arms
            n_features: Number of features in context
            alpha: Exploration parameter
        """
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha

        # Initialize parameters for each arm
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm given context

        Args:
            context: Context vector

        Returns:
            Selected arm index
        """
        p = np.zeros(self.n_arms)

        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv.dot(self.b[arm])

            # Calculate UCB
            p[arm] = theta.dot(context) + self.alpha * np.sqrt(
                context.dot(A_inv).dot(context)
            )

        return np.argmax(p)

    def update(self, arm: int, context: np.ndarray, reward: float):
        """
        Update arm statistics

        Args:
            arm: Arm that was pulled
            context: Context vector
            reward: Reward received
        """
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context


def generate_bandit_data(n_arms: int = 10, n_rounds: int = 10000,
                        contextual: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate bandit data with true reward distributions

    Args:
        n_arms: Number of arms
        n_rounds: Number of rounds
        contextual: Whether to include context

    Returns:
        True rewards and contexts (if contextual)
    """
    np.random.seed(42)

    # True reward probabilities for each arm
    true_rewards = np.random.beta(2, 5, n_arms)

    if contextual:
        # Generate contexts (e.g., user features)
        n_features = 5
        contexts = np.random.randn(n_rounds, n_features)

        # True parameters for each arm
        true_theta = np.random.randn(n_arms, n_features) * 0.5

        return true_rewards, contexts, true_theta
    else:
        return true_rewards, None


def simulate_bandit(algorithm, true_rewards: np.ndarray, n_rounds: int,
                   contexts: np.ndarray = None, true_theta: np.ndarray = None) -> Dict:
    """
    Simulate bandit algorithm

    Args:
        algorithm: Bandit algorithm instance
        true_rewards: True reward for each arm
        n_rounds: Number of rounds to simulate
        contexts: Context vectors (for contextual bandits)
        true_theta: True parameters (for contextual bandits)

    Returns:
        Dictionary with simulation results
    """
    rewards = []
    regrets = []
    cumulative_regret = 0
    arm_selections = []

    # Find optimal arm/reward
    if contexts is not None and true_theta is not None:
        # For contextual bandits, optimal arm varies by context
        optimal_rewards = []
    else:
        optimal_arm = np.argmax(true_rewards)
        optimal_reward = true_rewards[optimal_arm]

    for t in range(n_rounds):
        # Select arm
        if contexts is not None:
            context = contexts[t]
            arm = algorithm.select_arm(context)

            # Calculate reward based on context
            reward_mean = true_theta[arm].dot(context)
            reward = reward_mean + np.random.randn() * 0.1
            reward = np.clip(reward, 0, 1)

            # Optimal reward for this context
            optimal_reward = max(true_theta[a].dot(context) for a in range(len(true_rewards)))
            optimal_rewards.append(optimal_reward)
        else:
            arm = algorithm.select_arm()
            # Bernoulli reward
            reward = 1.0 if np.random.random() < true_rewards[arm] else 0.0

        # Update algorithm
        if contexts is not None:
            algorithm.update(arm, context, reward)
        else:
            algorithm.update(arm, reward)

        # Track metrics
        rewards.append(reward)
        arm_selections.append(arm)

        if contexts is not None:
            regret = optimal_reward - reward
        else:
            regret = optimal_reward - true_rewards[arm]

        cumulative_regret += regret
        regrets.append(cumulative_regret)

    return {
        'rewards': rewards,
        'regrets': regrets,
        'arm_selections': arm_selections,
        'cumulative_reward': sum(rewards)
    }


def plot_bandit_performance(results: Dict[str, Dict], save_path: str = None):
    """Plot bandit algorithm performance"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    algorithms = list(results.keys())

    # Cumulative regret
    for name, result in results.items():
        axes[0, 0].plot(result['regrets'], label=name, linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Cumulative Regret')
    axes[0, 0].set_title('Cumulative Regret Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative rewards
    for name, result in results.items():
        cumulative_rewards = np.cumsum(result['rewards'])
        axes[0, 1].plot(cumulative_rewards, label=name, linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Cumulative Reward')
    axes[0, 1].set_title('Cumulative Reward Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Average reward (moving average)
    window = 100
    for name, result in results.items():
        rewards_array = np.array(result['rewards'])
        moving_avg = np.convolve(rewards_array, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(moving_avg, label=name, linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Average Reward')
    axes[1, 0].set_title(f'Moving Average Reward (window={window})', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Final cumulative rewards comparison
    final_rewards = [result['cumulative_reward'] for result in results.values()]
    bars = axes[1, 1].bar(range(len(algorithms)), final_rewards,
                          color=['skyblue', 'coral', 'lightgreen', 'plum'][:len(algorithms)])
    axes[1, 1].set_xticks(range(len(algorithms)))
    axes[1, 1].set_xticklabels(algorithms, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Total Reward')
    axes[1, 1].set_title('Total Cumulative Reward', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_arm_selections(results: Dict[str, Dict], n_arms: int, save_path: str = None):
    """Plot arm selection distributions"""
    n_algorithms = len(results)
    fig, axes = plt.subplots(1, n_algorithms, figsize=(6 * n_algorithms, 5))

    if n_algorithms == 1:
        axes = [axes]

    for idx, (name, result) in enumerate(results.items()):
        arm_counts = np.bincount(result['arm_selections'], minlength=n_arms)
        axes[idx].bar(range(n_arms), arm_counts, color='skyblue', edgecolor='black')
        axes[idx].set_xlabel('Arm Index')
        axes[idx].set_ylabel('Selection Count')
        axes[idx].set_title(f'{name} - Arm Selections', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_exploration_exploitation(results: Dict[str, Dict], true_rewards: np.ndarray,
                                  save_path: str = None):
    """Plot exploration vs exploitation trade-off"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    optimal_arm = np.argmax(true_rewards)
    algorithms = list(results.keys())

    # Optimal arm selection rate over time
    window = 100
    for name, result in results.items():
        selections = np.array(result['arm_selections'])
        optimal_selections = (selections == optimal_arm).astype(int)
        moving_avg = np.convolve(optimal_selections, np.ones(window)/window, mode='valid')
        axes[0].plot(moving_avg, label=name, linewidth=2, alpha=0.8)

    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Optimal Arm Selection Rate')
    axes[0].set_title('Convergence to Optimal Arm', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Final optimal arm selection percentage
    optimal_percentages = []
    for name, result in results.items():
        selections = np.array(result['arm_selections'])
        optimal_pct = np.mean(selections == optimal_arm) * 100
        optimal_percentages.append(optimal_pct)

    bars = axes[1].bar(range(len(algorithms)), optimal_percentages,
                       color=['skyblue', 'coral', 'lightgreen', 'plum'][:len(algorithms)])
    axes[1].set_xticks(range(len(algorithms)))
    axes[1].set_xticklabels(algorithms, rotation=45, ha='right')
    axes[1].set_ylabel('Optimal Arm Selection (%)')
    axes[1].set_title('Overall Optimal Arm Selection', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_regret_analysis(results: Dict[str, Dict], save_path: str = None):
    """Plot detailed regret analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Regret growth rate
    for name, result in results.items():
        regrets = np.array(result['regrets'])
        # Calculate regret per round (derivative)
        regret_rate = np.diff(regrets)
        # Smooth with moving average
        window = 100
        if len(regret_rate) >= window:
            smoothed = np.convolve(regret_rate, np.ones(window)/window, mode='valid')
            axes[0].plot(smoothed, label=name, linewidth=2, alpha=0.8)

    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Regret per Round')
    axes[0].set_title('Regret Growth Rate', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Final regret comparison
    algorithms = list(results.keys())
    final_regrets = [result['regrets'][-1] for result in results.values()]

    bars = axes[1].bar(range(len(algorithms)), final_regrets,
                       color=['skyblue', 'coral', 'lightgreen', 'plum'][:len(algorithms)])
    axes[1].set_xticks(range(len(algorithms)))
    axes[1].set_xticklabels(algorithms, rotation=45, ha='right')
    axes[1].set_ylabel('Total Regret')
    axes[1].set_title('Final Cumulative Regret', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_contextual_performance(linucb_result: Dict, save_path: str = None):
    """Plot contextual bandit performance"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Cumulative reward
    cumulative_rewards = np.cumsum(linucb_result['rewards'])
    axes[0, 0].plot(cumulative_rewards, linewidth=2, color='steelblue')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Cumulative Reward')
    axes[0, 0].set_title('LinUCB Cumulative Reward', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative regret
    axes[0, 1].plot(linucb_result['regrets'], linewidth=2, color='coral')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Cumulative Regret')
    axes[0, 1].set_title('LinUCB Cumulative Regret', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Reward distribution
    axes[1, 0].hist(linucb_result['rewards'], bins=50, color='lightgreen',
                    edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Reward Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Arm selection distribution
    arm_counts = np.bincount(linucb_result['arm_selections'])
    axes[1, 1].bar(range(len(arm_counts)), arm_counts, color='plum', edgecolor='black')
    axes[1, 1].set_xlabel('Arm Index')
    axes[1, 1].set_ylabel('Selection Count')
    axes[1, 1].set_title('Arm Selection Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Multi-Armed Bandits for Recommendations")
    print("=" * 80)

    # Parameters
    n_arms = 10
    n_rounds = 10000

    # Generate data
    print("\n1. Generating bandit environment...")
    true_rewards, _ = generate_bandit_data(n_arms=n_arms, n_rounds=n_rounds,
                                          contextual=False)
    print(f"Number of arms: {n_arms}")
    print(f"Number of rounds: {n_rounds}")
    print(f"True reward probabilities: {true_rewards}")
    print(f"Optimal arm: {np.argmax(true_rewards)} (reward: {true_rewards.max():.3f})")

    # Initialize algorithms
    print("\n2. Initializing bandit algorithms...")
    algorithms = {
        'ε-Greedy (ε=0.1)': EpsilonGreedy(n_arms, epsilon=0.1),
        'UCB (c=2.0)': UCB(n_arms, c=2.0),
        'Thompson Sampling': ThompsonSampling(n_arms)
    }

    # Run simulations
    print("\n3. Running simulations...")
    results = {}
    for name, algorithm in algorithms.items():
        print(f"\n   Simulating {name}...")
        result = simulate_bandit(algorithm, true_rewards, n_rounds)
        results[name] = result
        print(f"   Total reward: {result['cumulative_reward']:.2f}")
        print(f"   Final regret: {result['regrets'][-1]:.2f}")

    # Visualize results
    print("\n4. Generating visualizations...")
    plot_bandit_performance(results)
    plot_arm_selections(results, n_arms)
    plot_exploration_exploitation(results, true_rewards)
    plot_regret_analysis(results)

    # Contextual bandits
    print("\n5. Testing contextual bandit (LinUCB)...")
    n_features = 5
    true_rewards_ctx, contexts, true_theta = generate_bandit_data(
        n_arms=n_arms, n_rounds=n_rounds, contextual=True
    )

    linucb = LinUCB(n_arms=n_arms, n_features=n_features, alpha=1.0)
    linucb_result = simulate_bandit(linucb, true_rewards_ctx, n_rounds,
                                   contexts=contexts, true_theta=true_theta)

    print(f"   LinUCB total reward: {linucb_result['cumulative_reward']:.2f}")
    print(f"   LinUCB final regret: {linucb_result['regrets'][-1]:.2f}")

    plot_contextual_performance(linucb_result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nNon-Contextual Bandit Performance:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Total Reward: {result['cumulative_reward']:.2f}")
        print(f"  Final Regret: {result['regrets'][-1]:.2f}")
        optimal_arm = np.argmax(true_rewards)
        optimal_pct = np.mean(np.array(result['arm_selections']) == optimal_arm) * 100
        print(f"  Optimal Arm Selection: {optimal_pct:.2f}%")

    print("\nContextual Bandit Performance (LinUCB):")
    print(f"  Total Reward: {linucb_result['cumulative_reward']:.2f}")
    print(f"  Final Regret: {linucb_result['regrets'][-1]:.2f}")

    # Best algorithm
    best_algo = min(results.items(), key=lambda x: x[1]['regrets'][-1])
    print(f"\nBest algorithm (lowest regret): {best_algo[0]}")
    print(f"  Final regret: {best_algo[1]['regrets'][-1]:.2f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
