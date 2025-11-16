"""
Markov Chain Monte Carlo (MCMC) - 馬可夫鏈蒙特卡洛
實現多種 MCMC 採樣算法進行貝葉斯推斷

數據集: 模擬貝葉斯推斷問題
難度: ⭐⭐⭐ 高級
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MetropolisHastings:
    """Metropolis-Hastings 採樣器"""

    def __init__(self, target_log_prob, proposal_std=1.0):
        """
        初始化 Metropolis-Hastings 採樣器

        Parameters:
        -----------
        target_log_prob : callable
            目標分佈的對數概率函數
        proposal_std : float
            提議分佈的標準差
        """
        self.target_log_prob = target_log_prob
        self.proposal_std = proposal_std
        self.samples = []
        self.acceptance_rate = 0

    def sample(self, n_samples, initial_state, burn_in=1000, thin=1):
        """
        執行 Metropolis-Hastings 採樣

        Parameters:
        -----------
        n_samples : int
            採樣數量
        initial_state : np.ndarray
            初始狀態
        burn_in : int
            燒入期樣本數
        thin : int
            抽稀間隔

        Returns:
        --------
        samples : np.ndarray
            採樣結果
        """
        current_state = np.array(initial_state)
        current_log_prob = self.target_log_prob(current_state)

        samples = []
        n_accepted = 0
        total_iterations = burn_in + n_samples * thin

        for i in range(total_iterations):
            # 提議新狀態（高斯隨機遊走）
            proposed_state = current_state + np.random.normal(
                0, self.proposal_std, size=current_state.shape
            )

            # 計算提議狀態的對數概率
            proposed_log_prob = self.target_log_prob(proposed_state)

            # Metropolis-Hastings 接受準則
            log_alpha = proposed_log_prob - current_log_prob

            if np.log(np.random.uniform()) < log_alpha:
                # 接受提議
                current_state = proposed_state
                current_log_prob = proposed_log_prob
                n_accepted += 1

            # 燒入期後，按抽稀間隔保存樣本
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(current_state.copy())

        self.samples = np.array(samples)
        self.acceptance_rate = n_accepted / total_iterations

        print(f"Metropolis-Hastings 採樣完成")
        print(f"接受率: {self.acceptance_rate:.4f}")

        return self.samples


class GibbsSampler:
    """Gibbs 採樣器（用於多元正態分佈示例）"""

    def __init__(self, mean, cov):
        """
        初始化 Gibbs 採樣器（二元正態分佈）

        Parameters:
        -----------
        mean : np.ndarray
            均值向量
        cov : np.ndarray
            協方差矩陣
        """
        self.mean = mean
        self.cov = cov
        self.samples = []

    def conditional_distribution(self, x_other, dim):
        """
        計算條件分佈參數

        Parameters:
        -----------
        x_other : float
            另一個維度的值
        dim : int
            當前維度 (0 或 1)
        """
        other_dim = 1 - dim

        # 條件均值
        cond_mean = (
            self.mean[dim] +
            self.cov[dim, other_dim] / self.cov[other_dim, other_dim] *
            (x_other - self.mean[other_dim])
        )

        # 條件方差
        cond_var = (
            self.cov[dim, dim] -
            self.cov[dim, other_dim]**2 / self.cov[other_dim, other_dim]
        )

        return cond_mean, np.sqrt(cond_var)

    def sample(self, n_samples, initial_state, burn_in=1000):
        """
        執行 Gibbs 採樣

        Parameters:
        -----------
        n_samples : int
            採樣數量
        initial_state : np.ndarray
            初始狀態
        burn_in : int
            燒入期樣本數
        """
        current_state = np.array(initial_state)
        samples = []

        for i in range(burn_in + n_samples):
            # 交替採樣每個維度
            for dim in range(2):
                other_dim = 1 - dim
                cond_mean, cond_std = self.conditional_distribution(
                    current_state[other_dim], dim
                )
                current_state[dim] = np.random.normal(cond_mean, cond_std)

            # 燒入期後保存樣本
            if i >= burn_in:
                samples.append(current_state.copy())

        self.samples = np.array(samples)
        print(f"Gibbs 採樣完成")

        return self.samples


class HamiltonianMonteCarlo:
    """Hamiltonian Monte Carlo (HMC) 採樣器"""

    def __init__(self, target_log_prob, gradient_log_prob, step_size=0.1, n_leapfrog=10):
        """
        初始化 HMC 採樣器

        Parameters:
        -----------
        target_log_prob : callable
            目標分佈的對數概率
        gradient_log_prob : callable
            對數概率的梯度
        step_size : float
            Leapfrog 積分步長
        n_leapfrog : int
            Leapfrog 步數
        """
        self.target_log_prob = target_log_prob
        self.gradient_log_prob = gradient_log_prob
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.samples = []
        self.acceptance_rate = 0

    def leapfrog(self, q, p):
        """
        Leapfrog 積分

        Parameters:
        -----------
        q : np.ndarray
            位置
        p : np.ndarray
            動量
        """
        q = q.copy()
        p = p.copy()

        # 半步動量更新
        p = p + 0.5 * self.step_size * self.gradient_log_prob(q)

        # 完整步位置和動量更新
        for _ in range(self.n_leapfrog - 1):
            q = q + self.step_size * p
            p = p + self.step_size * self.gradient_log_prob(q)

        # 最後一步
        q = q + self.step_size * p

        # 半步動量更新
        p = p + 0.5 * self.step_size * self.gradient_log_prob(q)

        return q, -p  # 翻轉動量以保證可逆性

    def sample(self, n_samples, initial_state, burn_in=1000):
        """
        執行 HMC 採樣

        Parameters:
        -----------
        n_samples : int
            採樣數量
        initial_state : np.ndarray
            初始狀態
        burn_in : int
            燒入期
        """
        current_q = np.array(initial_state)
        samples = []
        n_accepted = 0
        total_iterations = burn_in + n_samples

        for i in range(total_iterations):
            # 採樣動量
            current_p = np.random.normal(0, 1, size=current_q.shape)

            # 當前 Hamiltonian
            current_H = -self.target_log_prob(current_q) + 0.5 * np.sum(current_p**2)

            # Leapfrog 積分
            proposed_q, proposed_p = self.leapfrog(current_q, current_p)

            # 提議的 Hamiltonian
            proposed_H = -self.target_log_prob(proposed_q) + 0.5 * np.sum(proposed_p**2)

            # Metropolis 接受準則
            if np.log(np.random.uniform()) < current_H - proposed_H:
                current_q = proposed_q
                n_accepted += 1

            # 燒入期後保存樣本
            if i >= burn_in:
                samples.append(current_q.copy())

        self.samples = np.array(samples)
        self.acceptance_rate = n_accepted / total_iterations

        print(f"HMC 採樣完成")
        print(f"接受率: {self.acceptance_rate:.4f}")

        return self.samples


def analyze_samples(samples, true_mean=None, true_cov=None, method_name=""):
    """分析採樣結果"""
    print(f"\n{method_name} 採樣統計")
    print("=" * 60)

    # 計算統計量
    sample_mean = samples.mean(axis=0)
    sample_cov = np.cov(samples.T)

    print(f"樣本均值: {sample_mean}")
    print(f"樣本協方差:\n{sample_cov}")

    if true_mean is not None:
        print(f"\n真實均值: {true_mean}")
        print(f"均值誤差: {np.linalg.norm(sample_mean - true_mean):.6f}")

    if true_cov is not None:
        print(f"\n真實協方差:\n{true_cov}")
        print(f"協方差誤差: {np.linalg.norm(sample_cov - true_cov):.6f}")


def plot_mcmc_diagnostics(samplers_dict, true_mean=None, true_cov=None):
    """繪製 MCMC 診斷圖"""
    n_methods = len(samplers_dict)
    fig, axes = plt.subplots(n_methods, 3, figsize=(15, 5*n_methods))

    if n_methods == 1:
        axes = axes.reshape(1, -1)

    for idx, (method_name, samples) in enumerate(samplers_dict.items()):
        # 1. 軌跡圖（Trace Plot）
        ax = axes[idx, 0]
        for dim in range(samples.shape[1]):
            ax.plot(samples[:, dim], alpha=0.7, label=f'維度 {dim}')

        if true_mean is not None:
            for dim in range(len(true_mean)):
                ax.axhline(true_mean[dim], color=f'C{dim}',
                          linestyle='--', linewidth=2, alpha=0.5)

        ax.set_xlabel('迭代次數')
        ax.set_ylabel('參數值')
        ax.set_title(f'{method_name} - 軌跡圖')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 散點圖（僅適用於 2D）
        ax = axes[idx, 1]
        if samples.shape[1] == 2:
            ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10)

            if true_mean is not None:
                ax.scatter(true_mean[0], true_mean[1],
                          c='red', s=200, marker='*',
                          edgecolors='black', label='真實均值')

            # 繪製真實分佈的等高線
            if true_mean is not None and true_cov is not None:
                x_range = np.linspace(samples[:, 0].min(), samples[:, 0].max(), 100)
                y_range = np.linspace(samples[:, 1].min(), samples[:, 1].max(), 100)
                X, Y = np.meshgrid(x_range, y_range)

                pos = np.dstack((X, Y))
                rv = stats.multivariate_normal(true_mean, true_cov)
                Z = rv.pdf(pos)

                ax.contour(X, Y, Z, levels=5, colors='red',
                          alpha=0.5, linewidths=2)

            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title(f'{method_name} - 樣本分佈')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '僅適用於 2D',
                   ha='center', va='center', transform=ax.transAxes)

        # 3. 自相關圖
        ax = axes[idx, 2]
        if samples.shape[1] > 0:
            # 計算第一維度的自相關
            series = samples[:, 0]
            max_lag = min(100, len(series) // 2)
            autocorr = [1.0]

            for lag in range(1, max_lag):
                corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                autocorr.append(corr)

            ax.plot(autocorr, linewidth=2)
            ax.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax.axhline(0.1, color='red', linestyle='--', alpha=0.3)
            ax.axhline(-0.1, color='red', linestyle='--', alpha=0.3)

            ax.set_xlabel('滯後（Lag）')
            ax.set_ylabel('自相關')
            ax.set_title(f'{method_name} - 自相關（維度 0）')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mcmc_diagnostics.png', dpi=300, bbox_inches='tight')
    print("\nMCMC 診斷圖表已保存: mcmc_diagnostics.png")
    plt.show()


def main():
    """主函數"""
    print("開始 MCMC 採樣演示...")
    print()

    # 定義目標分佈：二元正態分佈
    true_mean = np.array([2.0, -1.0])
    true_cov = np.array([[1.0, 0.8],
                         [0.8, 2.0]])

    # 目標分佈的對數概率
    def target_log_prob(x):
        return stats.multivariate_normal.logpdf(x, true_mean, true_cov)

    # 對數概率的梯度
    def gradient_log_prob(x):
        inv_cov = np.linalg.inv(true_cov)
        return -inv_cov @ (x - true_mean)

    # 初始狀態
    initial_state = np.array([0.0, 0.0])

    print("=" * 60)
    print("目標分佈：二元正態分佈")
    print("=" * 60)
    print(f"真實均值: {true_mean}")
    print(f"真實協方差:\n{true_cov}")
    print()

    # 1. Metropolis-Hastings
    print("=" * 60)
    print("1. Metropolis-Hastings 採樣")
    print("=" * 60)
    mh = MetropolisHastings(target_log_prob, proposal_std=0.5)
    mh_samples = mh.sample(
        n_samples=5000,
        initial_state=initial_state,
        burn_in=1000,
        thin=1
    )
    analyze_samples(mh_samples, true_mean, true_cov, "Metropolis-Hastings")

    # 2. Gibbs Sampling
    print("\n" + "=" * 60)
    print("2. Gibbs 採樣")
    print("=" * 60)
    gibbs = GibbsSampler(true_mean, true_cov)
    gibbs_samples = gibbs.sample(
        n_samples=5000,
        initial_state=initial_state,
        burn_in=1000
    )
    analyze_samples(gibbs_samples, true_mean, true_cov, "Gibbs")

    # 3. Hamiltonian Monte Carlo
    print("\n" + "=" * 60)
    print("3. Hamiltonian Monte Carlo (HMC)")
    print("=" * 60)
    hmc = HamiltonianMonteCarlo(
        target_log_prob,
        gradient_log_prob,
        step_size=0.1,
        n_leapfrog=20
    )
    hmc_samples = hmc.sample(
        n_samples=5000,
        initial_state=initial_state,
        burn_in=1000
    )
    analyze_samples(hmc_samples, true_mean, true_cov, "HMC")

    # 繪製診斷圖
    samplers_dict = {
        'Metropolis-Hastings': mh_samples,
        'Gibbs': gibbs_samples,
        'HMC': hmc_samples
    }

    plot_mcmc_diagnostics(samplers_dict, true_mean, true_cov)

    print("\n" + "=" * 60)
    print("MCMC 採樣完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
