"""
Bayesian Optimization - 貝葉斯優化
使用貝葉斯優化進行超參數調優和黑盒函數優化

數據集: 模擬優化問題
難度: ⭐⭐⭐ 高級
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class BayesianOptimizer:
    """貝葉斯優化器"""

    def __init__(self, objective_function, bounds, n_initial=5, kernel=None):
        """
        初始化貝葉斯優化器

        Parameters:
        -----------
        objective_function : callable
            目標函數（我們想要最大化）
        bounds : list of tuples
            每個參數的搜索範圍 [(low1, high1), (low2, high2), ...]
        n_initial : int
            初始隨機採樣點數量
        kernel : sklearn kernel
            高斯過程的核函數
        """
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.n_initial = n_initial
        self.n_params = len(bounds)

        # 默認核函數
        if kernel is None:
            self.kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5)
        else:
            self.kernel = kernel

        # 高斯過程回歸器
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )

        # 存儲觀測
        self.X_observed = []
        self.y_observed = []
        self.iteration_history = []

    def _random_sample(self, n=1):
        """在搜索空間中隨機採樣"""
        samples = []
        for _ in range(n):
            sample = np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1]
            )
            samples.append(sample)
        return np.array(samples)

    def acquisition_EI(self, X, xi=0.01):
        """
        期望改進（Expected Improvement）採集函數

        Parameters:
        -----------
        X : np.ndarray
            候選點
        xi : float
            探索-利用權衡參數
        """
        X = np.atleast_2d(X)

        # 當前最優值
        y_max = np.max(self.y_observed)

        # GP 預測
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)

        # 計算 EI
        with np.errstate(divide='warn'):
            improvement = mu - y_max - xi
            Z = improvement / sigma
            ei = improvement * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei.flatten()

    def acquisition_UCB(self, X, kappa=2.0):
        """
        上置信界（Upper Confidence Bound）採集函數

        Parameters:
        -----------
        X : np.ndarray
            候選點
        kappa : float
            探索參數
        """
        X = np.atleast_2d(X)
        mu, sigma = self.gp.predict(X, return_std=True)
        return (mu + kappa * sigma).flatten()

    def acquisition_PI(self, X, xi=0.01):
        """
        改進概率（Probability of Improvement）採集函數

        Parameters:
        -----------
        X : np.ndarray
            候選點
        xi : float
            探索參數
        """
        X = np.atleast_2d(X)

        y_max = np.max(self.y_observed)
        mu, sigma = self.gp.predict(X, return_std=True)

        with np.errstate(divide='warn'):
            Z = (mu - y_max - xi) / sigma
            pi = stats.norm.cdf(Z)
            pi[sigma == 0.0] = 0.0

        return pi.flatten()

    def propose_location(self, acquisition='EI', n_restarts=25):
        """
        提議下一個採樣點

        Parameters:
        -----------
        acquisition : str
            採集函數類型 ('EI', 'UCB', 'PI')
        n_restarts : int
            優化重啟次數
        """
        # 選擇採集函數
        if acquisition == 'EI':
            acq_func = lambda x: -self.acquisition_EI(x)
        elif acquisition == 'UCB':
            acq_func = lambda x: -self.acquisition_UCB(x)
        elif acquisition == 'PI':
            acq_func = lambda x: -self.acquisition_PI(x)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition}")

        # 多次重啟優化
        min_val = float('inf')
        min_x = None

        for _ in range(n_restarts):
            # 隨機初始點
            x0 = self._random_sample(1)[0]

            # 優化採集函數
            result = minimize(
                acq_func,
                x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )

            if result.fun < min_val:
                min_val = result.fun
                min_x = result.x

        return min_x

    def optimize(self, n_iterations=20, acquisition='EI', verbose=True):
        """
        執行貝葉斯優化

        Parameters:
        -----------
        n_iterations : int
            優化迭代次數
        acquisition : str
            採集函數類型
        verbose : bool
            是否打印進度
        """
        if verbose:
            print("=" * 60)
            print("開始貝葉斯優化")
            print("=" * 60)
            print(f"迭代次數: {n_iterations}")
            print(f"採集函數: {acquisition}")
            print()

        # 初始隨機採樣
        X_init = self._random_sample(self.n_initial)
        y_init = np.array([self.objective_function(x) for x in X_init])

        self.X_observed = X_init.tolist()
        self.y_observed = y_init.tolist()

        # 貝葉斯優化迭代
        for i in range(n_iterations):
            # 擬合 GP
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))

            # 提議下一個點
            X_next = self.propose_location(acquisition=acquisition)

            # 評估目標函數
            y_next = self.objective_function(X_next)

            # 添加到觀測
            self.X_observed.append(X_next)
            self.y_observed.append(y_next)

            # 記錄當前最優
            current_best = np.max(self.y_observed)
            self.iteration_history.append(current_best)

            if verbose:
                print(f"迭代 {i+1}/{n_iterations}: "
                      f"f(x) = {y_next:.6f}, "
                      f"最優 = {current_best:.6f}")

        # 最終結果
        best_idx = np.argmax(self.y_observed)
        best_x = np.array(self.X_observed)[best_idx]
        best_y = self.y_observed[best_idx]

        if verbose:
            print("\n" + "=" * 60)
            print("優化完成！")
            print("=" * 60)
            print(f"最優參數: {best_x}")
            print(f"最優值: {best_y:.6f}")
            print()

        return best_x, best_y

    def plot_optimization_1d(self):
        """繪製 1D 優化過程（僅適用於 1 維問題）"""
        if self.n_params != 1:
            print("此可視化僅適用於 1 維問題")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 創建密集網格
        X_grid = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 1000).reshape(-1, 1)

        # 真實函數
        y_true = np.array([self.objective_function(x) for x in X_grid])

        # GP 預測
        mu, sigma = self.gp.predict(X_grid, return_std=True)

        # 1. 目標函數和 GP 預測
        ax = axes[0]
        ax.plot(X_grid, y_true, 'r-', label='真實函數', linewidth=2)
        ax.plot(X_grid, mu, 'b-', label='GP 均值', linewidth=2)
        ax.fill_between(X_grid.flatten(),
                        mu - 1.96 * sigma,
                        mu + 1.96 * sigma,
                        alpha=0.3, label='95% 置信區間')

        # 觀測點
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed)
        ax.scatter(X_obs, y_obs, c='red', s=100, zorder=10,
                  edgecolors='black', label='觀測點')

        # 最優點
        best_idx = np.argmax(y_obs)
        ax.scatter(X_obs[best_idx], y_obs[best_idx],
                  c='gold', s=200, marker='*', zorder=11,
                  edgecolors='black', label='最優點')

        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('貝葉斯優化：目標函數與高斯過程')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 採集函數
        ax = axes[1]
        ei = self.acquisition_EI(X_grid)
        ax.plot(X_grid, ei, 'g-', label='期望改進 (EI)', linewidth=2)
        ax.fill_between(X_grid.flatten(), 0, ei, alpha=0.3)

        # 下一個建議點
        next_x = self.propose_location(acquisition='EI')
        ax.axvline(next_x, color='purple', linestyle='--',
                  linewidth=2, label=f'下一個採樣點')

        ax.set_xlabel('x')
        ax.set_ylabel('採集函數值')
        ax.set_title('期望改進採集函數')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('bayesian_optimization_1d.png', dpi=300, bbox_inches='tight')
        print("1D 優化圖表已保存: bayesian_optimization_1d.png")
        plt.show()

    def plot_convergence(self):
        """繪製收斂曲線"""
        plt.figure(figsize=(12, 5))

        # 繪製每次迭代的最優值
        iterations = range(1, len(self.iteration_history) + 1)
        plt.plot(iterations, self.iteration_history,
                'o-', linewidth=2, markersize=8)

        plt.xlabel('迭代次數')
        plt.ylabel('當前最優值')
        plt.title('貝葉斯優化收斂曲線')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('convergence_curve.png', dpi=300, bbox_inches='tight')
        print("收斂曲線已保存: convergence_curve.png")
        plt.show()


def objective_1d(x):
    """1D 測試函數（有多個局部最優）"""
    x = x[0] if isinstance(x, np.ndarray) else x
    return -(x - 2)**2 + 5 + 3 * np.sin(3 * x) + np.sin(5 * x)


def objective_2d(x):
    """2D 測試函數（Branin 函數，取負使其為最大化問題）"""
    x1, x2 = x[0], x[1]
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    term3 = s

    return -(term1 + term2 + term3)  # 取負以最大化


def hyperparameter_tuning_example():
    """超參數調優示例：優化隨機森林"""
    print("=" * 60)
    print("示例：使用貝葉斯優化調優隨機森林超參數")
    print("=" * 60)

    # 創建數據集
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    # 定義目標函數（交叉驗證分數）
    def rf_objective(params):
        n_estimators = int(params[0])
        max_depth = int(params[1])
        min_samples_split = int(params[2])

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )

        score = cross_val_score(rf, X, y, cv=3, scoring='accuracy').mean()
        return score

    # 定義搜索空間
    bounds = [
        (10, 200),    # n_estimators
        (3, 20),      # max_depth
        (2, 20)       # min_samples_split
    ]

    # 貝葉斯優化
    optimizer = BayesianOptimizer(
        objective_function=rf_objective,
        bounds=bounds,
        n_initial=5
    )

    best_params, best_score = optimizer.optimize(
        n_iterations=15,
        acquisition='EI'
    )

    print(f"最優超參數:")
    print(f"  n_estimators: {int(best_params[0])}")
    print(f"  max_depth: {int(best_params[1])}")
    print(f"  min_samples_split: {int(best_params[2])}")
    print(f"  交叉驗證分數: {best_score:.4f}")

    # 繪製收斂曲線
    optimizer.plot_convergence()


def main():
    """主函數"""
    print("開始貝葉斯優化演示...")
    print()

    # 1. 1D 優化
    print("=" * 60)
    print("示例 1: 一維函數優化")
    print("=" * 60)
    print()

    optimizer_1d = BayesianOptimizer(
        objective_function=objective_1d,
        bounds=[(-5, 5)],
        n_initial=3
    )

    best_x_1d, best_y_1d = optimizer_1d.optimize(
        n_iterations=15,
        acquisition='EI'
    )

    optimizer_1d.plot_optimization_1d()
    optimizer_1d.plot_convergence()

    # 2. 2D 優化
    print("\n" + "=" * 60)
    print("示例 2: 二維函數優化（Branin 函數）")
    print("=" * 60)
    print()

    optimizer_2d = BayesianOptimizer(
        objective_function=objective_2d,
        bounds=[(-5, 10), (0, 15)],
        n_initial=5
    )

    best_x_2d, best_y_2d = optimizer_2d.optimize(
        n_iterations=20,
        acquisition='EI'
    )

    optimizer_2d.plot_convergence()

    # 3. 超參數調優
    print("\n")
    hyperparameter_tuning_example()

    print("\n" + "=" * 60)
    print("所有優化完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
