"""
Probabilistic Programming - 概率編程
使用純 NumPy/SciPy 實現概率編程的基本概念

數據集: 模擬概率模型
難度: ⭐⭐⭐ 高級
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass
from typing import Callable, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class RandomVariable:
    """隨機變量類"""

    def __init__(self, name, distribution, **params):
        """
        初始化隨機變量

        Parameters:
        -----------
        name : str
            變量名稱
        distribution : str
            分佈類型 ('normal', 'bernoulli', 'beta', etc.)
        **params : dict
            分佈參數
        """
        self.name = name
        self.distribution = distribution
        self.params = params
        self.value = None

    def sample(self, size=1):
        """從分佈中採樣"""
        if self.distribution == 'normal':
            return np.random.normal(
                self.params['mean'],
                self.params['std'],
                size
            )
        elif self.distribution == 'bernoulli':
            return np.random.binomial(1, self.params['p'], size)
        elif self.distribution == 'beta':
            return np.random.beta(
                self.params['alpha'],
                self.params['beta'],
                size
            )
        elif self.distribution == 'gamma':
            return np.random.gamma(
                self.params['shape'],
                self.params['scale'],
                size
            )
        elif self.distribution == 'uniform':
            return np.random.uniform(
                self.params['low'],
                self.params['high'],
                size
            )
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def log_prob(self, value):
        """計算對數概率"""
        if self.distribution == 'normal':
            return stats.norm.logpdf(
                value,
                self.params['mean'],
                self.params['std']
            )
        elif self.distribution == 'bernoulli':
            p = self.params['p']
            return value * np.log(p) + (1 - value) * np.log(1 - p)
        elif self.distribution == 'beta':
            return stats.beta.logpdf(
                value,
                self.params['alpha'],
                self.params['beta']
            )
        elif self.distribution == 'gamma':
            return stats.gamma.logpdf(
                value,
                self.params['shape'],
                scale=self.params['scale']
            )
        elif self.distribution == 'uniform':
            return stats.uniform.logpdf(
                value,
                self.params['low'],
                self.params['high'] - self.params['low']
            )


class ProbabilisticModel:
    """概率模型類"""

    def __init__(self):
        self.variables = {}
        self.observations = {}

    def add_variable(self, rv: RandomVariable):
        """添加隨機變量"""
        self.variables[rv.name] = rv

    def observe(self, var_name, value):
        """觀測變量"""
        self.observations[var_name] = value

    def log_likelihood(self, params):
        """計算對數似然"""
        log_lik = 0

        for var_name, value in self.observations.items():
            if var_name in self.variables:
                rv = self.variables[var_name]
                # 更新參數
                for param_name, param_value in params.items():
                    if param_name in rv.params:
                        rv.params[param_name] = param_value

                log_lik += np.sum(rv.log_prob(value))

        return log_lik

    def log_prior(self, params):
        """計算對數先驗"""
        log_prior = 0

        for param_name, param_value in params.items():
            # 假設參數有弱信息先驗
            if 'mean' in param_name or 'mu' in param_name:
                log_prior += stats.norm.logpdf(param_value, 0, 10)
            elif 'std' in param_name or 'sigma' in param_name:
                log_prior += stats.gamma.logpdf(param_value, 2, scale=1)
            elif 'alpha' in param_name or 'beta' in param_name:
                log_prior += stats.gamma.logpdf(param_value, 1, scale=1)

        return log_prior

    def log_posterior(self, params):
        """計算對數後驗（未歸一化）"""
        return self.log_prior(params) + self.log_likelihood(params)


class BayesianLinearModel:
    """貝葉斯線性回歸模型"""

    def __init__(self, prior_mean=0, prior_std=10, noise_std=1):
        """
        初始化貝葉斯線性模型

        Parameters:
        -----------
        prior_mean : float
            權重先驗均值
        prior_std : float
            權重先驗標準差
        noise_std : float
            觀測噪聲標準差
        """
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.noise_std = noise_std
        self.posterior_samples = None

    def fit(self, X, y, n_samples=1000):
        """
        使用簡單的 MCMC 採樣擬合模型

        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣
        y : np.ndarray
            目標值
        n_samples : int
            採樣數量
        """
        n_features = X.shape[1]

        # 定義對數後驗
        def log_posterior(params):
            weights = params[:n_features]
            sigma = params[n_features]

            # 先驗
            log_prior = (
                np.sum(stats.norm.logpdf(weights, self.prior_mean, self.prior_std)) +
                stats.gamma.logpdf(1/sigma**2, 2, scale=1)  # 精度的 Gamma 先驗
            )

            # 似然
            y_pred = X @ weights
            log_likelihood = np.sum(stats.norm.logpdf(y, y_pred, sigma))

            return log_prior + log_likelihood

        # Metropolis-Hastings 採樣
        current_params = np.concatenate([
            np.zeros(n_features),
            [self.noise_std]
        ])

        samples = []
        proposal_std = 0.1

        for i in range(n_samples + 1000):  # 包括燒入期
            # 提議新參數
            proposed_params = current_params + np.random.normal(
                0, proposal_std, size=current_params.shape
            )

            # 確保 sigma > 0
            if proposed_params[-1] <= 0:
                continue

            # 接受準則
            log_alpha = log_posterior(proposed_params) - log_posterior(current_params)

            if np.log(np.random.uniform()) < log_alpha:
                current_params = proposed_params

            # 燒入期後保存
            if i >= 1000:
                samples.append(current_params.copy())

        self.posterior_samples = np.array(samples)

        print(f"貝葉斯線性模型擬合完成")
        print(f"後驗樣本數: {len(self.posterior_samples)}")

    def predict(self, X, return_samples=False):
        """
        預測

        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣
        return_samples : bool
            是否返回所有後驗預測樣本

        Returns:
        --------
        y_pred : np.ndarray
            預測均值
        y_std : np.ndarray
            預測標準差
        """
        if self.posterior_samples is None:
            raise ValueError("Model not fitted yet")

        n_features = X.shape[1]
        predictions = []

        for sample in self.posterior_samples:
            weights = sample[:n_features]
            sigma = sample[n_features]

            y_pred = X @ weights
            # 添加觀測噪聲
            y_pred_noisy = y_pred + np.random.normal(0, sigma, size=y_pred.shape)
            predictions.append(y_pred_noisy)

        predictions = np.array(predictions)

        if return_samples:
            return predictions
        else:
            return predictions.mean(axis=0), predictions.std(axis=0)


class BayesianMixtureModel:
    """貝葉斯混合模型（簡化版）"""

    def __init__(self, n_components=2):
        """
        初始化貝葉斯混合模型

        Parameters:
        -----------
        n_components : int
            混合成分數量
        """
        self.n_components = n_components
        self.weights = None
        self.means = None
        self.stds = None

    def fit(self, X, n_iterations=100):
        """
        使用變分EM近似擬合模型

        Parameters:
        -----------
        X : np.ndarray
            數據
        n_iterations : int
            迭代次數
        """
        n_samples = len(X)

        # 初始化參數
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = np.random.choice(X, self.n_components)
        self.stds = np.ones(self.n_components)

        for iteration in range(n_iterations):
            # E 步：計算責任
            responsibilities = np.zeros((n_samples, self.n_components))

            for k in range(self.n_components):
                responsibilities[:, k] = (
                    self.weights[k] *
                    stats.norm.pdf(X, self.means[k], self.stds[k])
                )

            responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

            # M 步：更新參數
            Nk = responsibilities.sum(axis=0)

            self.weights = Nk / n_samples

            for k in range(self.n_components):
                self.means[k] = (responsibilities[:, k] @ X) / Nk[k]
                self.stds[k] = np.sqrt(
                    (responsibilities[:, k] @ (X - self.means[k])**2) / Nk[k]
                )

        print(f"貝葉斯混合模型擬合完成")
        print(f"混合權重: {self.weights}")
        print(f"混合均值: {self.means}")
        print(f"混合標準差: {self.stds}")

    def predict_proba(self, X):
        """預測每個成分的概率"""
        n_samples = len(X)
        probabilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            probabilities[:, k] = (
                self.weights[k] *
                stats.norm.pdf(X, self.means[k], self.stds[k])
            )

        return probabilities / probabilities.sum(axis=1, keepdims=True)


def example_coin_flip():
    """示例：硬幣投擲的貝葉斯推斷"""
    print("=" * 60)
    print("示例 1: 硬幣投擲的貝葉斯推斷")
    print("=" * 60)

    # 模擬數據
    true_p = 0.7
    n_flips = 100
    flips = np.random.binomial(1, true_p, n_flips)

    print(f"真實成功率: {true_p}")
    print(f"觀測到的成功次數: {flips.sum()}/{n_flips}")

    # Beta-Binomial 共軛更新
    alpha_prior = 1
    beta_prior = 1

    alpha_post = alpha_prior + flips.sum()
    beta_post = beta_prior + n_flips - flips.sum()

    print(f"\n先驗: Beta({alpha_prior}, {beta_prior})")
    print(f"後驗: Beta({alpha_post}, {beta_post})")

    # 繪製先驗和後驗
    plt.figure(figsize=(10, 6))

    p_range = np.linspace(0, 1, 1000)
    prior = stats.beta.pdf(p_range, alpha_prior, beta_prior)
    posterior = stats.beta.pdf(p_range, alpha_post, beta_post)

    plt.plot(p_range, prior, label='先驗分佈', linewidth=2)
    plt.plot(p_range, posterior, label='後驗分佈', linewidth=2)
    plt.axvline(true_p, color='red', linestyle='--',
               label=f'真實值 = {true_p}', linewidth=2)

    post_mean = alpha_post / (alpha_post + beta_post)
    plt.axvline(post_mean, color='green', linestyle='--',
               label=f'後驗均值 = {post_mean:.3f}', linewidth=2)

    plt.xlabel('成功率 p')
    plt.ylabel('概率密度')
    plt.title('硬幣投擲的貝葉斯推斷')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('coin_flip_bayesian.png', dpi=300, bbox_inches='tight')
    print("圖表已保存: coin_flip_bayesian.png")
    plt.show()


def example_linear_regression():
    """示例：貝葉斯線性回歸"""
    print("\n" + "=" * 60)
    print("示例 2: 貝葉斯線性回歸")
    print("=" * 60)

    # 生成數據
    np.random.seed(42)
    n_samples = 50
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    X_with_intercept = np.column_stack([np.ones(n_samples), X])

    true_weights = np.array([2.0, 0.5])
    y = X_with_intercept @ true_weights + np.random.normal(0, 0.5, n_samples)

    print(f"真實權重: {true_weights}")

    # 擬合貝葉斯線性模型
    model = BayesianLinearModel(prior_mean=0, prior_std=10, noise_std=1)
    model.fit(X_with_intercept, y, n_samples=2000)

    # 後驗統計
    weights_posterior = model.posterior_samples[:, :2]
    sigma_posterior = model.posterior_samples[:, 2]

    print(f"後驗權重均值: {weights_posterior.mean(axis=0)}")
    print(f"後驗權重標準差: {weights_posterior.std(axis=0)}")
    print(f"後驗噪聲標準差均值: {sigma_posterior.mean():.4f}")

    # 預測
    X_test = np.linspace(0, 12, 100).reshape(-1, 1)
    X_test_with_intercept = np.column_stack([np.ones(100), X_test])
    y_pred, y_std = model.predict(X_test_with_intercept)

    # 繪製結果
    plt.figure(figsize=(12, 6))

    plt.scatter(X, y, alpha=0.6, s=50, label='觀測數據')
    plt.plot(X_test, y_pred, 'r-', linewidth=2, label='預測均值')
    plt.fill_between(X_test.flatten(),
                     y_pred - 2*y_std,
                     y_pred + 2*y_std,
                     alpha=0.3, label='95% 預測區間')

    # 真實函數
    y_true = X_test_with_intercept @ true_weights
    plt.plot(X_test, y_true, 'g--', linewidth=2, label='真實函數')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('貝葉斯線性回歸')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bayesian_linear_regression.png', dpi=300, bbox_inches='tight')
    print("圖表已保存: bayesian_linear_regression.png")
    plt.show()


def example_mixture_model():
    """示例：貝葉斯混合模型"""
    print("\n" + "=" * 60)
    print("示例 3: 貝葉斯混合模型")
    print("=" * 60)

    # 生成混合數據
    np.random.seed(42)
    n_samples = 300

    # 第一個成分
    X1 = np.random.normal(-2, 0.8, n_samples // 2)
    # 第二個成分
    X2 = np.random.normal(2, 1.2, n_samples // 2)

    X = np.concatenate([X1, X2])
    np.random.shuffle(X)

    # 擬合混合模型
    model = BayesianMixtureModel(n_components=2)
    model.fit(X, n_iterations=100)

    # 繪製結果
    plt.figure(figsize=(12, 6))

    # 數據直方圖
    plt.hist(X, bins=50, density=True, alpha=0.5,
            edgecolor='black', label='數據')

    # 擬合的混合分佈
    x_range = np.linspace(X.min(), X.max(), 1000)
    mixture_pdf = np.zeros_like(x_range)

    for k in range(model.n_components):
        component_pdf = (
            model.weights[k] *
            stats.norm.pdf(x_range, model.means[k], model.stds[k])
        )
        mixture_pdf += component_pdf
        plt.plot(x_range, component_pdf, '--',
                label=f'成分 {k+1}', linewidth=2)

    plt.plot(x_range, mixture_pdf, 'r-',
            label='混合分佈', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('概率密度')
    plt.title('貝葉斯混合模型')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bayesian_mixture_model.png', dpi=300, bbox_inches='tight')
    print("圖表已保存: bayesian_mixture_model.png")
    plt.show()


def main():
    """主函數"""
    print("開始概率編程演示...")
    print()

    # 示例 1: 硬幣投擲
    example_coin_flip()

    # 示例 2: 貝葉斯線性回歸
    example_linear_regression()

    # 示例 3: 貝葉斯混合模型
    example_mixture_model()

    print("\n" + "=" * 60)
    print("概率編程演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
