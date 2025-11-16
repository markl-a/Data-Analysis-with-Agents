"""
Bayesian Neural Networks - 貝葉斯神經網絡
實現簡單的貝葉斯神經網絡，包含不確定性量化

數據集: 模擬回歸和分類數據
難度: ⭐⭐⭐ 高級
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_moons
import warnings
warnings.filterwarnings('ignore')


class BayesianNeuralNetwork:
    """貝葉斯神經網絡（使用變分推斷）"""

    def __init__(self, layer_sizes, prior_std=1.0, noise_std=0.1):
        """
        初始化貝葉斯神經網絡

        Parameters:
        -----------
        layer_sizes : list
            每層的神經元數量 [input_size, hidden_size, ..., output_size]
        prior_std : float
            權重先驗標準差
        noise_std : float
            似然噪聲標準差
        """
        self.layer_sizes = layer_sizes
        self.prior_std = prior_std
        self.noise_std = noise_std
        self.n_layers = len(layer_sizes) - 1

        # 初始化變分參數（均值和對數標準差）
        self.weight_means = []
        self.weight_log_stds = []

        for i in range(self.n_layers):
            # 權重
            w_mean = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            w_log_std = np.ones((layer_sizes[i], layer_sizes[i+1])) * np.log(0.1)

            self.weight_means.append(w_mean)
            self.weight_log_stds.append(w_log_std)

            # 偏置
            b_mean = np.zeros(layer_sizes[i+1])
            b_log_std = np.ones(layer_sizes[i+1]) * np.log(0.1)

            self.weight_means.append(b_mean)
            self.weight_log_stds.append(b_log_std)

    def relu(self, x):
        """ReLU 激活函數"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ReLU 導數"""
        return (x > 0).astype(float)

    def forward(self, X, weights):
        """
        前向傳播

        Parameters:
        -----------
        X : np.ndarray
            輸入數據
        weights : list
            權重和偏置列表

        Returns:
        --------
        output : np.ndarray
            輸出
        activations : list
            每層的激活值
        """
        activations = [X]

        for i in range(self.n_layers):
            W = weights[2*i]
            b = weights[2*i + 1]

            z = activations[-1] @ W + b

            if i < self.n_layers - 1:
                # 隱藏層使用 ReLU
                a = self.relu(z)
            else:
                # 輸出層線性
                a = z

            activations.append(a)

        return activations[-1], activations

    def sample_weights(self, n_samples=1):
        """從變分後驗採樣權重"""
        weight_samples = []

        for _ in range(n_samples):
            weights = []
            for mean, log_std in zip(self.weight_means, self.weight_log_stds):
                std = np.exp(log_std)
                sample = mean + std * np.random.randn(*mean.shape)
                weights.append(sample)

            weight_samples.append(weights)

        return weight_samples

    def kl_divergence(self):
        """計算 KL 散度（變分後驗 || 先驗）"""
        kl = 0

        for mean, log_std in zip(self.weight_means, self.weight_log_stds):
            std = np.exp(log_std)
            var = std ** 2

            # KL(q(w) || p(w)) for Gaussian
            kl += np.sum(
                0.5 * (
                    var / self.prior_std**2 +
                    mean**2 / self.prior_std**2 -
                    1 -
                    np.log(var / self.prior_std**2)
                )
            )

        return kl

    def elbo(self, X, y, n_samples=1):
        """
        計算 ELBO（證據下界）

        ELBO = E[log p(y|X,w)] - KL(q(w)||p(w))
        """
        # 採樣權重並計算似然
        log_likelihood = 0

        for _ in range(n_samples):
            weights = self.sample_weights(1)[0]
            y_pred, _ = self.forward(X, weights)

            # 高斯似然
            log_likelihood += -0.5 * np.sum(
                (y - y_pred)**2 / self.noise_std**2
            )

        log_likelihood /= n_samples

        # KL 散度
        kl = self.kl_divergence()

        # ELBO
        elbo = log_likelihood - kl / X.shape[0]

        return elbo, log_likelihood, kl

    def fit(self, X, y, n_iterations=1000, learning_rate=0.01, batch_size=32, verbose=True):
        """
        使用變分推斷訓練模型

        Parameters:
        -----------
        X : np.ndarray
            訓練數據
        y : np.ndarray
            訓練標籤
        n_iterations : int
            迭代次數
        learning_rate : float
            學習率
        batch_size : int
            批次大小
        """
        n_samples = X.shape[0]
        elbo_history = []

        for iteration in range(n_iterations):
            # 小批次採樣
            idx = np.random.choice(n_samples, batch_size, replace=False)
            X_batch = X[idx]
            y_batch = y[idx]

            # 計算梯度（使用重參數化技巧）
            weights = self.sample_weights(1)[0]

            # 前向傳播
            y_pred, activations = self.forward(X_batch, weights)

            # 計算損失梯度
            grad_output = (y_pred - y_batch) / (self.noise_std**2 * batch_size)

            # 反向傳播（簡化版）
            grad_weights = []

            for i in range(self.n_layers - 1, -1, -1):
                W = weights[2*i]
                b = weights[2*i + 1]

                # 梯度計算
                grad_W = activations[i].T @ grad_output
                grad_b = np.sum(grad_output, axis=0)

                grad_weights.insert(0, grad_b)
                grad_weights.insert(0, grad_W)

                if i > 0:
                    grad_output = grad_output @ W.T
                    grad_output = grad_output * self.relu_derivative(activations[i])

            # 更新變分參數
            for j in range(len(self.weight_means)):
                mean = self.weight_means[j]
                log_std = self.weight_log_stds[j]
                std = np.exp(log_std)

                # KL 梯度
                grad_mean_kl = mean / self.prior_std**2 / n_samples
                grad_std_kl = (std - self.prior_std**2 / std) / n_samples

                # 總梯度
                grad_mean = grad_weights[j] + grad_mean_kl
                grad_log_std = grad_weights[j] * std * (weights[j] - mean) + grad_std_kl * std

                # 梯度下降
                self.weight_means[j] -= learning_rate * grad_mean
                self.weight_log_stds[j] -= learning_rate * grad_log_std

            # 記錄 ELBO
            if iteration % 100 == 0:
                elbo, log_lik, kl = self.elbo(X, y, n_samples=10)
                elbo_history.append(elbo)

                if verbose:
                    print(f"迭代 {iteration}: ELBO = {elbo:.4f}, "
                          f"Log-Lik = {log_lik:.4f}, KL = {kl:.4f}")

        return elbo_history

    def predict(self, X, n_samples=100):
        """
        預測（返回均值和標準差）

        Parameters:
        -----------
        X : np.ndarray
            輸入數據
        n_samples : int
            後驗樣本數量

        Returns:
        --------
        y_mean : np.ndarray
            預測均值
        y_std : np.ndarray
            預測標準差（認識不確定性）
        """
        predictions = []

        for _ in range(n_samples):
            weights = self.sample_weights(1)[0]
            y_pred, _ = self.forward(X, weights)
            predictions.append(y_pred)

        predictions = np.array(predictions)

        y_mean = predictions.mean(axis=0)
        y_std = predictions.std(axis=0)

        return y_mean, y_std


def create_regression_data(n_samples=200):
    """創建回歸數據"""
    np.random.seed(42)

    X = np.linspace(-4, 4, n_samples)

    # 非線性函數
    y_true = np.sin(X) + 0.5 * np.cos(2*X)

    # 添加異方差噪聲
    noise = 0.1 + 0.2 * np.abs(X) / 4
    y = y_true + noise * np.random.randn(n_samples)

    return X.reshape(-1, 1), y.reshape(-1, 1), y_true


def plot_regression_results(X_train, y_train, X_test, y_test, y_test_true, bnn):
    """繪製回歸結果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 預測
    y_pred, y_std = bnn.predict(X_test, n_samples=100)
    y_pred = y_pred.flatten()
    y_std = y_std.flatten()

    # 1. 預測結果與不確定性
    ax = axes[0]

    ax.scatter(X_train, y_train, alpha=0.5, s=30, label='訓練數據')
    ax.plot(X_test, y_test_true, 'g--', linewidth=2, label='真實函數')
    ax.plot(X_test, y_pred, 'r-', linewidth=2, label='BNN 預測')
    ax.fill_between(X_test.flatten(),
                     y_pred - 2*y_std,
                     y_pred + 2*y_std,
                     alpha=0.3, label='認識不確定性 (2σ)')

    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('貝葉斯神經網絡回歸')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 不確定性分析
    ax = axes[1]

    ax.plot(X_test, y_std, 'b-', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('預測標準差')
    ax.set_title('認識不確定性（模型不確定性）')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bnn_regression_results.png', dpi=300, bbox_inches='tight')
    print("回歸結果圖表已保存: bnn_regression_results.png")
    plt.show()


def plot_posterior_samples(X_test, bnn, y_true):
    """繪製後驗樣本函數"""
    plt.figure(figsize=(12, 6))

    # 繪製多個後驗樣本
    n_plot_samples = 50
    for _ in range(n_plot_samples):
        weights = bnn.sample_weights(1)[0]
        y_sample, _ = bnn.forward(X_test, weights)
        plt.plot(X_test, y_sample, 'b-', alpha=0.1)

    # 真實函數
    plt.plot(X_test, y_true, 'r--', linewidth=3, label='真實函數')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'貝葉斯神經網絡後驗樣本（{n_plot_samples} 個樣本）')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bnn_posterior_samples.png', dpi=300, bbox_inches='tight')
    print("後驗樣本圖表已保存: bnn_posterior_samples.png")
    plt.show()


def example_classification():
    """分類示例"""
    print("\n" + "=" * 60)
    print("示例 2: 貝葉斯神經網絡分類（Moons 數據集）")
    print("=" * 60)

    # 創建數據
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    y = y.reshape(-1, 1)

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 創建和訓練 BNN
    bnn = BayesianNeuralNetwork(
        layer_sizes=[2, 20, 1],
        prior_std=1.0,
        noise_std=0.1
    )

    print("訓練貝葉斯神經網絡分類器...")
    bnn.fit(X_train, y_train, n_iterations=1000, learning_rate=0.01,
            batch_size=32, verbose=False)

    # 預測
    y_pred, y_std = bnn.predict(X_test, n_samples=100)

    # 二分類準確率
    y_pred_class = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_pred_class == y_test)

    print(f"\n測試準確率: {accuracy:.4f}")
    print(f"平均不確定性: {y_std.mean():.4f}")

    # 可視化決策邊界
    plot_decision_boundary(X_train, y_train, X_test, y_test, bnn, scaler)


def plot_decision_boundary(X_train, y_train, X_test, y_test, bnn, scaler):
    """繪製決策邊界"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 創建網格
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # 預測
    Z_mean, Z_std = bnn.predict(X_grid, n_samples=100)
    Z_mean = Z_mean.reshape(xx.shape)
    Z_std = Z_std.reshape(xx.shape)

    # 1. 決策邊界
    ax = axes[0]
    contour = ax.contourf(xx, yy, Z_mean, levels=20, cmap='RdYlBu', alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train.flatten(),
              cmap='RdYlBu', edgecolors='black', s=50, alpha=0.7)
    ax.set_xlabel('特徵 1')
    ax.set_ylabel('特徵 2')
    ax.set_title('BNN 決策邊界')
    plt.colorbar(contour, ax=ax)

    # 2. 不確定性圖
    ax = axes[1]
    contour = ax.contourf(xx, yy, Z_std, levels=20, cmap='viridis', alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c='white',
              edgecolors='black', s=30, alpha=0.5)
    ax.set_xlabel('特徵 1')
    ax.set_ylabel('特徵 2')
    ax.set_title('預測不確定性')
    plt.colorbar(contour, ax=ax, label='標準差')

    plt.tight_layout()
    plt.savefig('bnn_classification_results.png', dpi=300, bbox_inches='tight')
    print("分類結果圖表已保存: bnn_classification_results.png")
    plt.show()


def main():
    """主函數"""
    print("開始貝葉斯神經網絡演示...")
    print()

    # 示例 1: 回歸
    print("=" * 60)
    print("示例 1: 貝葉斯神經網絡回歸")
    print("=" * 60)

    # 創建數據
    X, y, y_true = create_regression_data(n_samples=200)

    # 分割數據
    train_size = 100
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    y_test_true = y_true[train_size:]

    # 創建和訓練 BNN
    bnn = BayesianNeuralNetwork(
        layer_sizes=[1, 50, 50, 1],
        prior_std=1.0,
        noise_std=0.1
    )

    print("訓練貝葉斯神經網絡...")
    elbo_history = bnn.fit(
        X_train, y_train,
        n_iterations=1000,
        learning_rate=0.01,
        batch_size=32,
        verbose=True
    )

    # 評估
    y_pred, y_std = bnn.predict(X_test, n_samples=100)
    mse = np.mean((y_pred - y_test)**2)

    print(f"\n測試 MSE: {mse:.4f}")
    print(f"平均預測不確定性: {y_std.mean():.4f}")

    # 繪製結果
    plot_regression_results(X_train, y_train, X_test, y_test, y_test_true, bnn)
    plot_posterior_samples(X_test, bnn, y_test_true)

    # 示例 2: 分類
    example_classification()

    print("\n" + "=" * 60)
    print("貝葉斯神經網絡演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
