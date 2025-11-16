"""
Bayesian Linear Regression - 貝葉斯線性回歸
使用貝葉斯方法進行線性回歸，包含不確定性量化

數據集: 模擬回歸數據
難度: ⭐⭐ 中級
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class BayesianLinearRegression:
    """貝葉斯線性回歸模型"""

    def __init__(self, alpha_prior=1.0, beta_prior=1.0):
        """
        初始化貝葉斯線性回歸

        Parameters:
        -----------
        alpha_prior : float
            先驗精度參數（權重的精度）
        beta_prior : float
            似然精度參數（噪聲的精度）
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.mean_post = None
        self.cov_post = None
        self.samples = None

    def create_sample_data(self, n_samples=100, noise_level=0.3):
        """
        創建回歸數據

        Parameters:
        -----------
        n_samples : int
            樣本數量
        noise_level : float
            噪聲水平
        """
        np.random.seed(42)

        # 生成特徵
        X = np.linspace(-3, 3, n_samples)

        # 真實函數：y = 2 + 0.5*x + 1.5*x^2 + noise
        true_weights = np.array([2.0, 0.5, 1.5])
        X_poly = np.column_stack([np.ones(n_samples), X, X**2])

        # 添加噪聲
        y = X_poly @ true_weights + np.random.normal(0, noise_level, n_samples)

        print("=" * 60)
        print("數據生成完成")
        print("=" * 60)
        print(f"樣本數量: {n_samples}")
        print(f"真實權重: {true_weights}")
        print(f"噪聲水平: {noise_level}")
        print()

        return X.reshape(-1, 1), y, X_poly, true_weights

    def fit(self, X, y):
        """
        擬合貝葉斯線性回歸

        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣 (n_samples, n_features)
        y : np.ndarray
            目標值 (n_samples,)
        """
        n_samples, n_features = X.shape

        # 先驗均值（假設為零均值）
        mean_prior = np.zeros(n_features)

        # 先驗協方差矩陣
        cov_prior = (1 / self.alpha_prior) * np.eye(n_features)

        # 計算後驗參數
        # 後驗精度矩陣
        precision_post = self.alpha_prior * np.eye(n_features) + self.beta_prior * (X.T @ X)

        # 後驗協方差矩陣
        self.cov_post = np.linalg.inv(precision_post)

        # 後驗均值
        self.mean_post = self.cov_post @ (
            self.alpha_prior * cov_prior @ mean_prior +
            self.beta_prior * X.T @ y
        )

        # 從後驗分佈採樣
        self.samples = np.random.multivariate_normal(
            self.mean_post, self.cov_post, size=1000
        )

        print("=" * 60)
        print("貝葉斯線性回歸擬合完成")
        print("=" * 60)
        print(f"後驗均值（權重）:\n{self.mean_post}")
        print(f"\n後驗標準差:\n{np.sqrt(np.diag(self.cov_post))}")
        print()

    def predict(self, X, return_std=True):
        """
        預測

        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣
        return_std : bool
            是否返回標準差

        Returns:
        --------
        y_pred : np.ndarray
            預測均值
        y_std : np.ndarray (optional)
            預測標準差
        """
        # 預測均值
        y_pred = X @ self.mean_post

        if return_std:
            # 預測方差包括：參數不確定性 + 噪聲
            y_var = np.zeros(len(X))
            for i in range(len(X)):
                # 參數不確定性
                param_var = X[i:i+1] @ self.cov_post @ X[i:i+1].T
                # 噪聲方差
                noise_var = 1 / self.beta_prior
                y_var[i] = param_var + noise_var

            y_std = np.sqrt(y_var)
            return y_pred, y_std
        else:
            return y_pred

    def predict_distribution(self, X, n_samples=1000):
        """
        預測完整分佈（通過後驗樣本）

        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣
        n_samples : int
            樣本數量
        """
        predictions = np.zeros((n_samples, len(X)))

        for i, weights in enumerate(self.samples[:n_samples]):
            # 使用採樣的權重預測
            y_pred = X @ weights
            # 添加觀測噪聲
            predictions[i] = y_pred + np.random.normal(0, 1/np.sqrt(self.beta_prior), len(X))

        return predictions

    def plot_results(self, X_train, y_train, X_test, y_test, X_poly_test, true_weights=None):
        """
        繪製結果

        Parameters:
        -----------
        X_train : np.ndarray
            訓練特徵（原始特徵，用於繪圖）
        y_train : np.ndarray
            訓練目標
        X_test : np.ndarray
            測試特徵（原始特徵）
        y_test : np.ndarray
            測試目標
        X_poly_test : np.ndarray
            測試特徵（多項式特徵，用於預測）
        true_weights : np.ndarray
            真實權重（如果已知）
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 預測結果與不確定性
        ax = axes[0, 0]

        # 預測
        y_pred, y_std = self.predict(X_poly_test, return_std=True)

        # 繪製訓練數據
        ax.scatter(X_train, y_train, alpha=0.6, s=50, label='訓練數據')

        # 繪製預測均值
        ax.plot(X_test, y_pred, 'r-', linewidth=2, label='預測均值')

        # 繪製不確定性區間
        ax.fill_between(X_test.flatten(),
                        y_pred - 2*y_std,
                        y_pred + 2*y_std,
                        alpha=0.3, label='95% 預測區間')

        # 如果有真實函數，繪製它
        if true_weights is not None:
            y_true = X_poly_test @ true_weights
            ax.plot(X_test, y_true, 'g--', linewidth=2, label='真實函數')

        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title('貝葉斯線性回歸預測')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 後驗樣本
        ax = axes[0, 1]
        # 繪製從後驗採樣的函數
        for i in range(min(50, len(self.samples))):
            y_sample = X_poly_test @ self.samples[i]
            ax.plot(X_test, y_sample, 'b-', alpha=0.1)

        ax.scatter(X_train, y_train, alpha=0.6, s=50, c='red', label='訓練數據')
        if true_weights is not None:
            y_true = X_poly_test @ true_weights
            ax.plot(X_test, y_true, 'g--', linewidth=2, label='真實函數')

        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title('後驗樣本函數（50 條）')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 權重後驗分佈
        ax = axes[1, 0]
        n_weights = len(self.mean_post)

        for i in range(n_weights):
            weights_i = self.samples[:, i]
            ax.hist(weights_i, bins=50, alpha=0.5, label=f'w_{i}', density=True)
            ax.axvline(self.mean_post[i], linestyle='--', linewidth=2)

            if true_weights is not None:
                ax.axvline(true_weights[i], linestyle=':', linewidth=2, color='red')

        ax.set_xlabel('權重值')
        ax.set_ylabel('概率密度')
        ax.set_title('權重後驗分佈')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 殘差分析
        ax = axes[1, 1]
        residuals = y_test - y_pred

        # Q-Q 圖
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('殘差 Q-Q 圖')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('bayesian_regression_results.png', dpi=300, bbox_inches='tight')
        print("圖表已保存: bayesian_regression_results.png")
        plt.show()

    def compute_metrics(self, X, y):
        """
        計算評估指標

        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣
        y : np.ndarray
            真實目標值
        """
        y_pred, y_std = self.predict(X, return_std=True)

        # MSE
        mse = np.mean((y - y_pred) ** 2)

        # RMSE
        rmse = np.sqrt(mse)

        # R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # 負對數似然（預測對數概率）
        log_likelihood = -0.5 * np.sum(
            np.log(2 * np.pi * y_std**2) + ((y - y_pred) / y_std) ** 2
        )

        print("=" * 60)
        print("評估指標")
        print("=" * 60)
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"負對數似然: {log_likelihood:.4f}")
        print()

        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'log_likelihood': log_likelihood
        }


class BayesianRidgeRegression:
    """貝葉斯嶺回歸（自動估計正則化參數）"""

    def __init__(self, n_iterations=100):
        """
        初始化貝葉斯嶺回歸

        Parameters:
        -----------
        n_iterations : int
            迭代次數
        """
        self.n_iterations = n_iterations
        self.alpha_history = []
        self.beta_history = []
        self.mean_post = None
        self.cov_post = None

    def fit(self, X, y):
        """
        擬合貝葉斯嶺回歸（使用證據近似）

        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣
        y : np.ndarray
            目標值
        """
        n_samples, n_features = X.shape

        # 初始化超參數
        alpha = 1.0  # 權重精度
        beta = 1.0   # 噪聲精度

        for iteration in range(self.n_iterations):
            # E 步：計算後驗
            precision_post = alpha * np.eye(n_features) + beta * (X.T @ X)
            self.cov_post = np.linalg.inv(precision_post)
            self.mean_post = beta * self.cov_post @ X.T @ y

            # M 步：更新超參數
            gamma = n_features - alpha * np.trace(self.cov_post)

            # 更新 alpha
            alpha = gamma / (self.mean_post.T @ self.mean_post)

            # 更新 beta
            residual_sum = np.sum((y - X @ self.mean_post) ** 2)
            beta = (n_samples - gamma) / residual_sum

            self.alpha_history.append(alpha)
            self.beta_history.append(beta)

        print("=" * 60)
        print("貝葉斯嶺回歸擬合完成（證據近似）")
        print("=" * 60)
        print(f"最終 alpha（權重精度）: {alpha:.4f}")
        print(f"最終 beta（噪聲精度）: {beta:.4f}")
        print(f"後驗均值（權重）:\n{self.mean_post}")
        print()

    def plot_hyperparameter_evolution(self):
        """繪製超參數演化"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        ax.plot(self.alpha_history, linewidth=2)
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('Alpha（權重精度）')
        ax.set_title('Alpha 演化')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(self.beta_history, linewidth=2)
        ax.set_xlabel('迭代次數')
        ax.set_ylabel('Beta（噪聲精度）')
        ax.set_title('Beta 演化')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('hyperparameter_evolution.png', dpi=300, bbox_inches='tight')
        print("超參數演化圖表已保存: hyperparameter_evolution.png")
        plt.show()


def main():
    """主函數"""
    print("開始貝葉斯線性回歸分析...")
    print()

    # 創建數據
    blr = BayesianLinearRegression(alpha_prior=1.0, beta_prior=10.0)
    X, y, X_poly, true_weights = blr.create_sample_data(n_samples=100, noise_level=0.3)

    # 分割數據
    n_train = 70
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    X_poly_train = X_poly[:n_train]
    X_poly_test = X_poly[n_train:]

    # 擬合模型
    blr.fit(X_poly_train, y_train)

    # 計算指標
    blr.compute_metrics(X_poly_test, y_test)

    # 繪製結果
    blr.plot_results(X_train, y_train, X_test, y_test, X_poly_test, true_weights)

    # 貝葉斯嶺回歸（自動正則化）
    print("\n" + "=" * 60)
    print("貝葉斯嶺回歸（自動正則化）")
    print("=" * 60)

    ridge = BayesianRidgeRegression(n_iterations=100)
    ridge.fit(X_poly_train, y_train)
    ridge.plot_hyperparameter_evolution()

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
