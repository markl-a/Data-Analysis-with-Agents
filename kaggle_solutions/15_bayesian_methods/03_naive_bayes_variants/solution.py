"""
Naive Bayes Variants - 樸素貝葉斯變體
比較不同樸素貝葉斯分類器的性能和特點

數據集: 模擬混合特徵數據
難度: ⭐⭐ 中級
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class GaussianNaiveBayes:
    """高斯樸素貝葉斯分類器（從零實現）"""

    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.feature_params = {}

    def fit(self, X, y):
        """
        訓練高斯樸素貝葉斯

        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣 (n_samples, n_features)
        y : np.ndarray
            標籤 (n_samples,)
        """
        self.classes = np.unique(y)
        n_samples = len(y)

        for c in self.classes:
            # 類別先驗概率
            X_c = X[y == c]
            self.class_priors[c] = len(X_c) / n_samples

            # 每個特徵的均值和標準差
            self.feature_params[c] = {
                'mean': X_c.mean(axis=0),
                'std': X_c.std(axis=0) + 1e-9  # 避免除零
            }

    def _gaussian_pdf(self, x, mean, std):
        """計算高斯概率密度"""
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict_proba(self, X):
        """
        預測類別概率

        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣

        Returns:
        --------
        proba : np.ndarray
            類別概率
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        proba = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.classes):
            # 先驗概率（對數）
            prior = np.log(self.class_priors[c])

            # 似然（對數）
            mean = self.feature_params[c]['mean']
            std = self.feature_params[c]['std']

            # 對數似然
            log_likelihood = np.sum(
                np.log(self._gaussian_pdf(X, mean, std)),
                axis=1
            )

            # 對數後驗（未歸一化）
            proba[:, idx] = prior + log_likelihood

        # 歸一化（log-sum-exp 技巧）
        log_sum = np.max(proba, axis=1, keepdims=True)
        proba = np.exp(proba - log_sum)
        proba = proba / proba.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X):
        """預測類別"""
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]


class MultinomialNaiveBayes:
    """多項式樸素貝葉斯分類器（用於計數數據）"""

    def __init__(self, alpha=1.0):
        """
        初始化多項式樸素貝葉斯

        Parameters:
        -----------
        alpha : float
            Laplace 平滑參數
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}

    def fit(self, X, y):
        """訓練多項式樸素貝葉斯"""
        self.classes = np.unique(y)
        n_samples = len(y)
        n_features = X.shape[1]

        for c in self.classes:
            X_c = X[y == c]

            # 類別先驗
            self.class_priors[c] = len(X_c) / n_samples

            # 特徵條件概率（帶 Laplace 平滑）
            feature_counts = X_c.sum(axis=0) + self.alpha
            total_count = feature_counts.sum()
            self.feature_probs[c] = feature_counts / total_count

    def predict_proba(self, X):
        """預測類別概率"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_proba = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.classes):
            # 先驗對數概率
            log_prior = np.log(self.class_priors[c])

            # 似然對數概率
            log_likelihood = X @ np.log(self.feature_probs[c])

            log_proba[:, idx] = log_prior + log_likelihood

        # 歸一化
        log_sum = np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba - log_sum)
        proba = proba / proba.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X):
        """預測類別"""
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]


class BernoulliNaiveBayes:
    """伯努利樸素貝葉斯分類器（用於二值特徵）"""

    def __init__(self, alpha=1.0):
        """
        初始化伯努利樸素貝葉斯

        Parameters:
        -----------
        alpha : float
            Laplace 平滑參數
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}

    def fit(self, X, y):
        """訓練伯努利樸素貝葉斯"""
        self.classes = np.unique(y)
        n_samples = len(y)

        for c in self.classes:
            X_c = X[y == c]
            n_samples_c = len(X_c)

            # 類別先驗
            self.class_priors[c] = n_samples_c / n_samples

            # 特徵為 1 的概率（帶平滑）
            self.feature_probs[c] = (X_c.sum(axis=0) + self.alpha) / (
                n_samples_c + 2 * self.alpha
            )

    def predict_proba(self, X):
        """預測類別概率"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_proba = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.classes):
            # 先驗對數概率
            log_prior = np.log(self.class_priors[c])

            # 似然對數概率
            p = self.feature_probs[c]
            log_likelihood = (
                X @ np.log(p + 1e-9) +
                (1 - X) @ np.log(1 - p + 1e-9)
            )

            log_proba[:, idx] = log_prior + log_likelihood

        # 歸一化
        log_sum = np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba - log_sum)
        proba = proba / proba.sum(axis=1, keepdims=True)

        return proba

    def predict(self, X):
        """預測類別"""
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]


def create_gaussian_data(n_samples=500):
    """創建高斯數據（用於高斯樸素貝葉斯）"""
    np.random.seed(42)

    # 類別 0
    X0 = np.random.randn(n_samples // 2, 4) * np.array([1, 2, 1, 3]) + np.array([0, 0, 0, 0])
    y0 = np.zeros(n_samples // 2)

    # 類別 1
    X1 = np.random.randn(n_samples // 2, 4) * np.array([2, 1, 3, 1]) + np.array([3, 3, -2, 2])
    y1 = np.ones(n_samples // 2)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    return X, y


def create_multinomial_data(n_samples=500, n_features=20):
    """創建多項式數據（詞頻計數）"""
    np.random.seed(42)

    # 類別 0: 某些詞頻更高
    X0 = np.random.poisson(lam=2, size=(n_samples // 2, n_features))
    X0[:, :10] = np.random.poisson(lam=5, size=(n_samples // 2, 10))  # 前 10 個詞頻高
    y0 = np.zeros(n_samples // 2)

    # 類別 1: 不同的詞頻模式
    X1 = np.random.poisson(lam=2, size=(n_samples // 2, n_features))
    X1[:, 10:] = np.random.poisson(lam=5, size=(n_samples // 2, 10))  # 後 10 個詞頻高
    y1 = np.ones(n_samples // 2)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    return X, y


def create_bernoulli_data(n_samples=500, n_features=20):
    """創建二值數據（文檔-詞矩陣）"""
    np.random.seed(42)

    # 類別 0: 某些詞更可能出現
    X0 = np.random.binomial(1, 0.2, size=(n_samples // 2, n_features))
    X0[:, :10] = np.random.binomial(1, 0.7, size=(n_samples // 2, 10))
    y0 = np.zeros(n_samples // 2)

    # 類別 1: 不同的詞出現模式
    X1 = np.random.binomial(1, 0.2, size=(n_samples // 2, n_features))
    X1[:, 10:] = np.random.binomial(1, 0.7, size=(n_samples // 2, 10))
    y1 = np.ones(n_samples // 2)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    return X, y


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """評估模型"""
    # 訓練
    model.fit(X_train, y_train)

    # 預測
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 準確率
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{model_name} 評估結果")
    print("=" * 60)
    print(f"準確率: {accuracy:.4f}")
    print("\n分類報告:")
    print(classification_report(y_test, y_pred))

    return accuracy, y_pred, y_proba


def plot_comparison(results):
    """繪製模型比較圖"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 準確率比較
    ax = axes[0, 0]
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]

    bars = ax.bar(models, accuracies, alpha=0.7, edgecolor='black')
    ax.set_ylabel('準確率')
    ax.set_title('模型準確率比較')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

    # 2. 混淆矩陣（以高斯為例）
    ax = axes[0, 1]
    if 'Gaussian NB' in results:
        cm = confusion_matrix(results['Gaussian NB']['y_test'],
                             results['Gaussian NB']['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar_kws={'label': '計數'})
        ax.set_xlabel('預測標籤')
        ax.set_ylabel('真實標籤')
        ax.set_title('混淆矩陣（高斯樸素貝葉斯）')

    # 3. 概率校準（可靠性圖）
    ax = axes[1, 0]
    for model_name in models:
        if model_name in results:
            y_proba = results[model_name]['y_proba']
            y_test = results[model_name]['y_test']

            # 計算校準曲線
            n_bins = 10
            bins = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # 只使用正類概率
            prob_pred = y_proba[:, 1]
            prob_true = np.zeros(n_bins)
            counts = np.zeros(n_bins)

            for i in range(n_bins):
                mask = (prob_pred >= bins[i]) & (prob_pred < bins[i + 1])
                if mask.sum() > 0:
                    prob_true[i] = y_test[mask].mean()
                    counts[i] = mask.sum()

            # 只繪製有數據的箱
            valid = counts > 0
            ax.plot(bin_centers[valid], prob_true[valid], 'o-', label=model_name,
                   markersize=8, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='完美校準')
    ax.set_xlabel('預測概率')
    ax.set_ylabel('真實比例')
    ax.set_title('概率校準曲線')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 預測概率分佈（以高斯為例）
    ax = axes[1, 1]
    if 'Gaussian NB' in results:
        y_proba = results['Gaussian NB']['y_proba']
        y_test = results['Gaussian NB']['y_test']

        # 正類概率
        prob_class1 = y_proba[:, 1]

        # 分別繪製兩個類別的概率分佈
        ax.hist(prob_class1[y_test == 0], bins=30, alpha=0.5,
               label='真實類別 0', density=True, edgecolor='black')
        ax.hist(prob_class1[y_test == 1], bins=30, alpha=0.5,
               label='真實類別 1', density=True, edgecolor='black')

        ax.set_xlabel('預測為類別 1 的概率')
        ax.set_ylabel('密度')
        ax.set_title('預測概率分佈（高斯樸素貝葉斯）')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('naive_bayes_comparison.png', dpi=300, bbox_inches='tight')
    print("\n圖表已保存: naive_bayes_comparison.png")
    plt.show()


def main():
    """主函數"""
    print("開始樸素貝葉斯變體比較...")
    print()

    results = {}

    # 1. 高斯樸素貝葉斯
    print("=" * 60)
    print("1. 高斯樸素貝葉斯（連續特徵）")
    print("=" * 60)
    X_gauss, y_gauss = create_gaussian_data(n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(
        X_gauss, y_gauss, test_size=0.3, random_state=42
    )

    gnb = GaussianNaiveBayes()
    acc, y_pred, y_proba = evaluate_model(
        gnb, X_train, X_test, y_train, y_test, "高斯樸素貝葉斯"
    )
    results['Gaussian NB'] = {
        'accuracy': acc,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'y_test': y_test
    }

    # 2. 多項式樸素貝葉斯
    print("\n" + "=" * 60)
    print("2. 多項式樸素貝葉斯（計數特徵）")
    print("=" * 60)
    X_multi, y_multi = create_multinomial_data(n_samples=500, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_multi, y_multi, test_size=0.3, random_state=42
    )

    mnb = MultinomialNaiveBayes(alpha=1.0)
    acc, y_pred, y_proba = evaluate_model(
        mnb, X_train, X_test, y_train, y_test, "多項式樸素貝葉斯"
    )
    results['Multinomial NB'] = {
        'accuracy': acc,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'y_test': y_test
    }

    # 3. 伯努利樸素貝葉斯
    print("\n" + "=" * 60)
    print("3. 伯努利樸素貝葉斯（二值特徵）")
    print("=" * 60)
    X_bern, y_bern = create_bernoulli_data(n_samples=500, n_features=20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_bern, y_bern, test_size=0.3, random_state=42
    )

    bnb = BernoulliNaiveBayes(alpha=1.0)
    acc, y_pred, y_proba = evaluate_model(
        bnb, X_train, X_test, y_train, y_test, "伯努利樸素貝葉斯"
    )
    results['Bernoulli NB'] = {
        'accuracy': acc,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'y_test': y_test
    }

    # 繪製比較圖
    plot_comparison(results)

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
