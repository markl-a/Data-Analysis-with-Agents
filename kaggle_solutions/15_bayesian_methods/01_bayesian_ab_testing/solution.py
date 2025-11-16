"""
Bayesian A/B Testing - 貝葉斯 A/B 測試
使用貝葉斯方法進行 A/B 測試和轉換率分析

數據集: 模擬網站轉換率數據
難度: ⭐⭐ 中級
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import beta as beta_function
import warnings
warnings.filterwarnings('ignore')


class BayesianABTesting:
    """貝葉斯 A/B 測試分析器"""

    def __init__(self, alpha_prior=1, beta_prior=1):
        """
        初始化貝葉斯 A/B 測試

        Parameters:
        -----------
        alpha_prior : int
            Beta 分佈的 alpha 先驗參數（成功次數 + 1）
        beta_prior : int
            Beta 分佈的 beta 先驗參數（失敗次數 + 1）
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.results = {}

    def create_sample_data(self, n_visitors_A=1000, n_visitors_B=1000,
                          true_rate_A=0.10, true_rate_B=0.12):
        """
        創建模擬 A/B 測試數據

        Parameters:
        -----------
        n_visitors_A : int
            變體 A 的訪客數量
        n_visitors_B : int
            變體 B 的訪客數量
        true_rate_A : float
            變體 A 的真實轉換率
        true_rate_B : float
            變體 B 的真實轉換率
        """
        np.random.seed(42)

        # 生成轉換數據
        conversions_A = np.random.binomial(1, true_rate_A, n_visitors_A)
        conversions_B = np.random.binomial(1, true_rate_B, n_visitors_B)

        # 創建數據框
        data_A = pd.DataFrame({
            'variant': 'A',
            'visitor_id': range(n_visitors_A),
            'converted': conversions_A,
            'timestamp': pd.date_range('2024-01-01', periods=n_visitors_A, freq='1min')
        })

        data_B = pd.DataFrame({
            'variant': 'B',
            'visitor_id': range(n_visitors_B),
            'converted': conversions_B,
            'timestamp': pd.date_range('2024-01-01', periods=n_visitors_B, freq='1min')
        })

        data = pd.concat([data_A, data_B], ignore_index=True)

        print("=" * 60)
        print("A/B 測試數據生成完成")
        print("=" * 60)
        print(f"變體 A: {n_visitors_A} 訪客, {conversions_A.sum()} 轉換")
        print(f"變體 B: {n_visitors_B} 訪客, {conversions_B.sum()} 轉換")
        print(f"真實轉換率 A: {true_rate_A:.2%}")
        print(f"真實轉換率 B: {true_rate_B:.2%}")
        print()

        return data

    def analyze_variant(self, conversions, trials, variant_name):
        """
        分析單個變體的貝葉斯統計

        Parameters:
        -----------
        conversions : int
            轉換次數
        trials : int
            總試驗次數
        variant_name : str
            變體名稱
        """
        # 更新後驗參數
        alpha_post = self.alpha_prior + conversions
        beta_post = self.beta_prior + (trials - conversions)

        # 計算統計量
        mean = alpha_post / (alpha_post + beta_post)
        mode = (alpha_post - 1) / (alpha_post + beta_post - 2) if alpha_post > 1 and beta_post > 1 else mean

        # 95% 可信區間
        credible_interval = stats.beta.interval(0.95, alpha_post, beta_post)

        self.results[variant_name] = {
            'conversions': conversions,
            'trials': trials,
            'alpha_post': alpha_post,
            'beta_post': beta_post,
            'mean': mean,
            'mode': mode,
            'credible_interval': credible_interval
        }

        return alpha_post, beta_post

    def compare_variants(self, data):
        """
        比較兩個變體並計算勝率

        Parameters:
        -----------
        data : pd.DataFrame
            包含變體和轉換數據的數據框
        """
        # 分析變體 A
        data_A = data[data['variant'] == 'A']
        conversions_A = data_A['converted'].sum()
        trials_A = len(data_A)
        alpha_A, beta_A = self.analyze_variant(conversions_A, trials_A, 'A')

        # 分析變體 B
        data_B = data[data['variant'] == 'B']
        conversions_B = data_B['converted'].sum()
        trials_B = len(data_B)
        alpha_B, beta_B = self.analyze_variant(conversions_B, trials_B, 'B')

        # 計算 B 優於 A 的概率（通過蒙特卡洛模擬）
        n_simulations = 100000
        samples_A = np.random.beta(alpha_A, beta_A, n_simulations)
        samples_B = np.random.beta(alpha_B, beta_B, n_simulations)

        prob_B_better = np.mean(samples_B > samples_A)
        prob_A_better = 1 - prob_B_better

        # 期望損失（Expected Loss）
        expected_loss_A = np.mean(np.maximum(samples_B - samples_A, 0))
        expected_loss_B = np.mean(np.maximum(samples_A - samples_B, 0))

        # 相對提升
        relative_lift = np.mean((samples_B - samples_A) / samples_A)

        print("=" * 60)
        print("貝葉斯 A/B 測試結果")
        print("=" * 60)
        print(f"\n變體 A:")
        print(f"  轉換率（後驗均值）: {self.results['A']['mean']:.4f}")
        print(f"  95% 可信區間: [{self.results['A']['credible_interval'][0]:.4f}, "
              f"{self.results['A']['credible_interval'][1]:.4f}]")

        print(f"\n變體 B:")
        print(f"  轉換率（後驗均值）: {self.results['B']['mean']:.4f}")
        print(f"  95% 可信區間: [{self.results['B']['credible_interval'][0]:.4f}, "
              f"{self.results['B']['credible_interval'][1]:.4f}]")

        print(f"\n比較結果:")
        print(f"  P(B > A) = {prob_B_better:.4f} ({prob_B_better*100:.2f}%)")
        print(f"  P(A > B) = {prob_A_better:.4f} ({prob_A_better*100:.2f}%)")
        print(f"  相對提升: {relative_lift*100:.2f}%")
        print(f"  選擇 A 的期望損失: {expected_loss_A:.6f}")
        print(f"  選擇 B 的期望損失: {expected_loss_B:.6f}")

        if prob_B_better > 0.95:
            print(f"\n結論: B 變體有 {prob_B_better*100:.2f}% 的概率優於 A（高信心）")
        elif prob_B_better > 0.90:
            print(f"\n結論: B 變體有 {prob_B_better*100:.2f}% 的概率優於 A（中等信心）")
        else:
            print(f"\n結論: 證據不足以確定哪個變體更優")

        return {
            'prob_B_better': prob_B_better,
            'prob_A_better': prob_A_better,
            'expected_loss_A': expected_loss_A,
            'expected_loss_B': expected_loss_B,
            'relative_lift': relative_lift
        }

    def plot_posteriors(self):
        """繪製後驗分佈"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 定義轉換率範圍
        x = np.linspace(0, 0.3, 1000)

        # 1. 後驗分佈
        ax = axes[0, 0]
        for variant in ['A', 'B']:
            alpha = self.results[variant]['alpha_post']
            beta = self.results[variant]['beta_post']
            y = stats.beta.pdf(x, alpha, beta)
            ax.plot(x, y, label=f'變體 {variant}', linewidth=2)
            ax.axvline(self.results[variant]['mean'],
                      linestyle='--', alpha=0.5,
                      label=f'{variant} 均值: {self.results[variant]["mean"]:.4f}')

        ax.set_xlabel('轉換率')
        ax.set_ylabel('概率密度')
        ax.set_title('後驗分佈比較')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 累積分佈函數
        ax = axes[0, 1]
        for variant in ['A', 'B']:
            alpha = self.results[variant]['alpha_post']
            beta = self.results[variant]['beta_post']
            y = stats.beta.cdf(x, alpha, beta)
            ax.plot(x, y, label=f'變體 {variant}', linewidth=2)

        ax.set_xlabel('轉換率')
        ax.set_ylabel('累積概率')
        ax.set_title('累積分佈函數 (CDF)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. 差異分佈（B - A）
        ax = axes[1, 0]
        n_simulations = 100000
        samples_A = np.random.beta(self.results['A']['alpha_post'],
                                   self.results['A']['beta_post'],
                                   n_simulations)
        samples_B = np.random.beta(self.results['B']['alpha_post'],
                                   self.results['B']['beta_post'],
                                   n_simulations)
        diff = samples_B - samples_A

        ax.hist(diff, bins=100, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='無差異')
        ax.axvline(np.mean(diff), color='green', linestyle='--',
                  linewidth=2, label=f'均值: {np.mean(diff):.4f}')

        ax.set_xlabel('轉換率差異 (B - A)')
        ax.set_ylabel('概率密度')
        ax.set_title('轉換率差異分佈')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 可信區間比較
        ax = axes[1, 1]
        variants = ['A', 'B']
        means = [self.results[v]['mean'] for v in variants]
        lower = [self.results[v]['credible_interval'][0] for v in variants]
        upper = [self.results[v]['credible_interval'][1] for v in variants]

        y_pos = np.arange(len(variants))
        ax.errorbar(means, y_pos,
                   xerr=[[means[i] - lower[i] for i in range(len(variants))],
                         [upper[i] - means[i] for i in range(len(variants))]],
                   fmt='o', markersize=10, capsize=5, capthick=2, linewidth=2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'變體 {v}' for v in variants])
        ax.set_xlabel('轉換率')
        ax.set_title('95% 可信區間比較')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('bayesian_ab_testing_results.png', dpi=300, bbox_inches='tight')
        print("\n圖表已保存: bayesian_ab_testing_results.png")
        plt.show()

    def sequential_analysis(self, data, check_points=10):
        """
        序列分析：觀察 P(B > A) 隨數據增加的變化

        Parameters:
        -----------
        data : pd.DataFrame
            測試數據
        check_points : int
            檢查點數量
        """
        data_A = data[data['variant'] == 'A'].reset_index(drop=True)
        data_B = data[data['variant'] == 'B'].reset_index(drop=True)

        min_samples = min(len(data_A), len(data_B))
        sample_sizes = np.linspace(100, min_samples, check_points, dtype=int)

        prob_B_better_history = []

        for n in sample_sizes:
            # 當前樣本的轉換數據
            conv_A = data_A.iloc[:n]['converted'].sum()
            conv_B = data_B.iloc[:n]['converted'].sum()

            # 後驗參數
            alpha_A = self.alpha_prior + conv_A
            beta_A = self.beta_prior + (n - conv_A)
            alpha_B = self.alpha_prior + conv_B
            beta_B = self.beta_prior + (n - conv_B)

            # 蒙特卡洛模擬
            samples_A = np.random.beta(alpha_A, beta_A, 10000)
            samples_B = np.random.beta(alpha_B, beta_B, 10000)
            prob_B_better = np.mean(samples_B > samples_A)

            prob_B_better_history.append(prob_B_better)

        # 繪製序列分析結果
        plt.figure(figsize=(12, 6))
        plt.plot(sample_sizes, prob_B_better_history,
                marker='o', linewidth=2, markersize=8)
        plt.axhline(y=0.95, color='g', linestyle='--',
                   label='95% 信心閾值', linewidth=2)
        plt.axhline(y=0.5, color='gray', linestyle='--',
                   label='無差異', alpha=0.5)

        plt.xlabel('每個變體的樣本數量')
        plt.ylabel('P(B > A)')
        plt.title('序列分析：P(B > A) 隨樣本量的變化')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('sequential_analysis.png', dpi=300, bbox_inches='tight')
        print("序列分析圖表已保存: sequential_analysis.png")
        plt.show()


def main():
    """主函數"""
    print("開始貝葉斯 A/B 測試分析...")
    print()

    # 初始化分析器（使用均勻先驗）
    ab_test = BayesianABTesting(alpha_prior=1, beta_prior=1)

    # 創建模擬數據
    # 場景：測試兩個網站設計的轉換率
    data = ab_test.create_sample_data(
        n_visitors_A=1000,
        n_visitors_B=1000,
        true_rate_A=0.10,  # A 版本 10% 轉換率
        true_rate_B=0.12   # B 版本 12% 轉換率（提升 20%）
    )

    # 執行貝葉斯分析
    comparison_results = ab_test.compare_variants(data)

    # 繪製後驗分佈
    ab_test.plot_posteriors()

    # 序列分析
    print("\n" + "=" * 60)
    print("執行序列分析...")
    print("=" * 60)
    ab_test.sequential_analysis(data, check_points=10)

    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
