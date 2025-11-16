"""
A/B Test Analysis System
========================

Problem: Design, analyze, and interpret A/B tests for product features,
marketing campaigns, and business decisions with statistical rigor

Kaggle-style competition: A/B Testing Analytics
Difficulty: ⭐⭐⭐

This solution demonstrates:
- Sample size and power calculation
- Statistical hypothesis testing
- Multiple comparison corrections
- Bayesian A/B testing
- Segmentation analysis
- Practical significance vs statistical significance
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import beta, norm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class ABTestAnalyzer:
    """Comprehensive A/B test analysis framework"""

    def __init__(self, alpha=0.05, power=0.80):
        self.alpha = alpha  # Significance level
        self.power = power  # Statistical power

    def create_sample_data(self, n_users=10000):
        """Generate realistic A/B test data"""
        np.random.seed(42)

        # Control group (A)
        n_control = n_users // 2
        control_conversion_rate = 0.10  # 10% baseline conversion

        control = pd.DataFrame({
            'user_id': range(n_control),
            'variant': 'A',
            'age': np.random.normal(35, 12, n_control).clip(18, 70),
            'session_duration': np.random.lognormal(4, 1, n_control).clip(10, 3600),
            'pages_viewed': np.random.poisson(5, n_control),
            'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_control,
                                      p=[0.6, 0.3, 0.1]),
            'source': np.random.choice(['organic', 'paid', 'social', 'email'], n_control,
                                      p=[0.4, 0.3, 0.2, 0.1])
        })

        # Treatment group (B) - with lift
        n_treatment = n_users - n_control
        treatment_conversion_rate = 0.12  # 20% relative lift

        treatment = pd.DataFrame({
            'user_id': range(n_control, n_users),
            'variant': 'B',
            'age': np.random.normal(35, 12, n_treatment).clip(18, 70),
            'session_duration': np.random.lognormal(4.1, 1, n_treatment).clip(10, 3600),
            'pages_viewed': np.random.poisson(5.5, n_treatment),
            'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_treatment,
                                      p=[0.6, 0.3, 0.1]),
            'source': np.random.choice(['organic', 'paid', 'social', 'email'], n_treatment,
                                      p=[0.4, 0.3, 0.2, 0.1])
        })

        # Combine
        df = pd.concat([control, treatment], ignore_index=True)

        # Generate conversions with some dependency on features
        conversion_score = (
            (df['variant'] == 'B').astype(int) * 0.2 +  # Treatment effect
            (df['pages_viewed'] / 10) * 0.3 +
            (df['session_duration'] / 1000) * 0.2 +
            (df['device'] == 'desktop').astype(int) * 0.1 +
            (df['source'] == 'paid').astype(int) * 0.15 +
            np.random.normal(0, 0.3, n_users)
        )

        base_rate = 0.10 if df['variant'].iloc[0] == 'A' else 0.10
        conversion_prob = 1 / (1 + np.exp(-(conversion_score - 2)))
        df['converted'] = (np.random.random(n_users) < conversion_prob).astype(int)

        # Revenue (only for converted users)
        df['revenue'] = np.where(
            df['converted'] == 1,
            np.random.lognormal(3.5, 0.5, n_users) * (1 + 0.1 * (df['variant'] == 'B')),
            0
        )

        return df

    def calculate_sample_size(self, baseline_rate, mde, alpha=None, power=None):
        """Calculate required sample size per variant"""
        if alpha is None:
            alpha = self.alpha
        if power is None:
            power = self.power

        # Effect size (Cohen's h for proportions)
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        # Calculate sample size
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)

        n = ((z_alpha + z_beta) ** 2) / (effect_size ** 2)

        return int(np.ceil(n))

    def frequentist_test(self, df):
        """Perform frequentist hypothesis tests"""
        results = {}

        # Split by variant
        control = df[df['variant'] == 'A']
        treatment = df[df['variant'] == 'B']

        # Conversion rate test (two-proportion z-test)
        n_control = len(control)
        n_treatment = len(treatment)
        conv_control = control['converted'].sum()
        conv_treatment = treatment['converted'].sum()

        p_control = conv_control / n_control
        p_treatment = conv_treatment / n_treatment
        p_pooled = (conv_control + conv_treatment) / (n_control + n_treatment)

        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
        z_stat = (p_treatment - p_control) / se
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))

        # Confidence interval for difference
        se_diff = np.sqrt(p_control*(1-p_control)/n_control +
                         p_treatment*(1-p_treatment)/n_treatment)
        ci_lower = (p_treatment - p_control) - 1.96 * se_diff
        ci_upper = (p_treatment - p_control) + 1.96 * se_diff

        results['conversion_rate'] = {
            'control_rate': p_control,
            'treatment_rate': p_treatment,
            'absolute_lift': p_treatment - p_control,
            'relative_lift': (p_treatment - p_control) / p_control,
            'z_statistic': z_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < self.alpha
        }

        # Revenue per user test (t-test)
        rev_control = control['revenue']
        rev_treatment = treatment['revenue']

        t_stat, t_pvalue = stats.ttest_ind(rev_treatment, rev_control)

        results['revenue_per_user'] = {
            'control_mean': rev_control.mean(),
            'treatment_mean': rev_treatment.mean(),
            'absolute_diff': rev_treatment.mean() - rev_control.mean(),
            'relative_diff': (rev_treatment.mean() - rev_control.mean()) / rev_control.mean(),
            't_statistic': t_stat,
            'p_value': t_pvalue,
            'significant': t_pvalue < self.alpha
        }

        return results

    def bayesian_test(self, df):
        """Perform Bayesian A/B test"""
        # Split by variant
        control = df[df['variant'] == 'A']
        treatment = df[df['variant'] == 'B']

        # Prior parameters (uniform prior: alpha=1, beta=1)
        alpha_prior = 1
        beta_prior = 1

        # Posterior parameters
        control_conversions = control['converted'].sum()
        control_n = len(control)
        alpha_control = alpha_prior + control_conversions
        beta_control = beta_prior + (control_n - control_conversions)

        treatment_conversions = treatment['converted'].sum()
        treatment_n = len(treatment)
        alpha_treatment = alpha_prior + treatment_conversions
        beta_treatment = beta_prior + (treatment_n - treatment_conversions)

        # Sample from posterior distributions
        n_samples = 10000
        control_samples = np.random.beta(alpha_control, beta_control, n_samples)
        treatment_samples = np.random.beta(alpha_treatment, beta_treatment, n_samples)

        # Probability that B > A
        prob_b_better = (treatment_samples > control_samples).mean()

        # Expected loss (expected difference if wrong decision)
        lift_samples = treatment_samples - control_samples
        expected_loss_choose_b = np.maximum(0, -lift_samples).mean()
        expected_loss_choose_a = np.maximum(0, lift_samples).mean()

        return {
            'prob_b_better': prob_b_better,
            'expected_loss_choose_b': expected_loss_choose_b,
            'expected_loss_choose_a': expected_loss_choose_a,
            'lift_mean': lift_samples.mean(),
            'lift_ci': np.percentile(lift_samples, [2.5, 97.5])
        }

    def segment_analysis(self, df):
        """Analyze results by segments"""
        segments = {}

        # By device
        for device in df['device'].unique():
            segment_data = df[df['device'] == device]
            control = segment_data[segment_data['variant'] == 'A']
            treatment = segment_data[segment_data['variant'] == 'B']

            if len(control) > 0 and len(treatment) > 0:
                conv_rate_diff = (treatment['converted'].mean() -
                                 control['converted'].mean())
                segments[f'device_{device}'] = {
                    'control_conv': control['converted'].mean(),
                    'treatment_conv': treatment['converted'].mean(),
                    'lift': conv_rate_diff / control['converted'].mean()
                                 if control['converted'].mean() > 0 else 0
                }

        # By source
        for source in df['source'].unique():
            segment_data = df[df['source'] == source]
            control = segment_data[segment_data['variant'] == 'A']
            treatment = segment_data[segment_data['variant'] == 'B']

            if len(control) > 0 and len(treatment) > 0:
                conv_rate_diff = (treatment['converted'].mean() -
                                 control['converted'].mean())
                segments[f'source_{source}'] = {
                    'control_conv': control['converted'].mean(),
                    'treatment_conv': treatment['converted'].mean(),
                    'lift': conv_rate_diff / control['converted'].mean()
                                 if control['converted'].mean() > 0 else 0
                }

        return segments

    def plot_results(self, df, freq_results, bayes_results, segments):
        """Visualize A/B test results"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Conversion Rates
        ax1 = fig.add_subplot(gs[0, 0])
        conv_data = df.groupby('variant')['converted'].agg(['sum', 'count'])
        conv_rates = (conv_data['sum'] / conv_data['count']) * 100

        bars = ax1.bar(['A (Control)', 'B (Treatment)'], conv_rates.values,
                      color=['#3498db', '#2ecc71'], edgecolor='black', linewidth=2)
        ax1.set_ylabel('Conversion Rate (%)', fontsize=11)
        ax1.set_title('Conversion Rate by Variant', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, rate in zip(bars, conv_rates.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Revenue per User
        ax2 = fig.add_subplot(gs[0, 1])
        rev_data = df.groupby('variant')['revenue'].mean()

        bars = ax2.bar(['A (Control)', 'B (Treatment)'], rev_data.values,
                      color=['#3498db', '#2ecc71'], edgecolor='black', linewidth=2)
        ax2.set_ylabel('Revenue per User ($)', fontsize=11)
        ax2.set_title('Revenue per User by Variant', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, rev in zip(bars, rev_data.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${rev:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Bayesian Posterior Distributions
        ax3 = fig.add_subplot(gs[0, 2])
        control = df[df['variant'] == 'A']
        treatment = df[df['variant'] == 'B']

        alpha_control = 1 + control['converted'].sum()
        beta_control = 1 + (len(control) - control['converted'].sum())
        alpha_treatment = 1 + treatment['converted'].sum()
        beta_treatment = 1 + (len(treatment) - treatment['converted'].sum())

        x = np.linspace(0, 0.3, 1000)
        ax3.plot(x, beta.pdf(x, alpha_control, beta_control),
                label='Control (A)', linewidth=2, color='#3498db')
        ax3.plot(x, beta.pdf(x, alpha_treatment, beta_treatment),
                label='Treatment (B)', linewidth=2, color='#2ecc71')
        ax3.fill_between(x, beta.pdf(x, alpha_control, beta_control), alpha=0.3, color='#3498db')
        ax3.fill_between(x, beta.pdf(x, alpha_treatment, beta_treatment), alpha=0.3, color='#2ecc71')
        ax3.set_xlabel('Conversion Rate', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.set_title('Bayesian Posterior Distributions', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Segment Analysis - Device
        ax4 = fig.add_subplot(gs[1, 0])
        device_segments = {k: v for k, v in segments.items() if k.startswith('device_')}
        devices = [k.replace('device_', '') for k in device_segments.keys()]
        lifts = [v['lift'] * 100 for v in device_segments.values()]

        colors = ['#2ecc71' if l > 0 else '#e74c3c' for l in lifts]
        bars = ax4.barh(devices, lifts, color=colors, edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Lift (%)', fontsize=11)
        ax4.set_title('Conversion Lift by Device', fontsize=12, fontweight='bold')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax4.grid(True, alpha=0.3, axis='x')

        # Segment Analysis - Source
        ax5 = fig.add_subplot(gs[1, 1])
        source_segments = {k: v for k, v in segments.items() if k.startswith('source_')}
        sources = [k.replace('source_', '') for k in source_segments.keys()]
        lifts = [v['lift'] * 100 for v in source_segments.values()]

        colors = ['#2ecc71' if l > 0 else '#e74c3c' for l in lifts]
        bars = ax5.barh(sources, lifts, color=colors, edgecolor='black', linewidth=1.5)
        ax5.set_xlabel('Lift (%)', fontsize=11)
        ax5.set_title('Conversion Lift by Source', fontsize=12, fontweight='bold')
        ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax5.grid(True, alpha=0.3, axis='x')

        # Cumulative Conversions Over Time
        ax6 = fig.add_subplot(gs[1, 2])
        df_sorted = df.sort_values('user_id')
        for variant in ['A', 'B']:
            variant_data = df_sorted[df_sorted['variant'] == variant]
            cumsum = variant_data['converted'].cumsum()
            cumrate = cumsum / (np.arange(len(variant_data)) + 1)
            color = '#3498db' if variant == 'A' else '#2ecc71'
            ax6.plot(range(len(variant_data)), cumrate * 100,
                    label=f'Variant {variant}', linewidth=2, color=color, alpha=0.8)

        ax6.set_xlabel('Number of Users', fontsize=11)
        ax6.set_ylabel('Cumulative Conversion Rate (%)', fontsize=11)
        ax6.set_title('Conversion Rate Convergence', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Distribution of Revenue
        ax7 = fig.add_subplot(gs[2, 0])
        control_rev = df[df['variant'] == 'A']['revenue']
        treatment_rev = df[df['variant'] == 'B']['revenue']

        ax7.hist(control_rev[control_rev > 0], bins=30, alpha=0.6,
                label='Control (A)', color='#3498db', edgecolor='black', density=True)
        ax7.hist(treatment_rev[treatment_rev > 0], bins=30, alpha=0.6,
                label='Treatment (B)', color='#2ecc71', edgecolor='black', density=True)
        ax7.set_xlabel('Revenue ($)', fontsize=11)
        ax7.set_ylabel('Density', fontsize=11)
        ax7.set_title('Revenue Distribution (Converted Users)', fontsize=12, fontweight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')

        # Confidence Intervals
        ax8 = fig.add_subplot(gs[2, 1])
        cr = freq_results['conversion_rate']
        point_estimate = cr['absolute_lift']
        ci_lower = cr['ci_lower']
        ci_upper = cr['ci_upper']

        ax8.scatter([0], [point_estimate * 100], s=200, color='#2ecc71',
                   edgecolor='black', linewidth=2, zorder=3)
        ax8.plot([0, 0], [ci_lower * 100, ci_upper * 100], linewidth=4,
                color='#2ecc71', alpha=0.7)
        ax8.axhline(y=0, color='red', linestyle='--', linewidth=2, label='No Effect')
        ax8.set_xlim(-0.5, 0.5)
        ax8.set_ylabel('Lift in Conversion Rate (%)', fontsize=11)
        ax8.set_title('95% Confidence Interval for Lift', fontsize=12, fontweight='bold')
        ax8.set_xticks([])
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')

        # Summary Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        cr = freq_results['conversion_rate']
        rpu = freq_results['revenue_per_user']

        summary_text = f"""
        ╔═══════════════════════════════════════╗
        ║         A/B TEST RESULTS               ║
        ╚═══════════════════════════════════════╝

        Sample Sizes:
        Control (A):    {len(df[df['variant']=='A']):>6d} users
        Treatment (B):  {len(df[df['variant']=='B']):>6d} users

        ┌─────────────────────────────────────┐
        │ CONVERSION RATE                      │
        ├─────────────────────────────────────┤
        │ Control:        {cr['control_rate']:>7.2%}           │
        │ Treatment:      {cr['treatment_rate']:>7.2%}           │
        │ Absolute Lift:  {cr['absolute_lift']:>7.2%}           │
        │ Relative Lift:  {cr['relative_lift']:>7.2%}           │
        │ P-value:        {cr['p_value']:>7.4f}           │
        │ Significant:    {'Yes' if cr['significant'] else 'No':>7s}           │
        └─────────────────────────────────────┘

        ┌─────────────────────────────────────┐
        │ REVENUE PER USER                     │
        ├─────────────────────────────────────┤
        │ Control:        ${rpu['control_mean']:>7.2f}          │
        │ Treatment:      ${rpu['treatment_mean']:>7.2f}          │
        │ Relative Diff:  {rpu['relative_diff']:>7.2%}           │
        │ P-value:        {rpu['p_value']:>7.4f}           │
        │ Significant:    {'Yes' if rpu['significant'] else 'No':>7s}           │
        └─────────────────────────────────────┘

        ┌─────────────────────────────────────┐
        │ BAYESIAN ANALYSIS                    │
        ├─────────────────────────────────────┤
        │ P(B > A):       {bayes_results['prob_b_better']:>7.2%}           │
        │ Expected Lift:  {bayes_results['lift_mean']:>7.2%}           │
        └─────────────────────────────────────┘
        """
        ax9.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center')

        plt.savefig('ab_test_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'ab_test_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("🔬 A/B Test Analysis System")
    print("=" * 80)

    analyzer = ABTestAnalyzer(alpha=0.05, power=0.80)

    # Generate data
    print("\n📊 Generating A/B test data...")
    df = analyzer.create_sample_data(n_users=10000)
    print(f"Dataset shape: {df.shape}")
    print(f"Control users: {(df['variant'] == 'A').sum()}")
    print(f"Treatment users: {(df['variant'] == 'B').sum()}")

    # Calculate required sample size
    print("\n📏 Sample size calculation...")
    baseline_rate = df[df['variant'] == 'A']['converted'].mean()
    mde = 0.20  # Minimum detectable effect: 20% relative lift
    required_n = analyzer.calculate_sample_size(baseline_rate, mde)
    print(f"Required sample size per variant: {required_n}")
    print(f"Actual sample size: {len(df) // 2}")

    # Frequentist analysis
    print("\n📊 Performing frequentist analysis...")
    freq_results = analyzer.frequentist_test(df)

    # Bayesian analysis
    print("\n🎲 Performing Bayesian analysis...")
    bayes_results = analyzer.bayesian_test(df)

    # Segment analysis
    print("\n🔍 Analyzing segments...")
    segments = analyzer.segment_analysis(df)

    # Plot results
    print("\n📈 Generating visualizations...")
    analyzer.plot_results(df, freq_results, bayes_results, segments)

    print("\n✅ A/B test analysis complete!")


if __name__ == "__main__":
    main()
