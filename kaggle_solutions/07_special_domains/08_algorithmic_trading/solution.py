"""
Algorithmic Trading Strategy System
====================================

Problem: Develop and backtest algorithmic trading strategies using
technical indicators and machine learning for profitable trading

Kaggle-style competition: Stock Market Prediction
Difficulty: ⭐⭐⭐⭐⭐

This solution demonstrates:
- Technical indicator calculation
- ML-based price movement prediction
- Trading strategy backtesting
- Risk management and position sizing
- Performance metrics (Sharpe ratio, drawdown)
- Portfolio optimization
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class AlgorithmicTradingSystem:
    """ML-based algorithmic trading strategy"""

    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.models = {}
        self.scaler = StandardScaler()

    def create_sample_data(self, n_days=1000):
        """Generate realistic stock price data"""
        np.random.seed(42)

        # Simulate price using geometric Brownian motion
        dt = 1/252  # Daily time step
        mu = 0.08  # Annual drift (8%)
        sigma = 0.20  # Annual volatility (20%)

        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        price = [100]  # Starting price

        for _ in range(n_days - 1):
            drift = mu * dt
            shock = sigma * np.sqrt(dt) * np.random.normal()
            price.append(price[-1] * (1 + drift + shock))

        df = pd.DataFrame({
            'date': dates,
            'close': price
        })

        # Add OHLV data
        df['open'] = df['close'] * (1 + np.random.normal(0, 0.005, n_days))
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, n_days))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, n_days))
        df['volume'] = np.random.lognormal(15, 0.5, n_days).astype(int)

        return df

    def calculate_technical_indicators(self, df):
        """Calculate technical trading indicators"""
        df = df.copy()

        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

        # Price momentum
        df['momentum_5'] = df['close'].pct_change(periods=5)
        df['momentum_10'] = df['close'].pct_change(periods=10)
        df['momentum_20'] = df['close'].pct_change(periods=20)

        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Price position relative to bands
        df['price_to_sma20'] = df['close'] / df['sma_20']
        df['price_to_bb_upper'] = df['close'] / df['bb_upper']
        df['price_to_bb_lower'] = df['close'] / df['bb_lower']

        return df

    def create_target_variable(self, df, horizon=5):
        """Create target: 1 if price goes up in next N days, 0 otherwise"""
        df = df.copy()
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        df['target'] = (df['future_return'] > 0.02).astype(int)  # 2% threshold
        return df

    def train_models(self, X, y):
        """Train trading models using time-series cross-validation"""
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)

        # Remove NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]

        print(f"Training samples: {len(X_clean)}")
        print(f"Positive samples (buy signals): {y_clean.sum()} ({y_clean.mean():.1%})")

        # Initialize models
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10,
                                                   random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100,
                                                           learning_rate=0.1,
                                                           max_depth=5, random_state=42)
        }

        results = {}
        for name, model in models_config.items():
            print(f"\nTraining {name}...")

            cv_scores = []
            for train_idx, val_idx in tscv.split(X_clean):
                X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]

                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)

                # Train and evaluate
                model.fit(X_train_scaled, y_train)
                score = model.score(X_val_scaled, y_val)
                cv_scores.append(score)

            # Final training on all data
            X_scaled = self.scaler.fit_transform(X_clean)
            model.fit(X_scaled, y_clean)

            # Predictions
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]

            results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'accuracy': accuracy_score(y_clean, y_pred),
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores)
            }

        return results, X_clean, y_clean

    def backtest_strategy(self, df, signals, transaction_cost=0.001):
        """Backtest trading strategy"""
        df = df.copy()
        df['signal'] = signals
        df['position'] = df['signal'].shift(1)  # Trade next day
        df['returns'] = df['close'].pct_change()

        # Strategy returns (with transaction costs)
        df['strategy_returns'] = df['position'] * df['returns']
        df['trade'] = df['position'].diff().abs()
        df['costs'] = df['trade'] * transaction_cost
        df['strategy_returns_net'] = df['strategy_returns'] - df['costs']

        # Cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        df['cumulative_strategy'] = (1 + df['strategy_returns_net']).cumprod()

        # Portfolio value
        df['portfolio_value'] = self.initial_capital * df['cumulative_strategy']

        return df

    def calculate_performance_metrics(self, df):
        """Calculate trading performance metrics"""
        # Returns
        total_return = (df['cumulative_strategy'].iloc[-1] - 1) * 100
        buy_hold_return = (df['cumulative_returns'].iloc[-1] - 1) * 100

        # Sharpe Ratio (annualized)
        returns_std = df['strategy_returns_net'].std() * np.sqrt(252)
        returns_mean = df['strategy_returns_net'].mean() * 252
        sharpe_ratio = returns_mean / returns_std if returns_std > 0 else 0

        # Maximum Drawdown
        cummax = df['cumulative_strategy'].cummax()
        drawdown = (df['cumulative_strategy'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        # Win rate
        winning_trades = (df['strategy_returns_net'] > 0).sum()
        total_trades = (df['trade'] > 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = df[df['strategy_returns_net'] > 0]['strategy_returns_net'].sum()
        gross_loss = abs(df[df['strategy_returns_net'] < 0]['strategy_returns_net'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate * 100,
            'total_trades': total_trades,
            'profit_factor': profit_factor,
            'final_portfolio': df['portfolio_value'].iloc[-1]
        }

    def plot_results(self, results, df_backtest, performance):
        """Visualize trading strategy results"""
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Price and Signals
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df_backtest['date'], df_backtest['close'], label='Price', linewidth=1.5, alpha=0.7)
        buy_signals = df_backtest[df_backtest['signal'] == 1]
        sell_signals = df_backtest[df_backtest['signal'] == 0]
        ax1.scatter(buy_signals['date'], buy_signals['close'], color='green',
                   marker='^', s=100, label='Buy Signal', alpha=0.7)
        ax1.scatter(sell_signals['date'], sell_signals['close'], color='red',
                   marker='v', s=100, label='Sell Signal', alpha=0.7)
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Price', fontsize=11)
        ax1.set_title('Price Chart with Trading Signals', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cumulative Returns
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(df_backtest['date'], (df_backtest['cumulative_strategy'] - 1) * 100,
                label='Strategy', linewidth=2, color='#2ecc71')
        ax2.plot(df_backtest['date'], (df_backtest['cumulative_returns'] - 1) * 100,
                label='Buy & Hold', linewidth=2, color='#3498db', alpha=0.7)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Return (%)', fontsize=11)
        ax2.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Portfolio Value
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(df_backtest['date'], df_backtest['portfolio_value'],
                linewidth=2, color='#9b59b6')
        ax3.axhline(y=self.initial_capital, color='red', linestyle='--',
                   label=f'Initial: ${self.initial_capital:,.0f}', linewidth=2)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax3.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Drawdown
        ax4 = fig.add_subplot(gs[1, 2])
        cummax = df_backtest['cumulative_strategy'].cummax()
        drawdown = (df_backtest['cumulative_strategy'] - cummax) / cummax * 100
        ax4.fill_between(df_backtest['date'], drawdown, 0, color='#e74c3c', alpha=0.6)
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_ylabel('Drawdown (%)', fontsize=11)
        ax4.set_title('Strategy Drawdown', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Model Performance
        ax5 = fig.add_subplot(gs[2, 0])
        model_names = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in model_names]
        cv_means = [results[m]['cv_mean'] for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35
        ax5.bar(x - width/2, accuracies, width, label='Train Accuracy', color='#3498db')
        ax5.bar(x + width/2, cv_means, width, label='CV Accuracy', color='#2ecc71')
        ax5.set_ylabel('Accuracy', fontsize=11)
        ax5.set_title('Model Performance', fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim(0, 1.0)

        # Monthly Returns Heatmap
        ax6 = fig.add_subplot(gs[2, 1])
        df_backtest['year'] = df_backtest['date'].dt.year
        df_backtest['month'] = df_backtest['date'].dt.month
        monthly_returns = df_backtest.groupby(['year', 'month'])['strategy_returns_net'].sum() * 100
        monthly_pivot = monthly_returns.reset_index().pivot(index='month', columns='year',
                                                             values='strategy_returns_net')
        sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   ax=ax6, cbar_kws={'label': 'Return (%)'})
        ax6.set_xlabel('Year', fontsize=11)
        ax6.set_ylabel('Month', fontsize=11)
        ax6.set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')

        # Performance Summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')

        summary_text = f"""
        ╔═══════════════════════════════════════╗
        ║   TRADING STRATEGY PERFORMANCE         ║
        ╚═══════════════════════════════════════╝

        Initial Capital:    ${self.initial_capital:>12,.0f}
        Final Portfolio:    ${performance['final_portfolio']:>12,.0f}

        ┌─────────────────────────────────────┐
        │ RETURNS                              │
        ├─────────────────────────────────────┤
        │ Strategy Return:    {performance['total_return']:>8.2f}%     │
        │ Buy & Hold Return:  {performance['buy_hold_return']:>8.2f}%     │
        │ Alpha:              {performance['total_return'] - performance['buy_hold_return']:>8.2f}%     │
        └─────────────────────────────────────┘

        ┌─────────────────────────────────────┐
        │ RISK METRICS                         │
        ├─────────────────────────────────────┤
        │ Sharpe Ratio:       {performance['sharpe_ratio']:>8.2f}      │
        │ Max Drawdown:       {performance['max_drawdown']:>8.2f}%     │
        └─────────────────────────────────────┘

        ┌─────────────────────────────────────┐
        │ TRADE METRICS                        │
        ├─────────────────────────────────────┤
        │ Total Trades:       {int(performance['total_trades']):>8d}      │
        │ Win Rate:           {performance['win_rate']:>8.2f}%     │
        │ Profit Factor:      {performance['profit_factor']:>8.2f}      │
        └─────────────────────────────────────┘
        """
        ax7.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center')

        plt.savefig('algorithmic_trading_analysis.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'algorithmic_trading_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("📈 Algorithmic Trading Strategy System")
    print("=" * 80)

    trader = AlgorithmicTradingSystem(initial_capital=100000)

    # Generate data
    print("\n📊 Generating stock price data...")
    df = trader.create_sample_data(n_days=1000)
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Calculate indicators
    print("\n🔧 Calculating technical indicators...")
    df = trader.calculate_technical_indicators(df)

    # Create target variable
    df = trader.create_target_variable(df, horizon=5)

    # Prepare features
    feature_cols = [col for col in df.columns if col not in
                   ['date', 'target', 'future_return', 'open', 'high', 'low', 'close', 'volume']]
    X = df[feature_cols]
    y = df['target']

    # Train models
    print("\n🤖 Training trading models...")
    results, X_clean, y_clean = trader.train_models(X, y)

    # Get best model predictions
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_predictions = results[best_model_name]['predictions']

    # Align predictions with dataframe
    clean_indices = X.index[~(X.isna().any(axis=1) | y.isna())]
    signals = pd.Series(0, index=df.index)
    signals.loc[clean_indices] = best_predictions

    # Backtest strategy
    print("\n📊 Backtesting strategy...")
    df_backtest = trader.backtest_strategy(df, signals)

    # Calculate performance
    performance = trader.calculate_performance_metrics(df_backtest)

    # Plot results
    print("\n📈 Generating visualizations...")
    trader.plot_results(results, df_backtest, performance)

    print("\n✅ Algorithmic trading analysis complete!")


if __name__ == "__main__":
    main()
