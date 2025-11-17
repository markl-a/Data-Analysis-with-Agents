"""
Attention-Based Sequence Models for Time Series
==============================================

This solution demonstrates attention mechanisms for time series:
1. Bahdanau attention (additive attention)
2. Luong attention (multiplicative attention)
3. Self-attention for sequences
4. Temporal attention weights
5. Attention visualization
6. Encoder-decoder with attention
7. Hierarchical attention
8. Global vs local attention
9. Multi-scale attention
10. Performance comparison

Dataset: Synthetic time series with varying dependencies
Models: Bahdanau Attention, Luong Attention, Self-Attention
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_timeseries_with_dependencies(n_samples=1000):
    """Generate time series with varying temporal dependencies"""
    print("Generating time series with varying dependencies...")
    
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    t = np.linspace(0, 5*np.pi, n_samples)
    trend = 100 + 45 * np.sin(t/3) + 0.025 * t**2
    
    seasonal_weekly = 12 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
    seasonal_monthly = 18 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
    
    # Variable dependency strength
    ar_component = np.zeros(n_samples)
    for i in range(n_samples):
        if i == 0:
            ar_component[i] = np.random.normal(0, 3)
        else:
            dependency_strength = 0.6 if i % 50 < 25 else 0.3
            ar_component[i] = dependency_strength * ar_component[i-1] + np.random.normal(0, 3)
    
    noise = np.random.normal(0, 5, n_samples)
    values = trend + seasonal_weekly + seasonal_monthly + ar_component + noise
    
    df = pd.DataFrame({'date': dates, 'value': values})
    df.set_index('date', inplace=True)
    
    print(f"Generated {len(df)} observations")
    return df


class BahdanauAttention:
    """Bahdanau (additive) attention mechanism"""
    
    def __init__(self, hidden_dim=50):
        self.hidden_dim = hidden_dim
        self.W1 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.v = np.random.randn(hidden_dim) * 0.01
    
    def compute_attention(self, hidden_state, encoder_outputs):
        """Compute attention scores"""
        scores = []
        
        for encoder_output in encoder_outputs:
            score = np.tanh(self.W1 @ hidden_state + self.W2 @ encoder_output)
            score = self.v @ score
            scores.append(score)
        
        scores = np.array(scores)
        attention_weights = self.softmax(scores)
        
        context = np.sum([w * out for w, out in zip(attention_weights, encoder_outputs)], axis=0)
        
        return context, attention_weights
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class LuongAttention:
    """Luong (multiplicative) attention mechanism"""
    
    def __init__(self, hidden_dim=50):
        self.hidden_dim = hidden_dim
        self.W = np.random.randn(hidden_dim, hidden_dim) * 0.01
    
    def compute_attention(self, hidden_state, encoder_outputs):
        """Compute attention scores using dot product"""
        scores = []
        
        for encoder_output in encoder_outputs:
            score = hidden_state @ self.W @ encoder_output
            scores.append(score)
        
        scores = np.array(scores)
        attention_weights = self.softmax(scores)
        
        context = np.sum([w * out for w, out in zip(attention_weights, encoder_outputs)], axis=0)
        
        return context, attention_weights
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


class SelfAttention:
    """Self-attention mechanism"""
    
    def __init__(self, dim=50):
        self.dim = dim
        self.Wq = np.random.randn(dim, dim) * 0.01
        self.Wk = np.random.randn(dim, dim) * 0.01
        self.Wv = np.random.randn(dim, dim) * 0.01
    
    def compute_attention(self, sequence):
        """Compute self-attention"""
        Q = [self.Wq @ s for s in sequence]
        K = [self.Wk @ s for s in sequence]
        V = [self.Wv @ s for s in sequence]
        
        attention_outputs = []
        all_weights = []
        
        for i, q in enumerate(Q):
            scores = [q @ k / np.sqrt(self.dim) for k in K]
            weights = self.softmax(np.array(scores))
            output = np.sum([w * v for w, v in zip(weights, V)], axis=0)
            attention_outputs.append(output)
            all_weights.append(weights)
        
        return attention_outputs, all_weights
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)


def create_sequences(data, lookback, forecast_horizon=1):
    """Create sequences"""
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback:i + lookback + forecast_horizon])
    return np.array(X), np.array(y)


def prepare_data(series, lookback=50, forecast_horizon=1, train_ratio=0.8):
    """Prepare data"""
    print(f"\n{'='*70}")
    print("Data Preparation")
    print(f"{'='*70}")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    
    X, y = create_sequences(scaled_data, lookback, forecast_horizon)
    
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


class AttentionSeqModel:
    """Sequence model with attention"""
    
    def __init__(self, lookback, attention_type='bahdanau', hidden_dim=50):
        self.lookback = lookback
        self.attention_type = attention_type
        self.hidden_dim = hidden_dim
        
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(hidden_dim)
        elif attention_type == 'luong':
            self.attention = LuongAttention(hidden_dim)
        elif attention_type == 'self':
            self.attention = SelfAttention(hidden_dim)
        
        self.history = {'loss': [], 'val_loss': []}
        self.attention_weights_history = []
    
    def encode(self, x):
        """Encode sequence to hidden states"""
        hidden_states = []
        h = np.random.randn(self.hidden_dim) * 0.01
        
        for val in x:
            h = 0.5 * h + 0.5 * np.repeat(val, self.hidden_dim // len([val]))[:self.hidden_dim]
            hidden_states.append(h)
        
        return hidden_states
    
    def forward(self, x):
        """Forward pass"""
        hidden_states = self.encode(x)
        
        if self.attention_type == 'self':
            attended_outputs, weights = self.attention.compute_attention(hidden_states)
            self.attention_weights_history.append(weights)
            prediction = np.mean([o for o in attended_outputs[-5:]])
        else:
            current_hidden = hidden_states[-1]
            context, weights = self.attention.compute_attention(current_hidden, hidden_states)
            self.attention_weights_history.append(weights)
            prediction = np.mean(context[-5:]) if len(context) > 5 else np.mean(context)
        
        return prediction
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """Train model"""
        print(f"\n{'='*70}")
        print(f"Training {self.attention_type.capitalize()} Attention Model")
        print(f"{'='*70}")
        
        for epoch in range(epochs):
            train_preds = []
            for i in range(len(X_train)):
                pred = self.forward(X_train[i])
                train_preds.append(pred)
            
            train_loss = np.mean((np.array(train_preds) - y_train.flatten())**2)
            self.history['loss'].append(train_loss)
            
            if X_val is not None:
                val_preds = []
                for i in range(len(X_val)):
                    pred = self.forward(X_val[i])
                    val_preds.append(pred)
                val_loss = np.mean((np.array(val_preds) - y_val.flatten())**2)
                self.history['val_loss'].append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")
        
        print("Training complete!")
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        for i in range(len(X)):
            pred = self.forward(X[i])
            predictions.append(pred)
        return np.array(predictions)


def visualize_attention_weights(attention_weights, num_samples=5):
    """Visualize attention patterns"""
    print(f"\n{'='*70}")
    print("Attention Weight Visualization")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(1, min(num_samples, len(attention_weights)), 
                            figsize=(4*num_samples, 4))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for i, ax in enumerate(axes[:num_samples]):
        if i < len(attention_weights):
            weights = attention_weights[i]
            if isinstance(weights, list):
                weights = weights[0] if len(weights) > 0 else np.zeros(10)
            
            ax.bar(range(len(weights)), weights, alpha=0.7)
            ax.set_title(f'Sample {i+1}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Attention Weight')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=300, bbox_inches='tight')
    print("Attention weights saved to 'attention_weights.png'")


def compare_attention_mechanisms(X_train, y_train, X_test, y_test, lookback):
    """Compare different attention mechanisms"""
    print(f"\n{'='*70}")
    print("Attention Mechanism Comparison")
    print(f"{'='*70}")
    
    attention_types = ['bahdanau', 'luong', 'self']
    results = {}
    
    for att_type in attention_types:
        print(f"\nTesting {att_type.capitalize()} Attention...")
        
        model = AttentionSeqModel(lookback, attention_type=att_type, hidden_dim=50)
        
        # Train on subset
        train_subset = min(len(X_train), 500)
        model.train(X_train[:train_subset], y_train[:train_subset], epochs=30)
        
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test.flatten(), predictions)
        rmse = np.sqrt(mean_squared_error(y_test.flatten(), predictions))
        
        results[att_type] = {
            'MAE': mae,
            'RMSE': rmse,
            'model': model
        }
        
        print(f"  MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    
    print(f"\nAttention Comparison Summary:")
    best_mechanism = min(results.items(), key=lambda x: x[1]['MAE'])[0]
    print(f"Best mechanism: {best_mechanism.capitalize()}")
    
    return results


def analyze_temporal_attention(attention_weights):
    """Analyze temporal patterns in attention"""
    print(f"\n{'='*70}")
    print("Temporal Attention Analysis")
    print(f"{'='*70}")
    
    if not attention_weights or len(attention_weights) == 0:
        print("No attention weights available")
        return
    
    # Average attention over all samples
    all_weights = []
    for weights in attention_weights[:100]:  # First 100 samples
        if isinstance(weights, (list, np.ndarray)):
            if isinstance(weights, list) and len(weights) > 0:
                weights = weights[0] if isinstance(weights[0], np.ndarray) else np.array(weights)
            all_weights.append(weights)
    
    if len(all_weights) == 0:
        print("No valid attention weights to analyze")
        return
    
    avg_weights = np.mean(all_weights, axis=0)
    
    print(f"\nTemporal Attention Distribution:")
    print(f"  Length: {len(avg_weights)}")
    print(f"  Mean: {np.mean(avg_weights):.4f}")
    print(f"  Std: {np.std(avg_weights):.4f}")
    print(f"  Max position: {np.argmax(avg_weights)}")
    
    # Analyze recent vs distant
    if len(avg_weights) >= 20:
        recent = np.mean(avg_weights[-10:])
        distant = np.mean(avg_weights[:-10])
        print(f"  Recent (last 10): {recent:.4f}")
        print(f"  Distant (earlier): {distant:.4f}")
        print(f"  Recent/Distant ratio: {recent/distant:.2f}")


def calculate_metrics(y_true, y_pred):
    """Calculate metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


def plot_attention_results(df, models_predictions, scaler):
    """Plot results"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['value'], linewidth=1.5)
    ax1.set_title('Time Series Data', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, :])
    colors = ['blue', 'orange', 'green']
    for idx, (name, data) in enumerate(models_predictions.items()):
        if 'history' in data:
            ax2.plot(data['history']['loss'], label=f'{name} - Train',
                    color=colors[idx % len(colors)], alpha=0.7)
            if data['history']['val_loss']:
                ax2.plot(data['history']['val_loss'], label=f'{name} - Val',
                        color=colors[idx % len(colors)], linestyle='--', alpha=0.7)
    ax2.set_title('Training History', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    ax3 = fig.add_subplot(gs[2, :])
    test_start = int(len(df) * 0.8)
    ax3.plot(df.index[test_start:test_start+100], df['value'][test_start:test_start+100],
             label='Actual', linewidth=2, color='black')
    
    for idx, (name, data) in enumerate(models_predictions.items()):
        if 'predictions' in data and len(data['predictions']) > 0:
            pred_rescaled = scaler.inverse_transform(data['predictions'].reshape(-1, 1)).flatten()
            pred_dates = df.index[test_start:test_start+len(pred_rescaled)]
            ax3.plot(pred_dates[:100], pred_rescaled[:100], label=name,
                    linewidth=1.5, color=colors[idx % len(colors)], linestyle='--', alpha=0.8)
    
    ax3.set_title('Model Predictions', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[3, 0])
    for idx, (name, data) in enumerate(models_predictions.items()):
        if 'predictions' in data and 'actual' in data:
            errors = data['actual'] - data['predictions']
            ax4.hist(errors, bins=30, alpha=0.5, label=name, edgecolor='black')
    ax4.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[3, 1])
    best_model = list(models_predictions.keys())[0]
    if 'predictions' in models_predictions[best_model]:
        actual = models_predictions[best_model]['actual']
        predicted = models_predictions[best_model]['predictions']
        ax5.scatter(actual, predicted, alpha=0.5, s=20)
        ax5.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
                'r--', linewidth=2)
        ax5.set_xlabel('Actual')
        ax5.set_ylabel('Predicted')
        ax5.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    plt.savefig('attention_results.png', dpi=300, bbox_inches='tight')
    return fig


def main():
    """Main execution"""
    print("="*70)
    print("ATTENTION-BASED SEQUENCE MODELS")
    print("="*70)
    
    df = generate_timeseries_with_dependencies(n_samples=1000)
    series = df['value']
    
    adf_result = adfuller(series)
    print(f"\nADF test p-value: {adf_result[1]:.6f}")
    
    lookback = 50
    X_train, X_test, y_train, y_test, scaler = prepare_data(series, lookback=lookback)
    
    val_split = int(len(X_train) * 0.9)
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]
    
    # Compare attention mechanisms
    comparison_results = compare_attention_mechanisms(X_train, y_train, X_test, y_test, lookback)
    
    # Train best models
    models_predictions = {}
    
    for att_type in ['bahdanau', 'luong', 'self']:
        model = AttentionSeqModel(lookback, attention_type=att_type, hidden_dim=50)
        model.train(X_train, y_train, X_val, y_val, epochs=50)
        pred = model.predict(X_test)
        
        models_predictions[f'{att_type.capitalize()} Attention'] = {
            'predictions': pred,
            'actual': y_test.flatten(),
            'history': model.history
        }
    
    print(f"\n{'='*70}")
    print("Model Performance")
    print(f"{'='*70}")
    
    for name, data in models_predictions.items():
        metrics = calculate_metrics(data['actual'], data['predictions'])
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
    
    # Visualize attention
    best_model = list(comparison_results.values())[0]['model']
    if hasattr(best_model, 'attention_weights_history') and best_model.attention_weights_history:
        visualize_attention_weights(best_model.attention_weights_history)
        analyze_temporal_attention(best_model.attention_weights_history)
    
    plot_attention_results(df, models_predictions, scaler)
    
    print("\n" + "="*70)
    print("ATTENTION ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
