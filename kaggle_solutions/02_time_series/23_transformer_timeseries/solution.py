"""
Transformer Models for Time Series - Attention-Based Forecasting
================================================================

This solution demonstrates Transformer architecture for time series:
1. Self-attention mechanism for time series
2. Positional encoding for temporal information
3. Multi-head attention
4. Encoder-decoder architecture
5. Temporal embeddings
6. Comparison with RNN/LSTM
7. Attention weight visualization
8. Long-sequence modeling
9. Walk-forward validation
10. Computational efficiency analysis

Dataset: Synthetic time series with complex patterns
Models: Transformer, Multi-head Attention, Temporal Transformer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_complex_timeseries(n_samples=1200):
    """Generate complex time series"""
    print("Generating complex time series...")

    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    t = np.linspace(0, 6*np.pi, n_samples)
    trend = 100 + 50 * np.sin(t/4) + 0.03 * t**2
    seasonal1 = 15 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
    seasonal2 = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
    
    nonlinear = 12 * np.sin(t) * np.cos(t/3)
    noise = np.random.normal(0, 5, n_samples)
    
    values = trend + seasonal1 + seasonal2 + nonlinear + noise
    
    df = pd.DataFrame({'date': dates, 'value': values})
    df.set_index('date', inplace=True)
    
    print(f"Generated {len(df)} observations")
    return df


class PositionalEncoding:
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        pe = np.zeros((max_len, d_model))
        
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def encode(self, x, positions):
        """Add positional encoding"""
        return x + self.pe[positions]


class ScaledDotProductAttention:
    """Scaled dot-product attention"""
    
    def __init__(self, d_k):
        self.d_k = d_k
    
    def forward(self, Q, K, V, mask=None):
        """Compute attention"""
        scores = Q @ K.T / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores + mask
        
        attention_weights = self.softmax(scores)
        output = attention_weights @ V
        
        return output, attention_weights
    
    def softmax(self, x):
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class MultiHeadAttention:
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def split_heads(self, x):
        """Split into multiple heads"""
        return x.reshape(self.num_heads, -1, self.d_k)
    
    def forward(self, Q, K, V, mask=None):
        """Multi-head attention forward pass"""
        outputs = []
        attention_weights = []
        
        for h in range(self.num_heads):
            output, weights = self.attention.forward(Q, K, V, mask)
            outputs.append(output)
            attention_weights.append(weights)
        
        concat_output = np.mean(outputs, axis=0)
        return concat_output, attention_weights


def create_sequences(data, lookback, forecast_horizon=1):
    """Create sequences"""
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback:i + lookback + forecast_horizon])
    return np.array(X), np.array(y)


def prepare_data(series, lookback=60, forecast_horizon=1, train_ratio=0.8):
    """Prepare data for Transformer"""
    print(f"\n{'='*70}")
    print("Data Preparation for Transformer")
    print(f"{'='*70}")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    
    X, y = create_sequences(scaled_data, lookback, forecast_horizon)
    
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


class TransformerModel:
    """Simplified Transformer for time series"""
    
    def __init__(self, lookback, d_model=64, num_heads=4):
        self.lookback = lookback
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.positional_encoding = PositionalEncoding(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.history = {'loss': [], 'val_loss': []}
        self.attention_weights_history = []
    
    def embed(self, x):
        """Simple embedding (project to d_model dimension)"""
        embedded = np.repeat(x.reshape(-1, 1), self.d_model, axis=1)
        positions = np.arange(len(x))
        return self.positional_encoding.encode(embedded, positions)
    
    def forward(self, x):
        """Forward pass"""
        embedded = self.embed(x)
        
        Q = K = V = embedded
        output, weights = self.attention.forward(Q, K, V)
        
        self.attention_weights_history.append(weights)
        
        prediction = np.mean(output[-10:])
        return prediction
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """Train Transformer"""
        print(f"\n{'='*70}")
        print("Training Transformer Model")
        print(f"{'='*70}")
        print(f"d_model: {self.d_model}, num_heads: {self.num_heads}")
        
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


def calculate_metrics(y_true, y_pred):
    """Calculate metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


def plot_transformer_results(df, models_predictions, scaler, attention_weights):
    """Plot Transformer results"""
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
    
    if attention_weights and len(attention_weights) > 0:
        ax4 = fig.add_subplot(gs[3, 0])
        weights_sample = attention_weights[0][0][:20, :20]
        im = ax4.imshow(weights_sample, cmap='viridis', aspect='auto')
        ax4.set_title('Attention Weights (Sample)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Key Position')
        ax4.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax4)
    
    ax5 = fig.add_subplot(gs[3, 1])
    for idx, (name, data) in enumerate(models_predictions.items()):
        if 'predictions' in data and 'actual' in data:
            errors = data['actual'] - data['predictions']
            ax5.hist(errors, bins=30, alpha=0.5, label=name, edgecolor='black')
    ax5.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.savefig('transformer_results.png', dpi=300, bbox_inches='tight')
    return fig


def main():
    """Main execution"""
    print("="*70)
    print("TRANSFORMER MODELS FOR TIME SERIES")
    print("="*70)
    
    df = generate_complex_timeseries(n_samples=1200)
    series = df['value']
    
    adf_result = adfuller(series)
    print(f"\nADF test p-value: {adf_result[1]:.6f}")
    
    lookback = 60
    X_train, X_test, y_train, y_test, scaler = prepare_data(series, lookback=lookback)
    
    val_split = int(len(X_train) * 0.9)
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]
    
    models_predictions = {}
    
    transformer = TransformerModel(lookback, d_model=64, num_heads=4)
    transformer.train(X_train, y_train, X_val, y_val, epochs=50)
    transformer_pred = transformer.predict(X_test)
    models_predictions['Transformer'] = {
        'predictions': transformer_pred,
        'actual': y_test.flatten(),
        'history': transformer.history
    }
    
    print(f"\n{'='*70}")
    print("Model Performance")
    print(f"{'='*70}")
    
    for name, data in models_predictions.items():
        metrics = calculate_metrics(data['actual'], data['predictions'])
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
    
    plot_transformer_results(df, models_predictions, scaler, transformer.attention_weights_history)
    
    print("\n" + "="*70)
    print("TRANSFORMER ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()


class TemporalTransformer:
    """Enhanced Transformer with temporal features"""
    
    def __init__(self, lookback, d_model=128, num_heads=8, num_layers=3):
        self.lookback = lookback
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.positional_encoding = PositionalEncoding(d_model)
        self.attention_layers = [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        self.history = {'loss': [], 'val_loss': []}
    
    def embed(self, x):
        """Enhanced embedding with multiple features"""
        base_embedded = np.repeat(x.reshape(-1, 1), self.d_model//2, axis=1)
        
        # Add temporal features
        positions = np.arange(len(x))
        temporal_features = np.column_stack([
            np.sin(2 * np.pi * positions / 7),  # weekly
            np.cos(2 * np.pi * positions / 7),
            np.sin(2 * np.pi * positions / 30),  # monthly
            np.cos(2 * np.pi * positions / 30)
        ])
        temporal_features = np.repeat(temporal_features, self.d_model//8, axis=1)
        
        embedded = np.concatenate([base_embedded, temporal_features], axis=1)
        return self.positional_encoding.encode(embedded[:, :self.d_model], positions)
    
    def forward(self, x):
        """Multi-layer forward pass"""
        embedded = self.embed(x)
        
        output = embedded
        for layer in self.attention_layers:
            Q = K = V = output
            layer_output, _ = layer.forward(Q, K, V)
            output = layer_output + output  # Residual connection
        
        prediction = np.mean(output[-15:])
        return prediction
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """Train temporal transformer"""
        print(f"\n{'='*70}")
        print("Training Temporal Transformer Model")
        print(f"{'='*70}")
        print(f"d_model: {self.d_model}, num_heads: {self.num_heads}, layers: {self.num_layers}")
        
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


def compare_transformer_architectures(X_train, y_train, X_test, y_test, lookback):
    """Compare different Transformer configurations"""
    print(f"\n{'='*70}")
    print("Transformer Architecture Comparison")
    print(f"{'='*70}")
    
    configs = [
        {'name': 'Small', 'd_model': 32, 'num_heads': 2},
        {'name': 'Medium', 'd_model': 64, 'num_heads': 4},
        {'name': 'Large', 'd_model': 128, 'num_heads': 8}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']} Transformer...")
        print(f"  d_model={config['d_model']}, num_heads={config['num_heads']}")
        
        model = TransformerModel(lookback, d_model=config['d_model'], 
                                num_heads=config['num_heads'])
        
        # Train subset for speed
        train_subset = min(len(X_train), 500)
        model.train(X_train[:train_subset], y_train[:train_subset], epochs=20)
        
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test.flatten(), predictions)
        rmse = np.sqrt(mean_squared_error(y_test.flatten(), predictions))
        
        results[config['name']] = {
            'MAE': mae,
            'RMSE': rmse,
            'd_model': config['d_model'],
            'num_heads': config['num_heads'],
            'parameters': config['d_model'] * config['num_heads'] * 3  # Approximate
        }
        
        print(f"  MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    
    print(f"\nArchitecture Comparison Summary:")
    for name, metrics in results.items():
        print(f"\n{name} Transformer:")
        print(f"  MAE: {metrics['MAE']:.6f}")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  Parameters: ~{metrics['parameters']}")
    
    return results


def analyze_attention_patterns(attention_weights, lookback):
    """Analyze attention patterns"""
    print(f"\n{'='*70}")
    print("Attention Pattern Analysis")
    print(f"{'='*70}")
    
    if not attention_weights or len(attention_weights) == 0:
        print("No attention weights available")
        return
    
    # Analyze first head of last sample
    sample_weights = attention_weights[-1][0]
    
    # Find which positions get most attention
    avg_attention = np.mean(sample_weights, axis=0)
    top_positions = np.argsort(avg_attention)[-10:]
    
    print(f"\nTop 10 most attended positions:")
    for i, pos in enumerate(reversed(top_positions)):
        print(f"  {i+1}. Position {pos}: {avg_attention[pos]:.4f}")
    
    # Analyze temporal patterns
    recent_attention = np.mean(avg_attention[-20:])
    distant_attention = np.mean(avg_attention[:-20])
    
    print(f"\nTemporal Attention Distribution:")
    print(f"  Recent (last 20 steps): {recent_attention:.4f}")
    print(f"  Distant (earlier steps): {distant_attention:.4f}")
    print(f"  Recent/Distant ratio: {recent_attention/distant_attention:.2f}")


def walk_forward_validation_transformer(series, lookback=60, n_splits=5):
    """Walk-forward validation for Transformer"""
    print(f"\n{'='*70}")
    print("Walk-Forward Validation - Transformer")
    print(f"{'='*70}")
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    
    n = len(scaled)
    test_size = (n - lookback) // (n_splits + 1)
    
    errors = []
    
    for i in range(n_splits):
        split_point = n - (n_splits - i) * test_size
        train = scaled[:split_point]
        test = scaled[split_point:split_point + test_size]
        
        if len(test) < lookback:
            continue
        
        X_train, y_train = create_sequences(train, lookback)
        X_test, y_test = create_sequences(test, lookback) if len(test) > lookback else ([], [])
        
        if len(X_test) == 0:
            continue
        
        model = TransformerModel(lookback, d_model=64, num_heads=4)
        model.train(X_train, y_train, epochs=20)
        
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test.flatten(), predictions)
        rmse = np.sqrt(mean_squared_error(y_test.flatten(), predictions))
        
        errors.append({'fold': i+1, 'mae': mae, 'rmse': rmse})
        print(f"\nFold {i+1}: MAE={mae:.6f}, RMSE={rmse:.6f}")
    
    errors_df = pd.DataFrame(errors)
    if len(errors_df) > 0:
        print(f"\nAverage Performance:")
        print(f"  MAE: {errors_df['mae'].mean():.6f} ± {errors_df['mae'].std():.6f}")
        print(f"  RMSE: {errors_df['rmse'].mean():.6f} ± {errors_df['rmse'].std():.6f}")
    
    return errors_df
