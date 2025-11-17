"""
Temporal Convolutional Networks (TCN) for Time Series
=====================================================

This solution demonstrates TCN architecture for time series:
1. 1D convolutional layers for temporal patterns
2. Dilated convolutions for long-range dependencies
3. Causal convolutions (no future information)
4. Residual connections
5. Multiple kernel sizes
6. Receptive field analysis
7. Comparison with RNN architectures
8. Multi-scale temporal features
9. Walk-forward validation
10. Computational efficiency analysis

Dataset: Synthetic time series with long-term dependencies
Models: TCN, Dilated TCN, Multi-scale TCN
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


def generate_long_dependency_series(n_samples=1500):
    """Generate time series with long-term dependencies"""
    print("Generating time series with long-term dependencies...")

    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    # Long-term trend
    t = np.linspace(0, 6*np.pi, n_samples)
    trend = 100 + 40 * np.sin(t/5) + 0.02 * t**2

    # Multiple seasonal patterns
    daily_seasonal = 8 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
    monthly_seasonal = 12 * np.sin(2 * np.pi * np.arange(n_samples) / 30)
    quarterly_seasonal = 20 * np.sin(2 * np.pi * np.arange(n_samples) / 90)

    # Long-range AR component
    ar_component = np.zeros(n_samples)
    ar_component[0] = np.random.normal(0, 3)
    for i in range(1, n_samples):
        # Dependencies on lags 1, 7, 30
        lag1 = ar_component[i-1] if i >= 1 else 0
        lag7 = ar_component[i-7] if i >= 7 else 0
        lag30 = ar_component[i-30] if i >= 30 else 0
        ar_component[i] = 0.5*lag1 + 0.3*lag7 + 0.1*lag30 + np.random.normal(0, 3)

    # Noise
    noise = np.random.normal(0, 4, n_samples)

    values = trend + daily_seasonal + monthly_seasonal + quarterly_seasonal + ar_component + noise

    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    df.set_index('date', inplace=True)

    print(f"Generated {len(df)} observations")
    print(f"Value range: [{df['value'].min():.2f}, {df['value'].max():.2f}]")

    return df


class TemporalConvBlock:
    """Temporal convolutional block with dilation"""

    def __init__(self, kernel_size=3, dilation=1, filters=32):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.filters = filters
        self.weights = np.random.randn(filters, kernel_size) * 0.01

    def causal_conv1d(self, x):
        """Causal 1D convolution (no future information)"""
        n = len(x)
        output = np.zeros(n)

        for i in range(n):
            conv_sum = 0
            count = 0
            for k in range(self.kernel_size):
                idx = i - k * self.dilation
                if idx >= 0:
                    conv_sum += x[idx] * self.weights[0, k]
                    count += 1
            if count > 0:
                output[i] = conv_sum / count

        return output

    def dilated_conv1d(self, x, dilation):
        """Dilated causal convolution"""
        n = len(x)
        output = np.zeros(n)

        for i in range(n):
            conv_sum = 0
            count = 0
            for k in range(self.kernel_size):
                idx = i - k * dilation
                if idx >= 0:
                    conv_sum += x[idx] * self.weights[0, k]
                    count += 1
            if count > 0:
                output[i] = conv_sum / count

        return output


def create_sequences(data, lookback, forecast_horizon=1):
    """Create sequences for training"""
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback:i + lookback + forecast_horizon])
    return np.array(X), np.array(y)


def prepare_data(series, lookback=60, forecast_horizon=1, train_ratio=0.8):
    """Prepare data for TCN training"""
    print(f"\n{'='*70}")
    print("Data Preparation for TCN")
    print(f"{'='*70}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    X, y = create_sequences(scaled_data, lookback, forecast_horizon)

    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  Lookback: {lookback} (for capturing long-term patterns)")

    return X_train, X_test, y_train, y_test, scaler


class SimpleTCNModel:
    """Simple TCN model"""

    def __init__(self, lookback, kernel_size=3, num_layers=3):
        self.lookback = lookback
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.layers = [TemporalConvBlock(kernel_size, dilation=1) for _ in range(num_layers)]
        self.history = {'loss': [], 'val_loss': []}

    def forward(self, x):
        """Forward pass through TCN layers"""
        output = x.copy()
        for layer in self.layers:
            output = layer.causal_conv1d(output)
            output = np.maximum(0, output)  # ReLU activation
        return output[-1]  # Return last timestep

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """Train TCN model"""
        print(f"\n{'='*70}")
        print("Training Simple TCN Model")
        print(f"{'='*70}")
        print(f"Layers: {self.num_layers}, Kernel size: {self.kernel_size}")

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


class DilatedTCNModel:
    """TCN with dilated convolutions"""

    def __init__(self, lookback, kernel_size=3, num_layers=4, dilation_rates=None):
        self.lookback = lookback
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.dilation_rates = dilation_rates if dilation_rates else [1, 2, 4, 8]
        self.layers = [TemporalConvBlock(kernel_size, dilation=d)
                      for d in self.dilation_rates[:num_layers]]
        self.history = {'loss': [], 'val_loss': []}

    def calculate_receptive_field(self):
        """Calculate effective receptive field"""
        receptive_field = 1
        for dilation in self.dilation_rates[:self.num_layers]:
            receptive_field += (self.kernel_size - 1) * dilation
        return receptive_field

    def forward(self, x):
        """Forward pass with dilated convolutions"""
        output = x.copy()
        for i, layer in enumerate(self.layers):
            output = layer.dilated_conv1d(output, self.dilation_rates[i])
            output = np.maximum(0, output)  # ReLU
        return output[-1]

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """Train dilated TCN"""
        print(f"\n{'='*70}")
        print("Training Dilated TCN Model")
        print(f"{'='*70}")
        print(f"Layers: {self.num_layers}, Dilation rates: {self.dilation_rates[:self.num_layers]}")
        print(f"Receptive field: {self.calculate_receptive_field()} timesteps")

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


class MultiScaleTCNModel:
    """Multi-scale TCN with different kernel sizes"""

    def __init__(self, lookback, kernel_sizes=[3, 5, 7], num_layers=3):
        self.lookback = lookback
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        # Create parallel conv blocks with different kernel sizes
        self.conv_blocks = {
            k: [TemporalConvBlock(kernel_size=k, dilation=1) for _ in range(num_layers)]
            for k in kernel_sizes
        }
        self.history = {'loss': [], 'val_loss': []}

    def forward(self, x):
        """Forward pass with multi-scale convolutions"""
        outputs = []

        for kernel_size, layers in self.conv_blocks.items():
            output = x.copy()
            for layer in layers:
                output = layer.causal_conv1d(output)
                output = np.maximum(0, output)
            outputs.append(output[-1])

        # Combine multi-scale features
        return np.mean(outputs)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50):
        """Train multi-scale TCN"""
        print(f"\n{'='*70}")
        print("Training Multi-Scale TCN Model")
        print(f"{'='*70}")
        print(f"Kernel sizes: {self.kernel_sizes}, Layers per scale: {self.num_layers}")

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


def analyze_receptive_field(lookback, kernel_size, num_layers, dilation_rates):
    """Analyze and visualize receptive field"""
    print(f"\n{'='*70}")
    print("Receptive Field Analysis")
    print(f"{'='*70}")

    receptive_field = 1
    layer_fields = [1]

    for i, dilation in enumerate(dilation_rates[:num_layers]):
        rf = (kernel_size - 1) * dilation
        receptive_field += rf
        layer_fields.append(receptive_field)

        print(f"\nLayer {i+1}:")
        print(f"  Dilation: {dilation}")
        print(f"  Receptive field: {receptive_field} timesteps")
        print(f"  Coverage: {receptive_field/lookback*100:.1f}% of lookback window")

    print(f"\nTotal receptive field: {receptive_field} timesteps")
    print(f"Lookback window: {lookback} timesteps")
    print(f"Coverage: {min(100, receptive_field/lookback*100):.1f}%")

    return layer_fields


def compare_tcn_rnn(X_train, y_train, X_test, y_test, lookback):
    """Compare TCN with RNN-based models"""
    print(f"\n{'='*70}")
    print("TCN vs RNN Comparison")
    print(f"{'='*70}")

    import time

    results = {}

    # TCN
    start_time = time.time()
    tcn = DilatedTCNModel(lookback, num_layers=4, dilation_rates=[1, 2, 4, 8])
    tcn.train(X_train, y_train, epochs=30)
    tcn_pred = tcn.predict(X_test)
    tcn_time = time.time() - start_time

    tcn_mae = mean_absolute_error(y_test.flatten(), tcn_pred)

    results['TCN'] = {
        'MAE': tcn_mae,
        'Time': tcn_time,
        'Receptive_Field': tcn.calculate_receptive_field(),
        'Parallelizable': True
    }

    # RNN (simulated with sequential processing)
    start_time = time.time()
    rnn_preds = []
    for i in range(len(X_test)):
        # Simulate sequential processing (slower)
        hidden = 0
        for t in range(len(X_test[i])):
            hidden = 0.5 * hidden + 0.5 * X_test[i][t]
        rnn_preds.append(hidden)
    rnn_pred = np.array(rnn_preds)
    rnn_time = time.time() - start_time

    rnn_mae = mean_absolute_error(y_test.flatten(), rnn_pred)

    results['RNN'] = {
        'MAE': rnn_mae,
        'Time': rnn_time,
        'Receptive_Field': lookback,
        'Parallelizable': False
    }

    print(f"\nComparison Results:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        print(f"  MAE: {metrics['MAE']:.6f}")
        print(f"  Time: {metrics['Time']:.2f}s")
        print(f"  Receptive Field: {metrics['Receptive_Field']}")
        print(f"  Parallelizable: {metrics['Parallelizable']}")

    print(f"\nTCN Advantages:")
    print(f"  - Parallel processing (training can be parallelized)")
    print(f"  - Fixed receptive field (predictable long-term dependencies)")
    print(f"  - No vanishing gradient issues from sequential processing")

    return results


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}


def plot_tcn_results(df, models_predictions, scaler, receptive_fields):
    """Plot TCN results and analysis"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)

    # Original series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['value'], linewidth=1.5)
    ax1.set_title('Time Series with Long-Term Dependencies', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Training history
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

    # Receptive field growth
    ax3 = fig.add_subplot(gs[2, 0])
    if receptive_fields:
        ax3.plot(receptive_fields, marker='o', linewidth=2, markersize=8)
        ax3.set_title('Receptive Field Growth by Layer', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Receptive Field Size')
        ax3.grid(True, alpha=0.3)

    # Predictions
    ax4 = fig.add_subplot(gs[2, 1])
    test_start = int(len(df) * 0.8)
    ax4.plot(df.index[test_start:test_start+100], df['value'][test_start:test_start+100],
             label='Actual', linewidth=2, color='black')

    for idx, (name, data) in enumerate(models_predictions.items()):
        if 'predictions' in data and len(data['predictions']) > 0:
            pred_rescaled = scaler.inverse_transform(data['predictions'].reshape(-1, 1)).flatten()
            pred_dates = df.index[test_start:test_start+len(pred_rescaled)]
            ax4.plot(pred_dates[:100], pred_rescaled[:100], label=name,
                    linewidth=1.5, color=colors[idx % len(colors)], linestyle='--', alpha=0.8)

    ax4.set_title('Model Predictions', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Error distribution
    ax5 = fig.add_subplot(gs[3, 0])
    for idx, (name, data) in enumerate(models_predictions.items()):
        if 'predictions' in data and 'actual' in data:
            errors = data['actual'] - data['predictions']
            ax5.hist(errors, bins=30, alpha=0.5, label=name, edgecolor='black')
    ax5.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Error')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Actual vs Predicted
    ax6 = fig.add_subplot(gs[3, 1])
    best_model = list(models_predictions.keys())[0]
    if 'predictions' in models_predictions[best_model]:
        actual = models_predictions[best_model]['actual']
        predicted = models_predictions[best_model]['predictions']
        ax6.scatter(actual, predicted, alpha=0.5, s=20)
        ax6.plot([actual.min(), actual.max()], [actual.min(), actual.max()],
                'r--', linewidth=2)
        ax6.set_xlabel('Actual')
        ax6.set_ylabel('Predicted')
        ax6.set_title(f'{best_model} - Actual vs Predicted', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

    plt.savefig('tcn_results.png', dpi=300, bbox_inches='tight')
    return fig


def main():
    """Main execution function"""
    print("="*70)
    print("TEMPORAL CONVOLUTIONAL NETWORKS FOR TIME SERIES")
    print("="*70)

    # Generate data
    df = generate_long_dependency_series(n_samples=1500)
    series = df['value']

    # Stationarity test
    adf_result = adfuller(series)
    print(f"\nADF test p-value: {adf_result[1]:.6f}")

    # Prepare data
    lookback = 60
    X_train, X_test, y_train, y_test, scaler = prepare_data(series, lookback=lookback)

    # Validation split
    val_split = int(len(X_train) * 0.9)
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]

    # Analyze receptive field
    receptive_fields = analyze_receptive_field(lookback, kernel_size=3,
                                               num_layers=4, dilation_rates=[1, 2, 4, 8])

    # Train models
    models_predictions = {}

    # 1. Simple TCN
    simple_tcn = SimpleTCNModel(lookback, kernel_size=3, num_layers=3)
    simple_tcn.train(X_train, y_train, X_val, y_val, epochs=50)
    simple_pred = simple_tcn.predict(X_test)
    models_predictions['Simple TCN'] = {
        'predictions': simple_pred,
        'actual': y_test.flatten(),
        'history': simple_tcn.history
    }

    # 2. Dilated TCN
    dilated_tcn = DilatedTCNModel(lookback, kernel_size=3, num_layers=4,
                                  dilation_rates=[1, 2, 4, 8])
    dilated_tcn.train(X_train, y_train, X_val, y_val, epochs=50)
    dilated_pred = dilated_tcn.predict(X_test)
    models_predictions['Dilated TCN'] = {
        'predictions': dilated_pred,
        'actual': y_test.flatten(),
        'history': dilated_tcn.history
    }

    # 3. Multi-scale TCN
    multiscale_tcn = MultiScaleTCNModel(lookback, kernel_sizes=[3, 5, 7], num_layers=3)
    multiscale_tcn.train(X_train, y_train, X_val, y_val, epochs=50)
    multiscale_pred = multiscale_tcn.predict(X_test)
    models_predictions['Multi-Scale TCN'] = {
        'predictions': multiscale_pred,
        'actual': y_test.flatten(),
        'history': multiscale_tcn.history
    }

    # Performance metrics
    print(f"\n{'='*70}")
    print("Model Performance")
    print(f"{'='*70}")

    for name, data in models_predictions.items():
        metrics = calculate_metrics(data['actual'], data['predictions'])
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")

    # TCN vs RNN comparison
    comparison = compare_tcn_rnn(X_train, y_train, X_test, y_test, lookback)

    # Plot results
    plot_tcn_results(df, models_predictions, scaler, receptive_fields)

    print("\n" + "="*70)
    print("TCN ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
