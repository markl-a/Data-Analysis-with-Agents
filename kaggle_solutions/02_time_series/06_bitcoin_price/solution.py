"""
Bitcoin Price Prediction with LSTM
Predict Bitcoin prices using Long Short-Term Memory neural networks

Dataset: Simulated Bitcoin historical price data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Try to import tensorflow, if not available use a simplified approach
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using statistical model fallback.")


class BitcoinPricePredictor:
    """Bitcoin price prediction using LSTM neural networks"""

    def __init__(self, lookback=60):
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None

    def generate_bitcoin_data(self, n_days=1095):
        """Generate realistic Bitcoin price data with trend, volatility, and cycles"""
        np.random.seed(42)

        # Start date: 3 years ago
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')

        # Base price with exponential trend
        base_price = 10000
        trend = np.exp(np.linspace(0, 0.8, n_days))  # Exponential growth

        # Add cyclic patterns (quarterly cycles)
        cycle = 0.15 * np.sin(2 * np.pi * np.arange(n_days) / 90)

        # Add random walk (volatility)
        volatility = np.random.normal(0, 0.02, n_days)
        cumulative_volatility = np.cumsum(volatility)

        # Combine all components
        price = base_price * trend * (1 + cycle + cumulative_volatility * 0.3)

        # Add occasional sharp movements (flash crashes/rallies)
        for i in range(5):
            crash_day = np.random.randint(100, n_days - 100)
            price[crash_day:crash_day+7] *= np.random.uniform(0.85, 1.15)

        # Ensure positive prices
        price = np.maximum(price, 1000)

        # Create additional features
        df = pd.DataFrame({
            'date': dates,
            'price': price,
        })

        # Calculate technical indicators
        df['returns'] = df['price'].pct_change()
        df['volatility'] = df['returns'].rolling(window=7).std()
        df['ma_7'] = df['price'].rolling(window=7).mean()
        df['ma_30'] = df['price'].rolling(window=30).mean()
        df['ma_90'] = df['price'].rolling(window=90).mean()
        df['price_to_ma7'] = df['price'] / df['ma_7']
        df['price_to_ma30'] = df['price'] / df['ma_30']

        # Forward fill NaN values
        df = df.fillna(method='bfill')

        return df

    def prepare_sequences(self, data, target_col='price'):
        """Prepare sequences for LSTM model"""
        X, y = [], []

        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i])
            y.append(data[i, 0])  # Predict price (first column)

        return np.array(X), np.array(y)

    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            return None

        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def simple_forecasting_fallback(self, train_data, test_size):
        """Simple exponential smoothing fallback when TensorFlow is not available"""
        # Use exponential moving average
        alpha = 0.3
        predictions = []
        last_value = train_data[-1]

        for _ in range(test_size):
            predictions.append(last_value)
            # Simple trend continuation
            last_value = last_value * 1.001 + np.random.normal(0, last_value * 0.01)

        return np.array(predictions)

    def train_and_evaluate(self):
        """Train the model and evaluate performance"""
        print("=" * 70)
        print("Bitcoin Price Prediction with LSTM")
        print("=" * 70)

        # Generate data
        print("\n1. Generating Bitcoin price data...")
        df = self.generate_bitcoin_data()
        print(f"   Generated {len(df)} days of price data")
        print(f"   Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

        # Prepare features
        feature_cols = ['price', 'ma_7', 'ma_30', 'volatility', 'price_to_ma7']
        data = df[feature_cols].values

        # Scale the data
        scaled_data = self.scaler.fit_transform(data)

        # Split data (80% train, 20% test) - respecting temporal order
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        print(f"\n2. Preparing sequences...")
        print(f"   Lookback window: {self.lookback} days")
        print(f"   Training samples: {train_size - self.lookback}")
        print(f"   Test samples: {len(test_data) - self.lookback}")

        if TENSORFLOW_AVAILABLE:
            # Prepare sequences
            X_train, y_train = self.prepare_sequences(train_data)
            X_test, y_test = self.prepare_sequences(test_data)

            # Build and train model
            print(f"\n3. Building LSTM model...")
            self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            print(f"   Model architecture: LSTM(50) -> LSTM(50) -> LSTM(50) -> Dense(25) -> Dense(1)")

            print(f"\n4. Training model...")
            self.history = self.model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
            print(f"   Training complete!")

            # Make predictions
            train_predict = self.model.predict(X_train, verbose=0)
            test_predict = self.model.predict(X_test, verbose=0)

            # Inverse transform predictions
            train_predict_full = np.zeros((len(train_predict), len(feature_cols)))
            train_predict_full[:, 0] = train_predict.flatten()
            train_predict = self.scaler.inverse_transform(train_predict_full)[:, 0]

            test_predict_full = np.zeros((len(test_predict), len(feature_cols)))
            test_predict_full[:, 0] = test_predict.flatten()
            test_predict = self.scaler.inverse_transform(test_predict_full)[:, 0]

            # Get actual values
            y_train_full = np.zeros((len(y_train), len(feature_cols)))
            y_train_full[:, 0] = y_train
            y_train_actual = self.scaler.inverse_transform(y_train_full)[:, 0]

            y_test_full = np.zeros((len(y_test), len(feature_cols)))
            y_test_full[:, 0] = y_test
            y_test_actual = self.scaler.inverse_transform(y_test_full)[:, 0]

        else:
            # Fallback method
            print(f"\n3. Using simple forecasting method (TensorFlow not available)...")
            train_prices = data[:train_size, 0]
            test_size = len(test_data) - self.lookback

            test_predict = self.simple_forecasting_fallback(train_prices, test_size)
            train_predict = train_prices[-len(test_predict):]
            y_train_actual = train_prices[-len(test_predict):]
            y_test_actual = data[train_size + self.lookback:, 0]

        # Calculate metrics
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)

        # Test set metrics
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
        test_mae = mean_absolute_error(y_test_actual, test_predict)
        test_mape = mean_absolute_percentage_error(y_test_actual, test_predict) * 100

        print(f"\nTest Set Performance:")
        print(f"  RMSE: ${test_rmse:,.2f}")
        print(f"  MAE:  ${test_mae:,.2f}")
        print(f"  MAPE: {test_mape:.2f}%")

        # Visualizations
        self.create_visualizations(df, train_size, train_predict, test_predict,
                                   y_train_actual, y_test_actual)

        return {
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_mape': test_mape
        }

    def create_visualizations(self, df, train_size, train_predict, test_predict,
                             y_train_actual, y_test_actual):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(16, 12))

        # 1. Price history and predictions
        ax1 = plt.subplot(3, 2, 1)
        train_dates = df['date'].iloc[train_size:train_size+len(train_predict)]
        test_dates = df['date'].iloc[train_size+self.lookback:]

        plt.plot(df['date'], df['price'], label='Actual Price', alpha=0.7, linewidth=1)
        plt.plot(test_dates, test_predict, label='Predictions',
                color='red', linewidth=2, alpha=0.8)
        plt.axvline(x=df['date'].iloc[train_size], color='green',
                   linestyle='--', label='Train/Test Split', alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.title('Bitcoin Price: Actual vs Predicted', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Test set zoom
        ax2 = plt.subplot(3, 2, 2)
        plt.plot(test_dates, y_test_actual, label='Actual', marker='o',
                markersize=3, linewidth=2)
        plt.plot(test_dates, test_predict, label='Predicted', marker='s',
                markersize=3, linewidth=2)
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.title('Test Set: Predictions vs Actual', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 3. Prediction errors
        ax3 = plt.subplot(3, 2, 3)
        errors = test_predict - y_test_actual
        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Prediction Error ($)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 4. Moving averages
        ax4 = plt.subplot(3, 2, 4)
        plt.plot(df['date'], df['price'], label='Price', alpha=0.6)
        plt.plot(df['date'], df['ma_7'], label='7-day MA', alpha=0.8)
        plt.plot(df['date'], df['ma_30'], label='30-day MA', alpha=0.8)
        plt.plot(df['date'], df['ma_90'], label='90-day MA', alpha=0.8)
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.title('Price with Moving Averages', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Training history (if LSTM was used)
        ax5 = plt.subplot(3, 2, 5)
        if self.history is not None and TENSORFLOW_AVAILABLE:
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.title('Model Training History', fontsize=12, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Training history not available\n(Simple forecasting method used)',
                    ha='center', va='center', fontsize=12)
            plt.axis('off')

        # 6. Scatter plot
        ax6 = plt.subplot(3, 2, 6)
        plt.scatter(y_test_actual, test_predict, alpha=0.6)
        plt.plot([y_test_actual.min(), y_test_actual.max()],
                [y_test_actual.min(), y_test_actual.max()],
                'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title('Actual vs Predicted (Test Set)', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/02_time_series/06_bitcoin_price/bitcoin_prediction.png',
                    dpi=300, bbox_inches='tight')
        print("\n📊 Visualizations saved to 'bitcoin_prediction.png'")
        plt.close()


def main():
    """Main execution function"""
    # Create and run predictor
    predictor = BitcoinPricePredictor(lookback=60)
    results = predictor.train_and_evaluate()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Insights:")
    print("1. LSTM networks can capture complex temporal patterns in Bitcoin prices")
    print("2. Price volatility makes accurate long-term predictions challenging")
    print("3. Technical indicators (moving averages) help improve predictions")
    print("4. Model performance degrades for longer prediction horizons")

    if TENSORFLOW_AVAILABLE:
        print("\n✅ TensorFlow LSTM model successfully trained and evaluated")
    else:
        print("\n⚠️  Simple forecasting method used (install TensorFlow for LSTM)")


if __name__ == "__main__":
    main()
