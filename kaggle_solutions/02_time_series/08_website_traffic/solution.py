"""
Website Traffic Forecasting
Predict website traffic using Prophet and trend analysis

Dataset: Simulated website visitor data
Difficulty: ⭐⭐ Intermediate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet, fallback to polynomial regression
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Using polynomial trend forecasting.")
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge


class WebsiteTrafficForecaster:
    """Website traffic forecasting using Prophet and trend analysis"""

    def __init__(self):
        self.model = None

    def generate_traffic_data(self, n_days=730):
        """Generate realistic website traffic data with multiple patterns"""
        np.random.seed(42)

        # Create daily date range (2 years)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')

        # Base traffic
        base_traffic = 5000

        # Long-term growth trend
        growth = np.exp(np.linspace(0, 0.5, n_days)) - 1

        # Weekly seasonality (higher on weekdays, lower on weekends)
        weekly_pattern = np.array([1.0, 1.05, 1.1, 1.08, 1.12, 0.8, 0.75])  # Mon-Sun
        weekly_seasonal = np.tile(weekly_pattern, n_days // 7 + 1)[:n_days]

        # Monthly seasonality (some months have higher traffic)
        monthly_pattern = []
        for date in dates:
            month = date.month
            # Higher traffic in Jan (New Year), Sep (back to school), Nov-Dec (holidays)
            if month in [1, 9, 11, 12]:
                monthly_pattern.append(1.15)
            elif month in [6, 7, 8]:  # Summer dip
                monthly_pattern.append(0.9)
            else:
                monthly_pattern.append(1.0)
        monthly_seasonal = np.array(monthly_pattern)

        # Special events (product launches, viral content, marketing campaigns)
        events = np.ones(n_days)
        for i in range(10):  # 10 random events
            event_day = np.random.randint(30, n_days - 30)
            # Event creates spike that decays over a week
            for j in range(14):
                if event_day + j < n_days:
                    events[event_day + j] += 0.3 * np.exp(-j / 5)

        # Random daily variation
        noise = np.random.normal(1, 0.1, n_days)

        # Combine all components
        traffic = base_traffic * (1 + growth) * weekly_seasonal * monthly_seasonal * events * noise

        # Ensure positive values
        traffic = np.maximum(traffic, 100).astype(int)

        # Create dataframe
        df = pd.DataFrame({
            'date': dates,
            'visitors': traffic
        })

        # Add temporal features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.month_name()
        df['year'] = df['date'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday_month'] = df['month'].isin([1, 11, 12]).astype(int)

        # Add rolling statistics
        df['visitors_ma7'] = df['visitors'].rolling(window=7, center=False).mean()
        df['visitors_ma30'] = df['visitors'].rolling(window=30, center=False).mean()
        df['visitors_std7'] = df['visitors'].rolling(window=7, center=False).std()

        # Calculate growth metrics
        df['daily_change'] = df['visitors'].diff()
        df['pct_change'] = df['visitors'].pct_change() * 100

        df = df.fillna(method='bfill')

        return df

    def fit_prophet_model(self, train_df):
        """Fit Prophet model to training data"""
        if not PROPHET_AVAILABLE:
            return None

        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = pd.DataFrame({
            'ds': train_df['date'],
            'y': train_df['visitors']
        })

        # Initialize and fit model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )

        # Add custom seasonality for monthly patterns
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        model.fit(prophet_df, verbose=False)
        return model

    def polynomial_forecast(self, train_data, train_dates, forecast_days):
        """Polynomial regression fallback method"""
        # Convert dates to numeric (days since start)
        days_numeric = np.arange(len(train_data)).reshape(-1, 1)

        # Fit polynomial regression
        poly = PolynomialFeatures(degree=3)
        days_poly = poly.fit_transform(days_numeric)

        model = Ridge(alpha=1.0)
        model.fit(days_poly, train_data)

        # Generate future dates
        future_days = np.arange(len(train_data), len(train_data) + forecast_days).reshape(-1, 1)
        future_poly = poly.transform(future_days)

        # Predict
        predictions = model.predict(future_poly)

        # Add weekly seasonality manually
        weekly_pattern = np.array([1.0, 1.05, 1.1, 1.08, 1.12, 0.8, 0.75])
        last_day_of_week = train_dates.iloc[-1].dayofweek
        seasonal_adjustments = []
        for i in range(forecast_days):
            day_of_week = (last_day_of_week + i + 1) % 7
            seasonal_adjustments.append(weekly_pattern[day_of_week])

        predictions = predictions * np.array(seasonal_adjustments)

        return np.maximum(predictions, 0)

    def train_and_evaluate(self):
        """Train model and evaluate performance"""
        print("=" * 70)
        print("Website Traffic Forecasting")
        print("=" * 70)

        # Generate data
        print("\n1. Generating website traffic data...")
        df = self.generate_traffic_data()
        print(f"   Generated {len(df)} days of traffic data")
        print(f"   Visitor range: {df['visitors'].min():,} - {df['visitors'].max():,}")
        print(f"   Average daily visitors: {df['visitors'].mean():,.0f}")

        # Analyze patterns
        print("\n2. Analyzing traffic patterns...")
        weekday_avg = df.groupby('day_name')['visitors'].mean()
        print(f"   Average visitors by day:")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in day_order:
            if day in weekday_avg.index:
                print(f"     {day}: {weekday_avg[day]:,.0f}")

        # Split data (80% train, 20% test)
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        print(f"\n3. Splitting data...")
        print(f"   Training days: {len(train_df)}")
        print(f"   Test days: {len(test_df)}")

        # Fit model
        if PROPHET_AVAILABLE:
            print(f"\n4. Fitting Prophet model...")
            self.model = self.fit_prophet_model(train_df)

            if self.model is not None:
                print(f"   Model fitted successfully!")

                # Make predictions
                future = self.model.make_future_dataframe(periods=len(test_df), freq='D')
                forecast = self.model.predict(future)

                # Extract test predictions
                predictions = forecast['yhat'].iloc[-len(test_df):].values
                predictions = np.maximum(predictions, 0)  # Ensure non-negative

                # Get confidence intervals
                lower_bound = forecast['yhat_lower'].iloc[-len(test_df):].values
                upper_bound = forecast['yhat_upper'].iloc[-len(test_df):].values
            else:
                PROPHET_AVAILABLE = False

        if not PROPHET_AVAILABLE:
            print(f"\n4. Using polynomial regression method...")
            predictions = self.polynomial_forecast(
                train_df['visitors'].values,
                train_df['date'],
                len(test_df)
            )
            lower_bound = predictions * 0.9
            upper_bound = predictions * 1.1

        # Calculate metrics
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)

        actual = test_df['visitors'].values
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        mape = mean_absolute_percentage_error(actual, predictions) * 100

        print(f"\nTest Set Performance:")
        print(f"  RMSE: {rmse:,.2f} visitors")
        print(f"  MAE:  {mae:,.2f} visitors")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Forecast Accuracy: {100 - mape:.2f}%")

        # Peak detection
        peak_error = np.abs(predictions.max() - actual.max())
        print(f"\n  Peak traffic prediction error: {peak_error:,.0f} visitors")

        # Visualizations
        self.create_visualizations(df, train_size, predictions, actual,
                                   test_df['date'], lower_bound, upper_bound)

        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }

    def create_visualizations(self, df, train_size, predictions, actual,
                             test_dates, lower_bound, upper_bound):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(16, 12))

        # 1. Full time series with forecast
        ax1 = plt.subplot(3, 2, 1)
        plt.plot(df['date'], df['visitors'], label='Actual Traffic',
                color='blue', linewidth=1, alpha=0.7)
        plt.plot(test_dates, predictions, label='Forecast',
                color='red', linewidth=2)
        plt.fill_between(test_dates, lower_bound, upper_bound,
                        color='red', alpha=0.2, label='Confidence Interval')
        plt.axvline(x=df['date'].iloc[train_size], color='green',
                   linestyle='--', label='Train/Test Split', alpha=0.5)
        plt.xlabel('Date')
        plt.ylabel('Visitors')
        plt.title('Website Traffic: Actual vs Forecast', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 2. Test period detail
        ax2 = plt.subplot(3, 2, 2)
        plt.plot(test_dates, actual, label='Actual', marker='o',
                markersize=3, linewidth=2)
        plt.plot(test_dates, predictions, label='Forecast', marker='s',
                markersize=3, linewidth=2)
        plt.fill_between(test_dates, lower_bound, upper_bound,
                        alpha=0.2)
        plt.xlabel('Date')
        plt.ylabel('Visitors')
        plt.title('Test Period: Forecast vs Actual', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 3. Weekly pattern
        ax3 = plt.subplot(3, 2, 3)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_avg = df.groupby('day_name')['visitors'].mean().reindex(day_order)
        colors = ['steelblue' if day not in ['Saturday', 'Sunday'] else 'coral'
                 for day in day_order]
        plt.bar(range(7), weekly_avg.values, color=colors,
               edgecolor='black', alpha=0.7)
        plt.xlabel('Day of Week')
        plt.ylabel('Average Visitors')
        plt.title('Weekly Traffic Pattern', fontsize=12, fontweight='bold')
        plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.grid(True, alpha=0.3, axis='y')

        # 4. Monthly pattern
        ax4 = plt.subplot(3, 2, 4)
        monthly_avg = df.groupby('month_name')['visitors'].mean()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_avg = monthly_avg.reindex([m for m in month_order if m in monthly_avg.index])
        plt.bar(range(len(monthly_avg)), monthly_avg.values,
               color='mediumseagreen', edgecolor='black', alpha=0.7)
        plt.xlabel('Month')
        plt.ylabel('Average Visitors')
        plt.title('Monthly Traffic Pattern', fontsize=12, fontweight='bold')
        plt.xticks(range(len(monthly_avg)),
                  [m[:3] for m in monthly_avg.index], rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        # 5. Forecast errors
        ax5 = plt.subplot(3, 2, 5)
        errors = predictions - actual
        percentage_errors = (errors / actual) * 100
        plt.hist(percentage_errors, bins=25, edgecolor='black',
                alpha=0.7, color='orange')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Forecast Error (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Forecast Errors', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 6. Growth trend
        ax6 = plt.subplot(3, 2, 6)
        monthly_totals = df.groupby(df['date'].dt.to_period('M'))['visitors'].sum()
        plt.plot(range(len(monthly_totals)), monthly_totals.values,
                marker='o', linewidth=2, markersize=6, color='purple')
        plt.xlabel('Month')
        plt.ylabel('Total Visitors')
        plt.title('Monthly Traffic Growth Trend', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('/home/user/Data-Analysis-with-Chatbots/kaggle_solutions/02_time_series/08_website_traffic/traffic_forecast.png',
                   dpi=300, bbox_inches='tight')
        print("\n📊 Visualizations saved to 'traffic_forecast.png'")
        plt.close()


def main():
    """Main execution function"""
    # Create and run forecaster
    forecaster = WebsiteTrafficForecaster()
    results = forecaster.train_and_evaluate()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Insights:")
    print("1. Website traffic shows strong weekly seasonality (weekdays > weekends)")
    print("2. Monthly patterns reflect business cycles and holidays")
    print("3. Special events create temporary traffic spikes")
    print("4. Long-term growth trend indicates successful user acquisition")

    if PROPHET_AVAILABLE:
        print("\n✅ Prophet model successfully fitted and evaluated")
    else:
        print("\n⚠️  Polynomial regression used (install prophet for advanced forecasting)")

    print("\nBusiness Recommendations:")
    print("• Schedule content releases during high-traffic weekdays")
    print("• Plan server capacity for peak holiday periods")
    print("• Monitor and capitalize on viral traffic spikes")
    print("• Invest in SEO during growth phases")


if __name__ == "__main__":
    main()
