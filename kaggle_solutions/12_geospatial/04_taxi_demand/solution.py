"""
Taxi Demand Prediction - Geospatial Analysis
Predict taxi demand patterns across city zones

Dataset: Synthetic taxi trip data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TaxiDemandPredictor:
    """Taxi demand prediction using geospatial and temporal features"""

    def __init__(self, city_name="Metro City"):
        self.city_name = city_name
        self.trip_data = None
        self.zones = None
        self.demand_model = None

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance in kilometers"""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def create_sample_data(self):
        """Generate synthetic taxi trip data"""
        np.random.seed(42)

        # City bounds
        center_lat, center_lon = 40.7580, -73.9855  # Times Square-like
        lat_spread, lon_spread = 0.08, 0.12

        # Generate high-demand zones
        n_zones = 8
        zone_centers_lat = center_lat + np.random.uniform(-lat_spread, lat_spread, n_zones)
        zone_centers_lon = center_lon + np.random.uniform(-lon_spread, lon_spread, n_zones)
        zone_demand_level = np.random.uniform(0.5, 2.0, n_zones)

        # Generate trips over one week
        n_trips = 10000
        trips = []
        start_date = datetime(2024, 1, 1)

        for i in range(n_trips):
            # Time generation (more trips during rush hours and weekends)
            day_offset = np.random.randint(0, 7)
            hour = self._sample_hour()
            minute = np.random.randint(0, 60)
            pickup_time = start_date + timedelta(days=day_offset, hours=hour, minutes=minute)

            # Pickup location (biased toward zones)
            zone_idx = np.random.choice(n_zones, p=zone_demand_level/zone_demand_level.sum())
            pickup_lat = zone_centers_lat[zone_idx] + np.random.normal(0, 0.008)
            pickup_lon = zone_centers_lon[zone_idx] + np.random.normal(0, 0.012)

            # Drop-off location
            if np.random.random() < 0.6:
                # Same zone or nearby
                dropoff_zone = zone_idx if np.random.random() < 0.4 else np.random.randint(0, n_zones)
                dropoff_lat = zone_centers_lat[dropoff_zone] + np.random.normal(0, 0.008)
                dropoff_lon = zone_centers_lon[dropoff_zone] + np.random.normal(0, 0.012)
            else:
                # Random destination
                dropoff_lat = center_lat + np.random.uniform(-lat_spread, lat_spread)
                dropoff_lon = center_lon + np.random.uniform(-lon_spread, lon_spread)

            # Calculate trip metrics
            distance = self.haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
            duration = max(5, distance * 8 + np.random.normal(0, 5))  # Minutes
            fare = max(5.0, 3.0 + distance * 2.5 + duration * 0.5)

            # Weather effect
            weather = np.random.choice(['Clear', 'Rain', 'Snow'], p=[0.7, 0.2, 0.1])
            if weather == 'Rain':
                fare *= 1.2
            elif weather == 'Snow':
                fare *= 1.4

            trips.append({
                'trip_id': f'TRIP{i+1:05d}',
                'pickup_datetime': pickup_time,
                'pickup_latitude': pickup_lat,
                'pickup_longitude': pickup_lon,
                'dropoff_latitude': dropoff_lat,
                'dropoff_longitude': dropoff_lon,
                'distance_km': distance,
                'duration_min': duration,
                'fare': fare,
                'hour': hour,
                'day_of_week': pickup_time.strftime('%A'),
                'is_weekend': 1 if pickup_time.weekday() >= 5 else 0,
                'is_rush_hour': 1 if hour in [7, 8, 9, 17, 18, 19] else 0,
                'weather': weather,
                'pickup_zone': zone_idx
            })

        self.trip_data = pd.DataFrame(trips)

        print(f"✓ Generated {len(self.trip_data)} taxi trips")
        print(f"✓ Date range: {self.trip_data['pickup_datetime'].min()} to {self.trip_data['pickup_datetime'].max()}")
        print(f"✓ Total fare collected: ${self.trip_data['fare'].sum():,.2f}")
        print(f"✓ Average trip distance: {self.trip_data['distance_km'].mean():.2f} km")

    def _sample_hour(self):
        """Sample hour with realistic distribution"""
        # Higher probability during rush hours and evenings
        hour_probs = np.array([
            0.01, 0.01, 0.01, 0.01, 0.02, 0.03,  # 0-5 AM
            0.04, 0.08, 0.09, 0.07, 0.05, 0.05,  # 6-11 AM
            0.06, 0.05, 0.05, 0.05, 0.06, 0.08,  # 12-5 PM
            0.09, 0.08, 0.04, 0.03, 0.02, 0.02   # 6-11 PM
        ])
        return np.random.choice(24, p=hour_probs/hour_probs.sum())

    def create_demand_zones(self, n_zones=10):
        """Create demand zones using K-means clustering"""
        print("\n" + "="*60)
        print("CREATING DEMAND ZONES")
        print("="*60)

        # Cluster pickup locations
        coords = self.trip_data[['pickup_latitude', 'pickup_longitude']].values
        kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
        self.trip_data['demand_zone'] = kmeans.fit_predict(coords)

        # Calculate zone statistics
        zone_stats = []
        for zone_id in range(n_zones):
            zone_trips = self.trip_data[self.trip_data['demand_zone'] == zone_id]

            zone_stats.append({
                'zone_id': zone_id,
                'n_trips': len(zone_trips),
                'center_lat': zone_trips['pickup_latitude'].mean(),
                'center_lon': zone_trips['pickup_longitude'].mean(),
                'avg_fare': zone_trips['fare'].mean(),
                'avg_distance': zone_trips['distance_km'].mean(),
                'rush_hour_pct': zone_trips['is_rush_hour'].mean() * 100,
                'weekend_pct': zone_trips['is_weekend'].mean() * 100
            })

        self.zones = pd.DataFrame(zone_stats).sort_values('n_trips', ascending=False)

        print(f"\n✓ Created {n_zones} demand zones")
        print("\nTop 5 High-Demand Zones:")
        for _, zone in self.zones.head(5).iterrows():
            print(f"\nZone {zone['zone_id']}:")
            print(f"  Location: ({zone['center_lat']:.4f}, {zone['center_lon']:.4f})")
            print(f"  Total Trips: {zone['n_trips']}")
            print(f"  Avg Fare: ${zone['avg_fare']:.2f}")
            print(f"  Rush Hour %: {zone['rush_hour_pct']:.1f}%")

        return self.zones

    def analyze_demand_patterns(self):
        """Analyze temporal demand patterns"""
        print("\n" + "="*60)
        print("DEMAND PATTERN ANALYSIS")
        print("="*60)

        # Hourly demand
        hourly_demand = self.trip_data.groupby('hour').size()
        peak_hour = hourly_demand.idxmax()

        # Daily demand
        daily_demand = self.trip_data.groupby('day_of_week').size()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_demand = daily_demand.reindex(day_order)

        # Weather impact
        weather_demand = self.trip_data.groupby('weather').agg({
            'trip_id': 'count',
            'fare': 'mean',
            'distance_km': 'mean'
        }).rename(columns={'trip_id': 'n_trips'})

        print(f"\nPeak Demand Hour: {peak_hour}:00 ({hourly_demand[peak_hour]} trips)")
        print(f"Weekend vs Weekday: {self.trip_data['is_weekend'].mean()*100:.1f}% weekend trips")
        print(f"Rush Hour Trips: {self.trip_data['is_rush_hour'].sum()} ({self.trip_data['is_rush_hour'].mean()*100:.1f}%)")

        print("\nWeather Impact:")
        for weather, stats in weather_demand.iterrows():
            print(f"  {weather}: {stats['n_trips']} trips, ${stats['fare']:.2f} avg fare")

        return hourly_demand, daily_demand, weather_demand

    def predict_demand(self):
        """Build demand prediction model"""
        print("\n" + "="*60)
        print("BUILDING DEMAND PREDICTION MODEL")
        print("="*60)

        # Aggregate to zone-hour level
        demand_features = self.trip_data.groupby(['demand_zone', 'hour', 'day_of_week', 'weather']).agg({
            'trip_id': 'count',
            'is_rush_hour': 'first',
            'is_weekend': 'first'
        }).reset_index()

        demand_features.rename(columns={'trip_id': 'n_trips'}, inplace=True)

        # Encode categorical variables
        demand_features['day_num'] = pd.Categorical(
            demand_features['day_of_week'],
            categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ).codes

        weather_map = {'Clear': 0, 'Rain': 1, 'Snow': 2}
        demand_features['weather_code'] = demand_features['weather'].map(weather_map)

        # Features and target
        feature_cols = ['demand_zone', 'hour', 'day_num', 'is_rush_hour', 'is_weekend', 'weather_code']
        X = demand_features[feature_cols]
        y = demand_features['n_trips']

        # Train model
        self.demand_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.demand_model.fit(X, y)

        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.demand_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFeature Importances:")
        for _, row in importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Add predictions
        demand_features['predicted_trips'] = self.demand_model.predict(X)

        return demand_features, importance

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Pickup Heatmap
        ax = axes[0, 0]
        scatter = ax.scatter(
            self.trip_data['pickup_longitude'],
            self.trip_data['pickup_latitude'],
            c=self.trip_data['demand_zone'],
            s=10,
            alpha=0.4,
            cmap='tab10'
        )
        if self.zones is not None:
            ax.scatter(self.zones['center_lon'], self.zones['center_lat'],
                      c='red', s=300, marker='*', edgecolors='darkred',
                      linewidths=2, label='Zone Centers')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Taxi Pickup Locations by Zone', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Hourly Demand
        ax = axes[0, 1]
        hourly = self.trip_data.groupby('hour').size()
        ax.plot(hourly.index, hourly.values, marker='o', linewidth=2, color='steelblue')
        ax.fill_between(hourly.index, hourly.values, alpha=0.3, color='steelblue')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Number of Trips')
        ax.set_title('Hourly Demand Pattern', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 3. Daily Demand
        ax = axes[0, 2]
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = self.trip_data.groupby('day_of_week').size().reindex(day_order)
        colors = ['steelblue'] * 5 + ['coral', 'coral']
        ax.bar(range(7), daily.values, color=colors, edgecolor='black')
        ax.set_xticks(range(7))
        ax.set_xticklabels([d[:3] for d in day_order], rotation=45)
        ax.set_ylabel('Number of Trips')
        ax.set_title('Daily Demand Pattern', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Zone Demand
        ax = axes[1, 0]
        if self.zones is not None:
            top_zones = self.zones.head(10)
            ax.barh(range(len(top_zones)), top_zones['n_trips'],
                   color='#3498db', edgecolor='navy')
            ax.set_yticks(range(len(top_zones)))
            ax.set_yticklabels([f"Zone {z}" for z in top_zones['zone_id']])
            ax.set_xlabel('Number of Trips')
            ax.set_title('Top 10 Demand Zones', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

        # 5. Trip Distance Distribution
        ax = axes[1, 1]
        ax.hist(self.trip_data['distance_km'], bins=50, color='#2ecc71',
               edgecolor='darkgreen', alpha=0.7)
        ax.axvline(self.trip_data['distance_km'].mean(), color='red',
                  linestyle='--', linewidth=2, label=f'Mean: {self.trip_data["distance_km"].mean():.2f} km')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Frequency')
        ax.set_title('Trip Distance Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 6. Fare vs Distance
        ax = axes[1, 2]
        ax.scatter(self.trip_data['distance_km'], self.trip_data['fare'],
                  alpha=0.3, s=20, c='purple', edgecolors='none')
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Fare ($)')
        ax.set_title('Fare vs Distance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('taxi_demand_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'taxi_demand_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("TAXI DEMAND PREDICTION - GEOSPATIAL ANALYSIS")
    print("="*60)

    # Initialize predictor
    predictor = TaxiDemandPredictor(city_name="New York")

    # Generate data
    predictor.create_sample_data()

    # Create demand zones
    zones = predictor.create_demand_zones(n_zones=10)

    # Analyze patterns
    hourly, daily, weather = predictor.analyze_demand_patterns()

    # Predict demand
    demand_pred, importance = predictor.predict_demand()

    # Visualize
    predictor.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
