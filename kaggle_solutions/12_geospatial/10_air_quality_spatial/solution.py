"""
Air Quality Spatial Analysis - Geospatial Analysis
Analyze and predict air quality patterns across urban areas

Dataset: Synthetic air quality monitoring data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class AirQualitySpatialAnalyzer:
    """Air quality spatial analysis and prediction"""

    def __init__(self, city_name="Metro City"):
        self.city_name = city_name
        self.air_quality_data = None
        self.model = None

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
        """Generate synthetic air quality monitoring data"""
        np.random.seed(42)

        # City center (typically higher pollution)
        center_lat, center_lon = 39.9042, 116.4074  # Beijing-like coordinates
        lat_spread, lon_spread = 0.20, 0.25

        # Industrial zones (pollution sources)
        n_industrial_zones = 5
        industrial_lats = center_lat + np.random.uniform(-lat_spread*0.6, lat_spread*0.6, n_industrial_zones)
        industrial_lons = center_lon + np.random.uniform(-lon_spread*0.6, lon_spread*0.6, n_industrial_zones)

        # Monitoring stations
        n_stations = 150
        stations = []

        for i in range(n_stations):
            lat = center_lat + np.random.uniform(-lat_spread, lat_spread)
            lon = center_lon + np.random.uniform(-lon_spread, lon_spread)

            # Distance to city center
            dist_to_center = self.haversine_distance(lat, lon, center_lat, center_lon)

            # Distance to nearest industrial zone
            dist_to_industrial = min([
                self.haversine_distance(lat, lon, ind_lat, ind_lon)
                for ind_lat, ind_lon in zip(industrial_lats, industrial_lons)
            ])

            # Land use and traffic
            if dist_to_center < 5:
                land_use = np.random.choice(['Commercial', 'Residential', 'Industrial'],
                                           p=[0.4, 0.5, 0.1])
                traffic_density = np.random.uniform(5000, 15000)  # vehicles/day
                green_space_pct = np.random.uniform(5, 20)
            elif dist_to_center < 12:
                land_use = np.random.choice(['Residential', 'Commercial', 'Park'],
                                           p=[0.6, 0.2, 0.2])
                traffic_density = np.random.uniform(2000, 8000)
                green_space_pct = np.random.uniform(15, 40)
            else:
                land_use = np.random.choice(['Residential', 'Park', 'Rural'],
                                           p=[0.5, 0.3, 0.2])
                traffic_density = np.random.uniform(500, 3000)
                green_space_pct = np.random.uniform(30, 70)

            # Meteorological factors
            temperature = np.random.normal(15, 8).clip(-5, 35)
            wind_speed = np.random.gamma(2, 2).clip(0, 15)  # m/s
            humidity = np.random.normal(65, 15).clip(30, 95)
            precipitation = np.random.exponential(2).clip(0, 50)  # mm

            # Season effect
            season = np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'])
            season_multiplier = {'Winter': 1.5, 'Spring': 1.1, 'Summer': 0.9, 'Fall': 1.2}[season]

            # Calculate PM2.5 concentration
            base_pm25 = 30  # baseline urban background

            # Distance effects
            pm25 = base_pm25 + 50 * np.exp(-dist_to_center / 8)  # Urban center effect
            pm25 += 40 * np.exp(-dist_to_industrial / 3)  # Industrial effect

            # Traffic contribution
            pm25 += traffic_density / 500

            # Green space reduction
            pm25 -= green_space_pct * 0.3

            # Meteorological effects
            if wind_speed < 2:
                pm25 *= 1.4  # Stagnant air
            else:
                pm25 *= (1 - wind_speed / 40)  # Wind dispersion

            if temperature < 5:
                pm25 *= 1.2  # Winter heating

            # Season
            pm25 *= season_multiplier

            # Add noise
            pm25 = pm25 * np.random.uniform(0.8, 1.2)
            pm25 = max(5, pm25)

            # Calculate PM10 (correlated with PM2.5)
            pm10 = pm25 * np.random.uniform(1.8, 2.5)

            # Calculate NO2
            no2_base = 20
            no2 = no2_base + traffic_density / 300 + 30 * np.exp(-dist_to_industrial / 3)
            no2 = no2 * np.random.uniform(0.85, 1.15)
            no2 = max(5, no2)

            # Calculate O3 (higher in summer, suburban areas)
            o3_base = 40
            o3 = o3_base
            if season == 'Summer':
                o3 *= 1.6
            if 5 < dist_to_center < 15:  # Suburban max
                o3 *= 1.3
            o3 = o3 * np.random.uniform(0.8, 1.2)
            o3 = max(10, o3)

            # Air Quality Index (simplified, based on PM2.5)
            if pm25 <= 12:
                aqi_category = 'Good'
            elif pm25 <= 35:
                aqi_category = 'Moderate'
            elif pm25 <= 55:
                aqi_category = 'Unhealthy for Sensitive'
            elif pm25 <= 150:
                aqi_category = 'Unhealthy'
            else:
                aqi_category = 'Very Unhealthy'

            stations.append({
                'station_id': f'AQ{i+1:03d}',
                'latitude': lat,
                'longitude': lon,
                'pm25': pm25,
                'pm10': pm10,
                'no2': no2,
                'o3': o3,
                'dist_to_center_km': dist_to_center,
                'dist_to_industrial_km': dist_to_industrial,
                'land_use': land_use,
                'traffic_density': traffic_density,
                'green_space_pct': green_space_pct,
                'temperature': temperature,
                'wind_speed': wind_speed,
                'humidity': humidity,
                'precipitation': precipitation,
                'season': season,
                'aqi_category': aqi_category
            })

        self.air_quality_data = pd.DataFrame(stations)

        print(f"✓ Generated {len(self.air_quality_data)} air quality monitoring stations")
        print(f"\nPM2.5 Statistics:")
        print(f"  Range: {self.air_quality_data['pm25'].min():.1f} - {self.air_quality_data['pm25'].max():.1f} μg/m³")
        print(f"  Mean: {self.air_quality_data['pm25'].mean():.1f} μg/m³")
        print(f"  Median: {self.air_quality_data['pm25'].median():.1f} μg/m³")

        print(f"\nAQI Category Distribution:")
        for category, count in self.air_quality_data['aqi_category'].value_counts().items():
            print(f"  {category}: {count} ({100*count/len(self.air_quality_data):.1f}%)")

    def analyze_spatial_patterns(self):
        """Analyze spatial pollution patterns"""
        print("\n" + "="*60)
        print("SPATIAL PATTERN ANALYSIS")
        print("="*60)

        # Pollution by distance
        distance_bins = pd.cut(
            self.air_quality_data['dist_to_center_km'],
            bins=[0, 5, 12, 30],
            labels=['City Center', 'Urban', 'Suburban']
        )
        self.air_quality_data['area_type'] = distance_bins

        area_pollution = self.air_quality_data.groupby('area_type')[['pm25', 'no2', 'o3']].mean()

        print("\nPollution Levels by Area:")
        for area, values in area_pollution.iterrows():
            print(f"\n{area}:")
            print(f"  PM2.5: {values['pm25']:.1f} μg/m³")
            print(f"  NO2: {values['no2']:.1f} μg/m³")
            print(f"  O3: {values['o3']:.1f} μg/m³")

        # Land use analysis
        land_use_pollution = self.air_quality_data.groupby('land_use')['pm25'].mean().sort_values(ascending=False)

        print("\nPM2.5 by Land Use:")
        for land_use, pm25 in land_use_pollution.items():
            print(f"  {land_use}: {pm25:.1f} μg/m³")

        # Correlations
        print("\nKey Correlations with PM2.5:")
        correlations = self.air_quality_data[[
            'pm25', 'traffic_density', 'green_space_pct',
            'wind_speed', 'dist_to_center_km'
        ]].corr()['pm25'].sort_values(ascending=False)

        for factor, corr in correlations.items():
            if factor != 'pm25':
                print(f"  {factor}: {corr:.3f}")

        return area_pollution, land_use_pollution

    def train_prediction_model(self):
        """Train air quality prediction model"""
        print("\n" + "="*60)
        print("TRAINING PREDICTION MODEL")
        print("="*60)

        # Features
        feature_cols = [
            'dist_to_center_km', 'dist_to_industrial_km',
            'traffic_density', 'green_space_pct',
            'temperature', 'wind_speed', 'humidity'
        ]

        # Encode categorical
        land_dummies = pd.get_dummies(self.air_quality_data['land_use'], prefix='land')
        season_dummies = pd.get_dummies(self.air_quality_data['season'], prefix='season')

        X = pd.concat([
            self.air_quality_data[feature_cols],
            land_dummies,
            season_dummies
        ], axis=1)

        y = self.air_quality_data['pm25']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        print(f"\nTrain MAE: {train_mae:.2f} μg/m³")
        print(f"Test MAE: {test_mae:.2f} μg/m³")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Feature Importances:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        self.air_quality_data['predicted_pm25'] = self.model.predict(X)

        return feature_importance

    def create_pollution_map(self, resolution=80):
        """Create interpolated pollution map"""
        print("\n" + "="*60)
        print("CREATING POLLUTION SURFACE MAP")
        print("="*60)

        # Get bounds
        lat_min, lat_max = self.air_quality_data['latitude'].min(), self.air_quality_data['latitude'].max()
        lon_min, lon_max = self.air_quality_data['longitude'].min(), self.air_quality_data['longitude'].max()

        # Create grid
        grid_lat = np.linspace(lat_min, lat_max, resolution)
        grid_lon = np.linspace(lon_min, lon_max, resolution)
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

        # Interpolate PM2.5
        points = self.air_quality_data[['longitude', 'latitude']].values
        values = self.air_quality_data['pm25'].values

        grid_pm25 = griddata(
            points, values,
            (grid_lon_mesh, grid_lat_mesh),
            method='cubic'
        )

        # Apply smoothing
        grid_pm25_smooth = gaussian_filter(grid_pm25, sigma=1.5)

        print(f"✓ Created {resolution}x{resolution} pollution surface")

        return grid_lat, grid_lon, grid_pm25_smooth

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. PM2.5 Spatial Map
        ax1 = fig.add_subplot(gs[0:2, 0:2])

        # Create pollution surface
        grid_lat, grid_lon, grid_pm25 = self.create_pollution_map(resolution=60)

        contour = ax1.contourf(grid_lon, grid_lat, grid_pm25,
                              levels=15, cmap='RdYlGn_r', alpha=0.7)
        plt.colorbar(contour, ax=ax1, label='PM2.5 (μg/m³)')

        # Overlay stations
        scatter = ax1.scatter(
            self.air_quality_data['longitude'],
            self.air_quality_data['latitude'],
            c=self.air_quality_data['pm25'],
            s=50,
            cmap='RdYlGn_r',
            edgecolors='black',
            linewidths=0.5,
            alpha=0.8
        )

        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('PM2.5 Spatial Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. PM2.5 by Distance
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.scatter(self.air_quality_data['dist_to_center_km'],
                   self.air_quality_data['pm25'],
                   alpha=0.5, s=40, c='coral', edgecolors='black', linewidths=0.5)
        ax2.set_xlabel('Distance from Center (km)')
        ax2.set_ylabel('PM2.5 (μg/m³)')
        ax2.set_title('PM2.5 vs Distance', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. AQI Distribution
        ax3 = fig.add_subplot(gs[1, 2])
        aqi_order = ['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy']
        aqi_counts = self.air_quality_data['aqi_category'].value_counts().reindex(
            [cat for cat in aqi_order if cat in self.air_quality_data['aqi_category'].values]
        )
        colors_aqi = ['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#8f3f97'][:len(aqi_counts)]
        ax3.barh(range(len(aqi_counts)), aqi_counts.values,
                color=colors_aqi, edgecolor='black')
        ax3.set_yticks(range(len(aqi_counts)))
        ax3.set_yticklabels([cat[:15] for cat in aqi_counts.index], fontsize=8)
        ax3.set_xlabel('Count')
        ax3.set_title('AQI Distribution', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. Pollutants Comparison
        ax4 = fig.add_subplot(gs[2, 0])
        pollutant_means = self.air_quality_data[['pm25', 'pm10', 'no2', 'o3']].mean()
        ax4.bar(range(len(pollutant_means)), pollutant_means.values,
               color=['#e74c3c', '#e67e22', '#f39c12', '#3498db'], edgecolor='black')
        ax4.set_xticks(range(len(pollutant_means)))
        ax4.set_xticklabels(pollutant_means.index)
        ax4.set_ylabel('Concentration (μg/m³)')
        ax4.set_title('Average Pollutant Levels', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Land Use Impact
        ax5 = fig.add_subplot(gs[2, 1])
        land_pm25 = self.air_quality_data.groupby('land_use')['pm25'].mean().sort_values()
        ax5.barh(range(len(land_pm25)), land_pm25.values,
                color='steelblue', edgecolor='navy')
        ax5.set_yticks(range(len(land_pm25)))
        ax5.set_yticklabels(land_pm25.index, fontsize=9)
        ax5.set_xlabel('Average PM2.5 (μg/m³)')
        ax5.set_title('PM2.5 by Land Use', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Predicted vs Actual
        ax6 = fig.add_subplot(gs[2, 2])
        if 'predicted_pm25' in self.air_quality_data.columns:
            ax6.scatter(self.air_quality_data['predicted_pm25'],
                       self.air_quality_data['pm25'],
                       alpha=0.5, s=40, c='purple', edgecolors='black', linewidths=0.5)
            min_val = min(self.air_quality_data['pm25'].min(),
                         self.air_quality_data['predicted_pm25'].min())
            max_val = max(self.air_quality_data['pm25'].max(),
                         self.air_quality_data['predicted_pm25'].max())
            ax6.plot([min_val, max_val], [min_val, max_val],
                    'r--', linewidth=2, label='Perfect Prediction')
            ax6.set_xlabel('Predicted PM2.5 (μg/m³)')
            ax6.set_ylabel('Actual PM2.5 (μg/m³)')
            ax6.set_title('Prediction Accuracy', fontsize=11, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        plt.savefig('air_quality_spatial_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'air_quality_spatial_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("AIR QUALITY SPATIAL ANALYSIS - GEOSPATIAL ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = AirQualitySpatialAnalyzer(city_name="Beijing")

    # Generate data
    analyzer.create_sample_data()

    # Analyze patterns
    area_pollution, land_use_pollution = analyzer.analyze_spatial_patterns()

    # Train model
    feature_importance = analyzer.train_prediction_model()

    # Visualize
    analyzer.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
