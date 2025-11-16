"""
Wildfire Risk Prediction - Geospatial Analysis
Predict wildfire risk using environmental and geographic factors

Dataset: Synthetic environmental data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class WildfireRiskPredictor:
    """Wildfire risk prediction using geospatial and environmental features"""

    def __init__(self, region_name="California"):
        self.region_name = region_name
        self.risk_data = None
        self.model = None

    def create_sample_data(self):
        """Generate synthetic wildfire risk data"""
        np.random.seed(42)

        # Region bounds (California-like)
        center_lat, center_lon = 37.2707, -119.2709
        lat_spread, lon_spread = 2.0, 2.5

        n_locations = 1000

        locations = []

        for i in range(n_locations):
            lat = center_lat + np.random.uniform(-lat_spread, lat_spread)
            lon = center_lon + np.random.uniform(-lon_spread, lon_spread)

            # Environmental features
            elevation_m = np.random.uniform(0, 3000)
            slope_degrees = np.random.beta(2, 5) * 40  # Most areas have moderate slope
            aspect = np.random.uniform(0, 360)  # Degrees from north

            # Vegetation
            vegetation_density = np.random.beta(3, 2) * 100  # %
            vegetation_type = np.random.choice(
                ['Forest', 'Grassland', 'Shrubland', 'Mixed', 'Sparse'],
                p=[0.3, 0.2, 0.25, 0.15, 0.1]
            )

            # Climate factors
            temperature_avg = np.random.normal(22, 5).clip(10, 35)  # Celsius
            precipitation_mm = np.random.gamma(2, 50).clip(0, 500)  # Annual
            humidity_avg = np.random.normal(60, 15).clip(20, 90)  # %
            wind_speed_kmh = np.random.gamma(2, 5).clip(0, 40)

            # Human factors
            distance_to_road_km = np.random.exponential(2).clip(0, 20)
            distance_to_urban_km = np.random.exponential(5).clip(0, 50)
            population_density = np.exp(-distance_to_urban_km / 5) * 100

            # Historical data
            fires_in_10km = np.random.poisson(2)
            years_since_fire = np.random.exponential(10).clip(0, 50)

            # Calculate risk score (ground truth)
            risk_score = 0

            # Vegetation risk
            veg_risk = {'Forest': 0.7, 'Grassland': 0.5, 'Shrubland': 0.8,
                       'Mixed': 0.6, 'Sparse': 0.3}
            risk_score += veg_risk[vegetation_type] * 25

            # Climate risk
            if temperature_avg > 25:
                risk_score += (temperature_avg - 25) * 2
            if precipitation_mm < 200:
                risk_score += (200 - precipitation_mm) / 10
            if humidity_avg < 40:
                risk_score += (40 - humidity_avg) * 0.5
            risk_score += wind_speed_kmh * 0.5

            # Topography
            if slope_degrees > 20:
                risk_score += (slope_degrees - 20) * 0.5
            if 135 <= aspect <= 225:  # South-facing slopes (northern hemisphere)
                risk_score += 5

            # Human factors
            if distance_to_road_km < 1:
                risk_score += 10  # Human ignition risk
            if 5 < distance_to_urban_km < 20:  # Wildland-urban interface
                risk_score += 8

            # Historical
            risk_score += fires_in_10km * 3
            if years_since_fire < 5:
                risk_score -= 10  # Recent burn reduces fuel

            # Normalize and add noise
            risk_score = risk_score * np.random.uniform(0.8, 1.2)

            # Classify risk
            if risk_score < 30:
                risk_class = 'Low'
            elif risk_score < 60:
                risk_class = 'Moderate'
            elif risk_score < 90:
                risk_class = 'High'
            else:
                risk_class = 'Extreme'

            locations.append({
                'location_id': f'LOC{i+1:04d}',
                'latitude': lat,
                'longitude': lon,
                'elevation_m': elevation_m,
                'slope_degrees': slope_degrees,
                'aspect_degrees': aspect,
                'vegetation_density': vegetation_density,
                'vegetation_type': vegetation_type,
                'temperature_avg': temperature_avg,
                'precipitation_mm': precipitation_mm,
                'humidity_avg': humidity_avg,
                'wind_speed_kmh': wind_speed_kmh,
                'dist_to_road_km': distance_to_road_km,
                'dist_to_urban_km': distance_to_urban_km,
                'fires_in_10km': fires_in_10km,
                'years_since_fire': years_since_fire,
                'risk_score': risk_score,
                'risk_class': risk_class
            })

        self.risk_data = pd.DataFrame(locations)

        print(f"✓ Generated {len(self.risk_data)} location risk assessments")
        print(f"\nRisk Distribution:")
        for risk, count in self.risk_data['risk_class'].value_counts().items():
            print(f"  {risk}: {count} ({100*count/len(self.risk_data):.1f}%)")

    def analyze_risk_factors(self):
        """Analyze key risk factors"""
        print("\n" + "="*60)
        print("RISK FACTOR ANALYSIS")
        print("="*60)

        # Risk by vegetation type
        veg_risk = self.risk_data.groupby('vegetation_type')['risk_score'].mean().sort_values(ascending=False)
        print("\nAverage Risk Score by Vegetation:")
        for veg, score in veg_risk.items():
            print(f"  {veg}: {score:.1f}")

        # Climate factors
        high_risk = self.risk_data[self.risk_data['risk_class'].isin(['High', 'Extreme'])]
        low_risk = self.risk_data[self.risk_data['risk_class'] == 'Low']

        print(f"\nHigh Risk Areas (n={len(high_risk)}):")
        print(f"  Avg Temperature: {high_risk['temperature_avg'].mean():.1f}°C")
        print(f"  Avg Precipitation: {high_risk['precipitation_mm'].mean():.0f} mm")
        print(f"  Avg Humidity: {high_risk['humidity_avg'].mean():.1f}%")
        print(f"  Avg Wind Speed: {high_risk['wind_speed_kmh'].mean():.1f} km/h")

        print(f"\nLow Risk Areas (n={len(low_risk)}):")
        print(f"  Avg Temperature: {low_risk['temperature_avg'].mean():.1f}°C")
        print(f"  Avg Precipitation: {low_risk['precipitation_mm'].mean():.0f} mm")
        print(f"  Avg Humidity: {low_risk['humidity_avg'].mean():.1f}%")

        return veg_risk

    def train_risk_model(self):
        """Train wildfire risk classification model"""
        print("\n" + "="*60)
        print("TRAINING RISK PREDICTION MODEL")
        print("="*60)

        # Prepare features
        feature_cols = [
            'elevation_m', 'slope_degrees', 'aspect_degrees',
            'vegetation_density', 'temperature_avg', 'precipitation_mm',
            'humidity_avg', 'wind_speed_kmh', 'dist_to_road_km',
            'dist_to_urban_km', 'fires_in_10km', 'years_since_fire'
        ]

        # Encode vegetation type
        veg_dummies = pd.get_dummies(self.risk_data['vegetation_type'], prefix='veg')
        X = pd.concat([self.risk_data[feature_cols], veg_dummies], axis=1)

        # Binary classification: High risk or not
        y = (self.risk_data['risk_class'].isin(['High', 'Extreme'])).astype(int)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        test_proba = self.model.predict_proba(X_test)[:, 1]

        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        test_auc = roc_auc_score(y_test, test_proba)

        print(f"\nTrain Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test AUC-ROC: {test_auc:.4f}")

        print("\nTest Set Classification Report:")
        print(classification_report(y_test, test_pred,
                                   target_names=['Low/Moderate', 'High/Extreme']))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Feature Importances:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Add predictions to data
        self.risk_data['predicted_high_risk'] = self.model.predict(X)
        self.risk_data['risk_probability'] = self.model.predict_proba(X)[:, 1]

        return feature_importance

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Risk Map
        ax = axes[0, 0]
        risk_colors = {'Low': '#2ecc71', 'Moderate': '#f39c12',
                      'High': '#e67e22', 'Extreme': '#e74c3c'}
        for risk_class in ['Low', 'Moderate', 'High', 'Extreme']:
            data = self.risk_data[self.risk_data['risk_class'] == risk_class]
            ax.scatter(data['longitude'], data['latitude'],
                      c=risk_colors[risk_class], s=50, alpha=0.6,
                      label=risk_class, edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Wildfire Risk Map', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Risk by Vegetation
        ax = axes[0, 1]
        veg_risk = self.risk_data.groupby('vegetation_type')['risk_score'].mean().sort_values()
        ax.barh(range(len(veg_risk)), veg_risk.values,
               color='coral', edgecolor='darkred')
        ax.set_yticks(range(len(veg_risk)))
        ax.set_yticklabels(veg_risk.index)
        ax.set_xlabel('Average Risk Score')
        ax.set_title('Risk by Vegetation Type', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # 3. Temperature vs Precipitation
        ax = axes[0, 2]
        scatter = ax.scatter(
            self.risk_data['precipitation_mm'],
            self.risk_data['temperature_avg'],
            c=self.risk_data['risk_score'],
            s=50,
            cmap='YlOrRd',
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Risk Score')
        ax.set_xlabel('Annual Precipitation (mm)')
        ax.set_ylabel('Average Temperature (°C)')
        ax.set_title('Climate Factors vs Risk', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 4. Risk Distribution
        ax = axes[1, 0]
        risk_order = ['Low', 'Moderate', 'High', 'Extreme']
        risk_counts = self.risk_data['risk_class'].value_counts().reindex(risk_order)
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        bars = ax.bar(range(len(risk_counts)), risk_counts.values,
                     color=colors, edgecolor='black')
        ax.set_xticks(range(len(risk_counts)))
        ax.set_xticklabels(risk_order)
        ax.set_ylabel('Number of Locations')
        ax.set_title('Risk Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')

        # 5. Slope vs Elevation
        ax = axes[1, 1]
        scatter = ax.scatter(
            self.risk_data['elevation_m'],
            self.risk_data['slope_degrees'],
            c=self.risk_data['risk_score'],
            s=40,
            cmap='YlOrRd',
            alpha=0.5,
            edgecolors='black',
            linewidths=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Risk Score')
        ax.set_xlabel('Elevation (m)')
        ax.set_ylabel('Slope (degrees)')
        ax.set_title('Topography vs Risk', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 6. Prediction Probability
        ax = axes[1, 2]
        if 'risk_probability' in self.risk_data.columns:
            ax.hist(self.risk_data['risk_probability'], bins=30,
                   color='steelblue', edgecolor='navy', alpha=0.7)
            ax.axvline(0.5, color='red', linestyle='--', linewidth=2,
                      label='Decision Threshold')
            ax.set_xlabel('High Risk Probability')
            ax.set_ylabel('Frequency')
            ax.set_title('Risk Prediction Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('wildfire_risk_prediction.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'wildfire_risk_prediction.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("WILDFIRE RISK PREDICTION - GEOSPATIAL ANALYSIS")
    print("="*60)

    # Initialize predictor
    predictor = WildfireRiskPredictor(region_name="California")

    # Generate data
    predictor.create_sample_data()

    # Analyze risk factors
    veg_risk = predictor.analyze_risk_factors()

    # Train model
    feature_importance = predictor.train_risk_model()

    # Visualize
    predictor.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
