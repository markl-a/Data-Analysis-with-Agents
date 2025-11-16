"""
Flood Risk Assessment - Geospatial Analysis
Assess flood risk using topographic and hydrological data

Dataset: Synthetic elevation and watershed data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class FloodRiskAssessor:
    """Flood risk assessment using geospatial and hydrological analysis"""

    def __init__(self, region_name="River Valley"):
        self.region_name = region_name
        self.risk_data = None
        self.model = None

    def create_sample_data(self):
        """Generate synthetic flood risk data"""
        np.random.seed(42)

        # Region bounds
        center_lat, center_lon = 29.7604, -95.3698  # Houston-like coordinates
        lat_spread, lon_spread = 0.15, 0.20

        # Create river path (main flood source)
        river_path_lat = np.linspace(center_lat - lat_spread*0.6,
                                     center_lat + lat_spread*0.6, 50)
        river_path_lon = center_lon + 0.02 * np.sin(np.linspace(0, 4*np.pi, 50))

        n_locations = 800
        locations = []

        for i in range(n_locations):
            lat = center_lat + np.random.uniform(-lat_spread, lat_spread)
            lon = center_lon + np.random.uniform(-lon_spread, lon_spread)

            # Distance to river
            distances_to_river = np.sqrt(
                (river_path_lat - lat)**2 + (river_path_lon - lon)**2
            )
            dist_to_river_km = distances_to_river.min() * 111  # Convert degrees to km

            # Elevation (lower near river)
            base_elevation = 50  # meters above sea level
            elevation = base_elevation + dist_to_river_km * 3 + np.random.uniform(-5, 10)
            elevation = max(0, elevation)

            # Slope
            slope = np.random.beta(2, 5) * 20  # degrees

            # Soil properties
            soil_type = np.random.choice(
                ['Clay', 'Sand', 'Silt', 'Loam', 'Rocky'],
                p=[0.25, 0.20, 0.20, 0.25, 0.10]
            )
            permeability_rating = {
                'Clay': 20, 'Sand': 80, 'Silt': 40, 'Loam': 60, 'Rocky': 30
            }[soil_type]

            # Land use
            if dist_to_river_km < 2:
                land_use = np.random.choice(['Residential', 'Agricultural', 'Wetland'],
                                           p=[0.3, 0.5, 0.2])
            else:
                land_use = np.random.choice(['Residential', 'Commercial', 'Forest', 'Agricultural'],
                                           p=[0.4, 0.2, 0.2, 0.2])

            # Drainage
            drainage_density = np.random.uniform(0.5, 5.0)  # km per sq km
            impervious_surface_pct = {
                'Residential': np.random.uniform(30, 60),
                'Commercial': np.random.uniform(60, 90),
                'Agricultural': np.random.uniform(5, 15),
                'Forest': np.random.uniform(0, 5),
                'Wetland': np.random.uniform(0, 10)
            }.get(land_use, 30)

            # Climate factors
            annual_rainfall_mm = np.random.normal(1200, 200).clip(600, 2000)
            storm_frequency = np.random.poisson(8)  # Number of severe storms per year

            # Infrastructure
            has_levee = 1 if (dist_to_river_km < 1.5 and np.random.random() < 0.4) else 0
            has_drainage = 1 if (land_use in ['Residential', 'Commercial'] and np.random.random() < 0.7) else 0

            # Historical floods
            historical_floods = np.random.poisson(3) if dist_to_river_km < 3 else np.random.poisson(1)

            # Calculate flood risk score
            risk_score = 0

            # Distance to water body
            if dist_to_river_km < 0.5:
                risk_score += 40
            elif dist_to_river_km < 1.5:
                risk_score += 25
            elif dist_to_river_km < 3:
                risk_score += 10

            # Elevation
            if elevation < 10:
                risk_score += 30
            elif elevation < 25:
                risk_score += 15
            elif elevation < 50:
                risk_score += 5

            # Slope (flat areas flood more)
            if slope < 2:
                risk_score += 15
            elif slope < 5:
                risk_score += 8

            # Soil permeability (low = high risk)
            risk_score += (100 - permeability_rating) / 5

            # Impervious surfaces
            risk_score += impervious_surface_pct / 5

            # Rainfall
            if annual_rainfall_mm > 1400:
                risk_score += (annual_rainfall_mm - 1400) / 50

            # Drainage
            if has_drainage:
                risk_score -= 10
            risk_score -= drainage_density * 2

            # Protection
            if has_levee:
                risk_score -= 15

            # Historical
            risk_score += historical_floods * 3

            # Add variability
            risk_score = risk_score * np.random.uniform(0.85, 1.15)
            risk_score = max(0, risk_score)

            # Classify risk
            if risk_score < 25:
                risk_level = 'Low'
            elif risk_score < 50:
                risk_level = 'Moderate'
            elif risk_score < 75:
                risk_level = 'High'
            else:
                risk_level = 'Extreme'

            locations.append({
                'location_id': f'LOC{i+1:04d}',
                'latitude': lat,
                'longitude': lon,
                'elevation_m': elevation,
                'slope_degrees': slope,
                'dist_to_river_km': dist_to_river_km,
                'soil_type': soil_type,
                'permeability': permeability_rating,
                'land_use': land_use,
                'impervious_surface_pct': impervious_surface_pct,
                'drainage_density': drainage_density,
                'annual_rainfall_mm': annual_rainfall_mm,
                'storm_frequency': storm_frequency,
                'has_levee': has_levee,
                'has_drainage': has_drainage,
                'historical_floods': historical_floods,
                'risk_score': risk_score,
                'risk_level': risk_level
            })

        self.risk_data = pd.DataFrame(locations)

        print(f"✓ Generated {len(self.risk_data)} flood risk assessments")
        print(f"\nRisk Level Distribution:")
        for level, count in self.risk_data['risk_level'].value_counts().items():
            print(f"  {level}: {count} ({100*count/len(self.risk_data):.1f}%)")

        print(f"\nElevation range: {self.risk_data['elevation_m'].min():.1f}m - {self.risk_data['elevation_m'].max():.1f}m")
        print(f"Mean distance to river: {self.risk_data['dist_to_river_km'].mean():.2f} km")

    def analyze_flood_factors(self):
        """Analyze key flood risk factors"""
        print("\n" + "="*60)
        print("FLOOD RISK FACTOR ANALYSIS")
        print("="*60)

        # High vs low risk comparison
        high_risk = self.risk_data[self.risk_data['risk_level'].isin(['High', 'Extreme'])]
        low_risk = self.risk_data[self.risk_data['risk_level'] == 'Low']

        print(f"\nHigh Risk Areas (n={len(high_risk)}):")
        print(f"  Avg Elevation: {high_risk['elevation_m'].mean():.1f}m")
        print(f"  Avg Distance to River: {high_risk['dist_to_river_km'].mean():.2f} km")
        print(f"  Avg Impervious Surface: {high_risk['impervious_surface_pct'].mean():.1f}%")
        print(f"  Levee Protection: {high_risk['has_levee'].sum()} ({100*high_risk['has_levee'].mean():.1f}%)")

        print(f"\nLow Risk Areas (n={len(low_risk)}):")
        print(f"  Avg Elevation: {low_risk['elevation_m'].mean():.1f}m")
        print(f"  Avg Distance to River: {low_risk['dist_to_river_km'].mean():.2f} km")
        print(f"  Avg Impervious Surface: {low_risk['impervious_surface_pct'].mean():.1f}%")

        # Risk by soil type
        soil_risk = self.risk_data.groupby('soil_type')['risk_score'].mean().sort_values(ascending=False)
        print("\nAverage Risk Score by Soil Type:")
        for soil, score in soil_risk.items():
            print(f"  {soil}: {score:.1f}")

        # Risk by land use
        land_use_risk = self.risk_data.groupby('land_use')['risk_score'].mean().sort_values(ascending=False)
        print("\nAverage Risk Score by Land Use:")
        for land_use, score in land_use_risk.items():
            print(f"  {land_use}: {score:.1f}")

        return soil_risk, land_use_risk

    def train_flood_model(self):
        """Train flood risk prediction model"""
        print("\n" + "="*60)
        print("TRAINING FLOOD RISK MODEL")
        print("="*60)

        # Features
        feature_cols = [
            'elevation_m', 'slope_degrees', 'dist_to_river_km',
            'permeability', 'impervious_surface_pct', 'drainage_density',
            'annual_rainfall_mm', 'storm_frequency', 'has_levee',
            'has_drainage', 'historical_floods'
        ]

        # Encode categorical
        soil_dummies = pd.get_dummies(self.risk_data['soil_type'], prefix='soil')
        land_dummies = pd.get_dummies(self.risk_data['land_use'], prefix='land')

        X = pd.concat([
            self.risk_data[feature_cols],
            soil_dummies,
            land_dummies
        ], axis=1)

        # Binary target: High risk or not
        y = (self.risk_data['risk_level'].isin(['High', 'Extreme'])).astype(int)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        test_proba = self.model.predict_proba(X_test)[:, 1]

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        test_auc = roc_auc_score(y_test, test_proba)

        print(f"\nTrain Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test AUC-ROC: {test_auc:.4f}")

        print("\nTest Classification Report:")
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

        self.risk_data['predicted_high_risk'] = self.model.predict(X)
        self.risk_data['risk_probability'] = self.model.predict_proba(X)[:, 1]

        return feature_importance

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Flood Risk Map
        ax = axes[0, 0]
        risk_colors = {'Low': '#2ecc71', 'Moderate': '#f39c12',
                      'High': '#e67e22', 'Extreme': '#e74c3c'}
        for risk_level in ['Low', 'Moderate', 'High', 'Extreme']:
            data = self.risk_data[self.risk_data['risk_level'] == risk_level]
            ax.scatter(data['longitude'], data['latitude'],
                      c=risk_colors[risk_level], s=50, alpha=0.6,
                      label=risk_level, edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Flood Risk Map', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Elevation Map
        ax = axes[0, 1]
        scatter = ax.scatter(
            self.risk_data['longitude'],
            self.risk_data['latitude'],
            c=self.risk_data['elevation_m'],
            s=50,
            cmap='terrain',
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Elevation (m)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Elevation Map', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 3. Distance vs Risk
        ax = axes[0, 2]
        ax.scatter(self.risk_data['dist_to_river_km'],
                  self.risk_data['risk_score'],
                  alpha=0.4, s=40, c='steelblue', edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Distance to River (km)')
        ax.set_ylabel('Risk Score')
        ax.set_title('Distance vs Flood Risk', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 4. Risk Distribution
        ax = axes[1, 0]
        risk_order = ['Low', 'Moderate', 'High', 'Extreme']
        risk_counts = self.risk_data['risk_level'].value_counts().reindex(risk_order)
        colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        bars = ax.bar(range(len(risk_counts)), risk_counts.values,
                     color=colors, edgecolor='black')
        ax.set_xticks(range(len(risk_counts)))
        ax.set_xticklabels(risk_order)
        ax.set_ylabel('Number of Locations')
        ax.set_title('Risk Level Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')

        # 5. Elevation vs Risk
        ax = axes[1, 1]
        ax.scatter(self.risk_data['elevation_m'],
                  self.risk_data['risk_score'],
                  alpha=0.4, s=40, c='coral', edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Elevation (m)')
        ax.set_ylabel('Risk Score')
        ax.set_title('Elevation vs Flood Risk', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 6. Land Use Risk
        ax = axes[1, 2]
        land_risk = self.risk_data.groupby('land_use')['risk_score'].mean().sort_values()
        ax.barh(range(len(land_risk)), land_risk.values,
               color='#3498db', edgecolor='navy')
        ax.set_yticks(range(len(land_risk)))
        ax.set_yticklabels(land_risk.index, fontsize=9)
        ax.set_xlabel('Average Risk Score')
        ax.set_title('Risk by Land Use', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('flood_risk_assessment.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'flood_risk_assessment.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("FLOOD RISK ASSESSMENT - GEOSPATIAL ANALYSIS")
    print("="*60)

    # Initialize assessor
    assessor = FloodRiskAssessor(region_name="River Valley")

    # Generate data
    assessor.create_sample_data()

    # Analyze factors
    soil_risk, land_use_risk = assessor.analyze_flood_factors()

    # Train model
    feature_importance = assessor.train_flood_model()

    # Visualize
    assessor.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
