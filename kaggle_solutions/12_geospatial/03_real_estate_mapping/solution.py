"""
Real Estate Price Mapping - Geospatial Analysis
Analyze and predict real estate prices using geographic features

Dataset: Synthetic property listing data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class RealEstatePriceMapper:
    """Real estate price analysis and mapping"""

    def __init__(self, city_name="Metro City"):
        self.city_name = city_name
        self.properties = None
        self.model = None
        self.poi_data = None

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
        """Generate synthetic real estate data"""
        np.random.seed(42)

        # City center and high-value areas
        city_center = (37.7749, -122.4194)  # SF-like coordinates
        lat_spread, lon_spread = 0.12, 0.15

        # Define valuable neighborhoods
        n_neighborhoods = 5
        neighborhood_centers = []
        neighborhood_values = []

        for i in range(n_neighborhoods):
            lat = city_center[0] + np.random.uniform(-lat_spread*0.6, lat_spread*0.6)
            lon = city_center[1] + np.random.uniform(-lon_spread*0.6, lon_spread*0.6)
            value_multiplier = np.random.uniform(0.7, 1.5)
            neighborhood_centers.append((lat, lon))
            neighborhood_values.append(value_multiplier)

        # Generate properties
        n_properties = 1000
        properties = []

        for i in range(n_properties):
            # Assign to neighborhood
            neighborhood_idx = np.random.randint(0, n_neighborhoods)
            center_lat, center_lon = neighborhood_centers[neighborhood_idx]
            value_mult = neighborhood_values[neighborhood_idx]

            # Property location
            lat = center_lat + np.random.normal(0, 0.015)
            lon = center_lon + np.random.normal(0, 0.02)

            # Property features
            bedrooms = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.30, 0.35, 0.15, 0.05])
            bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3], p=[0.20, 0.15, 0.40, 0.15, 0.10])
            sqft = np.random.normal(1200 + bedrooms * 400, 300).clip(500, 5000)
            age = np.random.exponential(20).clip(0, 100)
            lot_size = np.random.lognormal(8, 0.5).clip(1000, 20000)

            # Distance to city center
            dist_to_center = self.haversine_distance(lat, lon, city_center[0], city_center[1])

            # Base price calculation
            base_price = 200000
            base_price += bedrooms * 80000
            base_price += bathrooms * 40000
            base_price += sqft * 150
            base_price += lot_size * 10
            base_price -= age * 2000
            base_price -= dist_to_center * 15000  # Location premium
            base_price *= value_mult  # Neighborhood premium

            # Add noise
            price = base_price * np.random.uniform(0.85, 1.15)
            price = max(100000, price)

            properties.append({
                'property_id': f'PROP{i+1:04d}',
                'latitude': lat,
                'longitude': lon,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'sqft': int(sqft),
                'lot_size_sqft': int(lot_size),
                'age_years': int(age),
                'dist_to_center_km': dist_to_center,
                'neighborhood_id': neighborhood_idx,
                'price': int(price)
            })

        self.properties = pd.DataFrame(properties)

        # Generate Points of Interest (POI)
        self._generate_poi(city_center, lat_spread, lon_spread)

        # Add POI-based features
        self._add_poi_features()

        print(f"✓ Generated {len(self.properties)} property listings")
        print(f"✓ Price range: ${self.properties['price'].min():,.0f} - ${self.properties['price'].max():,.0f}")
        print(f"✓ Median price: ${self.properties['price'].median():,.0f}")
        print(f"✓ Mean price: ${self.properties['price'].mean():,.0f}")

    def _generate_poi(self, city_center, lat_spread, lon_spread):
        """Generate points of interest"""
        n_schools = 15
        n_parks = 10
        n_transit = 20

        poi_list = []

        # Schools
        for i in range(n_schools):
            lat = city_center[0] + np.random.uniform(-lat_spread, lat_spread)
            lon = city_center[1] + np.random.uniform(-lon_spread, lon_spread)
            poi_list.append({'type': 'school', 'latitude': lat, 'longitude': lon})

        # Parks
        for i in range(n_parks):
            lat = city_center[0] + np.random.uniform(-lat_spread, lat_spread)
            lon = city_center[1] + np.random.uniform(-lon_spread, lon_spread)
            poi_list.append({'type': 'park', 'latitude': lat, 'longitude': lon})

        # Transit stations
        for i in range(n_transit):
            lat = city_center[0] + np.random.uniform(-lat_spread, lat_spread)
            lon = city_center[1] + np.random.uniform(-lon_spread, lon_spread)
            poi_list.append({'type': 'transit', 'latitude': lat, 'longitude': lon})

        self.poi_data = pd.DataFrame(poi_list)

    def _add_poi_features(self):
        """Add distance to nearest POI features"""
        # Distance to nearest school
        school_dist = []
        park_dist = []
        transit_dist = []

        schools = self.poi_data[self.poi_data['type'] == 'school']
        parks = self.poi_data[self.poi_data['type'] == 'park']
        transit = self.poi_data[self.poi_data['type'] == 'transit']

        for _, prop in self.properties.iterrows():
            # Nearest school
            if len(schools) > 0:
                dists = self.haversine_distance(
                    prop['latitude'], prop['longitude'],
                    schools['latitude'].values, schools['longitude'].values
                )
                school_dist.append(dists.min())
            else:
                school_dist.append(999)

            # Nearest park
            if len(parks) > 0:
                dists = self.haversine_distance(
                    prop['latitude'], prop['longitude'],
                    parks['latitude'].values, parks['longitude'].values
                )
                park_dist.append(dists.min())
            else:
                park_dist.append(999)

            # Nearest transit
            if len(transit) > 0:
                dists = self.haversine_distance(
                    prop['latitude'], prop['longitude'],
                    transit['latitude'].values, transit['longitude'].values
                )
                transit_dist.append(dists.min())
            else:
                transit_dist.append(999)

        self.properties['dist_to_school_km'] = school_dist
        self.properties['dist_to_park_km'] = park_dist
        self.properties['dist_to_transit_km'] = transit_dist

    def train_price_model(self):
        """Train price prediction model"""
        print("\n" + "="*60)
        print("TRAINING PRICE PREDICTION MODEL")
        print("="*60)

        # Features for modeling
        feature_cols = ['bedrooms', 'bathrooms', 'sqft', 'lot_size_sqft',
                       'age_years', 'dist_to_center_km', 'dist_to_school_km',
                       'dist_to_park_km', 'dist_to_transit_km']

        X = self.properties[feature_cols]
        y = self.properties['price']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Random Forest
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        print(f"\nTrain MAE: ${train_mae:,.0f}")
        print(f"Test MAE: ${test_mae:,.0f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")

        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop Feature Importances:")
        for _, row in importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Add predictions to data
        self.properties['predicted_price'] = self.model.predict(X)
        self.properties['price_residual'] = self.properties['price'] - self.properties['predicted_price']

        return importance

    def identify_value_areas(self):
        """Identify undervalued and overvalued areas"""
        print("\n" + "="*60)
        print("VALUE AREA ANALYSIS")
        print("="*60)

        # Calculate price per sqft
        self.properties['price_per_sqft'] = self.properties['price'] / self.properties['sqft']

        # Find undervalued properties (actual < predicted)
        self.properties['value_score'] = (
            (self.properties['predicted_price'] - self.properties['price']) /
            self.properties['predicted_price'] * 100
        )

        undervalued = self.properties.nlargest(10, 'value_score')
        overvalued = self.properties.nsmallest(10, 'value_score')

        print("\nTop 5 Undervalued Properties:")
        for _, prop in undervalued.head(5).iterrows():
            print(f"  {prop['property_id']}: {prop['bedrooms']}BR, {prop['bathrooms']}BA")
            print(f"    Actual: ${prop['price']:,.0f}, Predicted: ${prop['predicted_price']:,.0f}")
            print(f"    Potential Value: {prop['value_score']:.1f}%")

        print("\nTop 5 Overvalued Properties:")
        for _, prop in overvalued.head(5).iterrows():
            print(f"  {prop['property_id']}: {prop['bedrooms']}BR, {prop['bathrooms']}BA")
            print(f"    Actual: ${prop['price']:,.0f}, Predicted: ${prop['predicted_price']:,.0f}")
            print(f"    Overvaluation: {-prop['value_score']:.1f}%")

    def visualize_results(self):
        """Create comprehensive real estate visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Price Heatmap
        ax = axes[0, 0]
        scatter = ax.scatter(
            self.properties['longitude'],
            self.properties['latitude'],
            c=self.properties['price'],
            s=50,
            cmap='RdYlGn',
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Price ($)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Property Price Heatmap', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. Price per SqFt
        ax = axes[0, 1]
        scatter = ax.scatter(
            self.properties['longitude'],
            self.properties['latitude'],
            c=self.properties['price_per_sqft'],
            s=50,
            cmap='viridis',
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Price per SqFt ($)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Price per Square Foot Map', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 3. Value Score Map
        ax = axes[0, 2]
        scatter = ax.scatter(
            self.properties['longitude'],
            self.properties['latitude'],
            c=self.properties['value_score'],
            s=50,
            cmap='RdYlGn',
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5,
            vmin=-20,
            vmax=20
        )
        plt.colorbar(scatter, ax=ax, label='Value Score (%)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Undervalued (Green) vs Overvalued (Red)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 4. Price vs Size
        ax = axes[1, 0]
        ax.scatter(self.properties['sqft'], self.properties['price'],
                  alpha=0.5, s=30, c='steelblue', edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Square Footage')
        ax.set_ylabel('Price ($)')
        ax.set_title('Price vs Property Size', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 5. Predicted vs Actual
        ax = axes[1, 1]
        ax.scatter(self.properties['predicted_price'], self.properties['price'],
                  alpha=0.5, s=30, c='coral', edgecolors='black', linewidths=0.5)
        min_val = min(self.properties['price'].min(), self.properties['predicted_price'].min())
        max_val = max(self.properties['price'].max(), self.properties['predicted_price'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_xlabel('Predicted Price ($)')
        ax.set_ylabel('Actual Price ($)')
        ax.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Price by Bedroom Count
        ax = axes[1, 2]
        bedroom_prices = self.properties.groupby('bedrooms')['price'].mean()
        ax.bar(bedroom_prices.index, bedroom_prices.values,
              color='#3498db', edgecolor='navy', width=0.6)
        ax.set_xlabel('Number of Bedrooms')
        ax.set_ylabel('Average Price ($)')
        ax.set_title('Average Price by Bedrooms', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('real_estate_price_mapping.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'real_estate_price_mapping.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("REAL ESTATE PRICE MAPPING - GEOSPATIAL ANALYSIS")
    print("="*60)

    # Initialize mapper
    mapper = RealEstatePriceMapper(city_name="San Francisco")

    # Generate data
    mapper.create_sample_data()

    # Train model
    feature_importance = mapper.train_price_model()

    # Identify value areas
    mapper.identify_value_areas()

    # Visualize results
    mapper.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
