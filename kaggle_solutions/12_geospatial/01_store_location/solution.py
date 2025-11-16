"""
Store Location Optimization - Geospatial Analysis
Optimize retail store locations based on population density and competitor analysis

Dataset: Synthetic geospatial data
Difficulty: ⭐⭐ Intermediate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class StoreLocationOptimizer:
    """Store location optimization using geospatial analysis"""

    def __init__(self, city_name="Metropolitan Area"):
        self.city_name = city_name
        self.population_data = None
        self.competitor_locations = None
        self.optimal_locations = None

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance between coordinates in km"""
        R = 6371  # Earth radius in kilometers

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def create_sample_data(self):
        """Generate synthetic geospatial data"""
        np.random.seed(42)

        # City bounds (roughly 40km x 40km area)
        center_lat, center_lon = 40.7128, -74.0060  # NYC-like coordinates
        lat_spread, lon_spread = 0.18, 0.24

        # Generate population density points (residential areas)
        n_population_points = 500

        # Create clustered population centers
        n_clusters = 8
        cluster_centers_lat = center_lat + np.random.uniform(-lat_spread, lat_spread, n_clusters)
        cluster_centers_lon = center_lon + np.random.uniform(-lon_spread, lon_spread, n_clusters)

        population_lats = []
        population_lons = []
        population_density = []

        for i in range(n_population_points):
            cluster = np.random.randint(0, n_clusters)
            lat = cluster_centers_lat[cluster] + np.random.normal(0, 0.02)
            lon = cluster_centers_lon[cluster] + np.random.normal(0, 0.03)
            density = np.random.lognormal(8, 1.5)  # People per square km

            population_lats.append(lat)
            population_lons.append(lon)
            population_density.append(density)

        self.population_data = pd.DataFrame({
            'latitude': population_lats,
            'longitude': population_lons,
            'population_density': population_density,
            'area_sqkm': np.random.uniform(0.5, 2.0, n_population_points)
        })

        self.population_data['total_population'] = (
            self.population_data['population_density'] *
            self.population_data['area_sqkm']
        )

        # Generate competitor store locations
        n_competitors = 25
        competitor_lats = center_lat + np.random.uniform(-lat_spread*0.8, lat_spread*0.8, n_competitors)
        competitor_lons = center_lon + np.random.uniform(-lon_spread*0.8, lon_spread*0.8, n_competitors)

        self.competitor_locations = pd.DataFrame({
            'latitude': competitor_lats,
            'longitude': competitor_lons,
            'competitor_name': [f'Competitor_{i+1}' for i in range(n_competitors)],
            'market_share': np.random.uniform(0.02, 0.15, n_competitors)
        })

        print(f"✓ Generated {len(self.population_data)} population points")
        print(f"✓ Generated {len(self.competitor_locations)} competitor locations")
        print(f"✓ Total population in area: {self.population_data['total_population'].sum():,.0f}")

    def calculate_accessibility_score(self, lat, lon):
        """Calculate accessibility score for a location"""
        # Distance to all population centers
        distances = self.haversine_distance(
            lat, lon,
            self.population_data['latitude'].values,
            self.population_data['longitude'].values
        )

        # Weight by population and inverse distance
        weights = self.population_data['total_population'].values / (distances + 0.5)
        accessibility = weights.sum()

        return accessibility

    def calculate_competition_score(self, lat, lon):
        """Calculate competition intensity for a location"""
        if len(self.competitor_locations) == 0:
            return 0

        distances = self.haversine_distance(
            lat, lon,
            self.competitor_locations['latitude'].values,
            self.competitor_locations['longitude'].values
        )

        # Competition decreases with distance
        competition = (self.competitor_locations['market_share'].values /
                      (distances + 0.5)).sum()

        return competition

    def find_optimal_locations(self, n_stores=5, grid_resolution=50):
        """Find optimal store locations using grid search"""
        print("\n" + "="*60)
        print("FINDING OPTIMAL STORE LOCATIONS")
        print("="*60)

        # Create search grid
        lat_min = self.population_data['latitude'].min()
        lat_max = self.population_data['latitude'].max()
        lon_min = self.population_data['longitude'].min()
        lon_max = self.population_data['longitude'].max()

        lats = np.linspace(lat_min, lat_max, grid_resolution)
        lons = np.linspace(lon_min, lon_max, grid_resolution)

        # Evaluate each grid point
        candidate_locations = []

        for lat in lats:
            for lon in lons:
                accessibility = self.calculate_accessibility_score(lat, lon)
                competition = self.calculate_competition_score(lat, lon)

                # Score = high accessibility, low competition
                score = accessibility - (competition * 500)

                candidate_locations.append({
                    'latitude': lat,
                    'longitude': lon,
                    'accessibility_score': accessibility,
                    'competition_score': competition,
                    'total_score': score
                })

        candidates_df = pd.DataFrame(candidate_locations)

        # Select top locations ensuring minimum distance between stores
        optimal_locs = []
        min_distance_km = 2.0  # Minimum 2km between stores

        remaining = candidates_df.sort_values('total_score', ascending=False)

        for i in range(n_stores):
            if len(remaining) == 0:
                break

            best = remaining.iloc[0]
            optimal_locs.append(best.to_dict())

            # Remove nearby locations
            distances = self.haversine_distance(
                best['latitude'], best['longitude'],
                remaining['latitude'].values,
                remaining['longitude'].values
            )
            remaining = remaining[distances > min_distance_km]

        self.optimal_locations = pd.DataFrame(optimal_locs)

        print(f"\n✓ Found {len(self.optimal_locations)} optimal locations")
        for i, row in self.optimal_locations.iterrows():
            print(f"\nLocation {i+1}:")
            print(f"  Coordinates: ({row['latitude']:.4f}, {row['longitude']:.4f})")
            print(f"  Accessibility Score: {row['accessibility_score']:.2f}")
            print(f"  Competition Score: {row['competition_score']:.2f}")
            print(f"  Total Score: {row['total_score']:.2f}")

        return self.optimal_locations

    def calculate_coverage_metrics(self):
        """Calculate coverage metrics for optimal locations"""
        print("\n" + "="*60)
        print("COVERAGE ANALYSIS")
        print("="*60)

        # For each population point, find nearest store
        distances_to_stores = []

        for _, pop_point in self.population_data.iterrows():
            distances = self.haversine_distance(
                pop_point['latitude'], pop_point['longitude'],
                self.optimal_locations['latitude'].values,
                self.optimal_locations['longitude'].values
            )
            min_distance = distances.min()
            distances_to_stores.append(min_distance)

        distances_to_stores = np.array(distances_to_stores)

        # Calculate metrics
        pop_within_1km = self.population_data[distances_to_stores <= 1]['total_population'].sum()
        pop_within_3km = self.population_data[distances_to_stores <= 3]['total_population'].sum()
        pop_within_5km = self.population_data[distances_to_stores <= 5]['total_population'].sum()
        total_pop = self.population_data['total_population'].sum()

        print(f"\nPopulation Coverage:")
        print(f"  Within 1km: {pop_within_1km:,.0f} ({100*pop_within_1km/total_pop:.1f}%)")
        print(f"  Within 3km: {pop_within_3km:,.0f} ({100*pop_within_3km/total_pop:.1f}%)")
        print(f"  Within 5km: {pop_within_5km:,.0f} ({100*pop_within_5km/total_pop:.1f}%)")

        print(f"\nAverage Distance to Nearest Store: {distances_to_stores.mean():.2f} km")
        print(f"Maximum Distance to Nearest Store: {distances_to_stores.max():.2f} km")

        return {
            'coverage_1km': pop_within_1km / total_pop,
            'coverage_3km': pop_within_3km / total_pop,
            'coverage_5km': pop_within_5km / total_pop,
            'avg_distance': distances_to_stores.mean(),
            'max_distance': distances_to_stores.max()
        }

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Population Density Map
        ax = axes[0, 0]
        scatter = ax.scatter(
            self.population_data['longitude'],
            self.population_data['latitude'],
            c=self.population_data['population_density'],
            s=self.population_data['total_population'] / 100,
            alpha=0.6,
            cmap='YlOrRd',
            edgecolors='black',
            linewidths=0.5
        )
        plt.colorbar(scatter, ax=ax, label='Population Density')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Population Density Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. Optimal Locations with Competitors
        ax = axes[0, 1]
        ax.scatter(
            self.population_data['longitude'],
            self.population_data['latitude'],
            c='lightgray',
            s=20,
            alpha=0.3,
            label='Population Centers'
        )
        ax.scatter(
            self.competitor_locations['longitude'],
            self.competitor_locations['latitude'],
            c='red',
            s=100,
            marker='s',
            alpha=0.6,
            edgecolors='darkred',
            linewidths=1,
            label='Competitors'
        )
        ax.scatter(
            self.optimal_locations['longitude'],
            self.optimal_locations['latitude'],
            c='green',
            s=300,
            marker='*',
            edgecolors='darkgreen',
            linewidths=2,
            label='Optimal Locations',
            zorder=5
        )
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Optimal Store Locations vs Competitors', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Score Distribution
        ax = axes[1, 0]
        ax.bar(
            range(len(self.optimal_locations)),
            self.optimal_locations['total_score'],
            color='steelblue',
            edgecolor='navy'
        )
        ax.set_xlabel('Location Rank')
        ax.set_ylabel('Total Score')
        ax.set_title('Location Quality Scores', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Coverage Analysis
        ax = axes[1, 1]
        coverage_metrics = self.calculate_coverage_metrics()

        distances = [1, 3, 5]
        coverage = [
            coverage_metrics['coverage_1km'] * 100,
            coverage_metrics['coverage_3km'] * 100,
            coverage_metrics['coverage_5km'] * 100
        ]

        bars = ax.bar(distances, coverage, color=['#2ecc71', '#3498db', '#9b59b6'],
                     edgecolor='black', width=0.8)
        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Population Coverage (%)')
        ax.set_title('Population Coverage by Distance', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('store_location_optimization.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'store_location_optimization.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("STORE LOCATION OPTIMIZATION - GEOSPATIAL ANALYSIS")
    print("="*60)

    # Initialize optimizer
    optimizer = StoreLocationOptimizer(city_name="Metro City")

    # Generate data
    optimizer.create_sample_data()

    # Find optimal locations
    optimal_locations = optimizer.find_optimal_locations(n_stores=5, grid_resolution=40)

    # Calculate coverage
    coverage_metrics = optimizer.calculate_coverage_metrics()

    # Visualize results
    optimizer.visualize_results()

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
