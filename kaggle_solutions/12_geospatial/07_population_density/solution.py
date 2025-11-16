"""
Population Density Estimation - Geospatial Analysis
Estimate and map population density using spatial interpolation

Dataset: Synthetic census data
Difficulty: ⭐⭐ Intermediate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class PopulationDensityEstimator:
    """Population density estimation and mapping"""

    def __init__(self, city_name="Metro City"):
        self.city_name = city_name
        self.census_data = None
        self.density_grid = None

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
        """Generate synthetic population census data"""
        np.random.seed(42)

        # City bounds
        center_lat, center_lon = 41.8781, -87.6298  # Chicago-like coordinates
        lat_spread, lon_spread = 0.15, 0.20

        # Create population centers
        n_centers = 8
        center_lats = center_lat + np.random.uniform(-lat_spread*0.6, lat_spread*0.6, n_centers)
        center_lons = center_lon + np.random.uniform(-lon_spread*0.6, lon_spread*0.6, n_centers)
        center_densities = np.random.uniform(2000, 15000, n_centers)  # People per sq km

        # Generate census tracts
        n_tracts = 200
        census_tracts = []

        for i in range(n_tracts):
            # Assign to a center with probability based on distance
            center_idx = np.random.choice(n_centers, p=center_densities/center_densities.sum())

            # Tract location
            lat = center_lats[center_idx] + np.random.normal(0, 0.015)
            lon = center_lons[center_idx] + np.random.normal(0, 0.020)

            # Distance from city center
            dist_from_center = self.haversine_distance(lat, lon, center_lat, center_lon)

            # Base density decreases with distance
            base_density = center_densities[center_idx] * np.exp(-dist_from_center / 8.0)

            # Tract area
            area_sqkm = np.random.uniform(0.5, 3.0)

            # Population
            population = int(base_density * area_sqkm * np.random.uniform(0.7, 1.3))

            # Demographics
            median_age = np.random.normal(35, 10).clip(20, 70)
            median_income = np.random.lognormal(10.8, 0.5).clip(20000, 200000)

            # Housing units
            housing_units = int(population / np.random.uniform(2.2, 3.0))
            occupied_pct = np.random.uniform(85, 98)

            census_tracts.append({
                'tract_id': f'TRACT{i+1:04d}',
                'latitude': lat,
                'longitude': lon,
                'population': population,
                'area_sqkm': area_sqkm,
                'density_per_sqkm': population / area_sqkm,
                'median_age': median_age,
                'median_income': median_income,
                'housing_units': housing_units,
                'occupied_pct': occupied_pct,
                'dist_from_center_km': dist_from_center
            })

        self.census_data = pd.DataFrame(census_tracts)

        print(f"✓ Generated {len(self.census_data)} census tracts")
        print(f"✓ Total population: {self.census_data['population'].sum():,}")
        print(f"✓ Density range: {self.census_data['density_per_sqkm'].min():.0f} - {self.census_data['density_per_sqkm'].max():.0f} per sq km")
        print(f"✓ Mean density: {self.census_data['density_per_sqkm'].mean():.0f} per sq km")

    def analyze_density_patterns(self):
        """Analyze population density patterns"""
        print("\n" + "="*60)
        print("DENSITY PATTERN ANALYSIS")
        print("="*60)

        # Classify areas by density
        self.census_data['density_class'] = pd.cut(
            self.census_data['density_per_sqkm'],
            bins=[0, 2000, 5000, 10000, np.inf],
            labels=['Low', 'Medium', 'High', 'Very High']
        )

        density_stats = self.census_data.groupby('density_class').agg({
            'population': 'sum',
            'area_sqkm': 'sum',
            'tract_id': 'count'
        }).rename(columns={'tract_id': 'n_tracts'})

        print("\nDensity Distribution:")
        for density_class, stats in density_stats.iterrows():
            print(f"\n{density_class} Density:")
            print(f"  Tracts: {stats['n_tracts']}")
            print(f"  Population: {stats['population']:,}")
            print(f"  Area: {stats['area_sqkm']:.1f} sq km")
            print(f"  Density: {stats['population']/stats['area_sqkm']:.0f} per sq km")

        # Distance vs density analysis
        distance_bins = pd.cut(
            self.census_data['dist_from_center_km'],
            bins=[0, 5, 10, 20],
            labels=['Inner City', 'Urban', 'Suburban']
        )
        self.census_data['area_zone'] = distance_bins

        zone_stats = self.census_data.groupby('area_zone')['density_per_sqkm'].agg([
            'mean', 'median', 'std'
        ])

        print("\nDensity by Zone:")
        for zone, stats in zone_stats.iterrows():
            print(f"  {zone}: {stats['mean']:.0f} ± {stats['std']:.0f} per sq km")

        # Income vs density correlation
        correlation = self.census_data[['density_per_sqkm', 'median_income']].corr().iloc[0, 1]
        print(f"\nDensity-Income Correlation: {correlation:.3f}")

        return density_stats, zone_stats

    def create_density_grid(self, resolution=100):
        """Create interpolated density grid"""
        print("\n" + "="*60)
        print("CREATING DENSITY GRID")
        print("="*60)

        # Get bounds
        lat_min, lat_max = self.census_data['latitude'].min(), self.census_data['latitude'].max()
        lon_min, lon_max = self.census_data['longitude'].min(), self.census_data['longitude'].max()

        # Create grid
        grid_lat = np.linspace(lat_min, lat_max, resolution)
        grid_lon = np.linspace(lon_min, lon_max, resolution)
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

        # Interpolate density
        points = self.census_data[['longitude', 'latitude']].values
        values = self.census_data['density_per_sqkm'].values

        self.density_grid = griddata(
            points, values,
            (grid_lon_mesh, grid_lat_mesh),
            method='cubic',
            fill_value=0
        )

        # Ensure non-negative
        self.density_grid = np.maximum(self.density_grid, 0)

        print(f"✓ Created {resolution}x{resolution} density grid")
        print(f"✓ Grid density range: {self.density_grid.min():.0f} - {self.density_grid.max():.0f}")

        return grid_lat, grid_lon, self.density_grid

    def identify_density_clusters(self, n_clusters=5):
        """Identify population density clusters"""
        print("\n" + "="*60)
        print("DENSITY CLUSTERING")
        print("="*60)

        # Features for clustering
        features = self.census_data[['latitude', 'longitude', 'density_per_sqkm']].values

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.census_data['cluster'] = kmeans.fit_predict(features_scaled)

        # Cluster statistics
        cluster_stats = self.census_data.groupby('cluster').agg({
            'population': 'sum',
            'density_per_sqkm': 'mean',
            'latitude': 'mean',
            'longitude': 'mean',
            'tract_id': 'count'
        }).rename(columns={'tract_id': 'n_tracts'})

        cluster_stats = cluster_stats.sort_values('density_per_sqkm', ascending=False)

        print(f"\n✓ Identified {n_clusters} density clusters")
        print("\nCluster Summary:")
        for cluster_id, stats in cluster_stats.iterrows():
            print(f"\nCluster {cluster_id}:")
            print(f"  Tracts: {stats['n_tracts']}")
            print(f"  Population: {stats['population']:,}")
            print(f"  Avg Density: {stats['density_per_sqkm']:.0f} per sq km")
            print(f"  Center: ({stats['latitude']:.4f}, {stats['longitude']:.4f})")

        return cluster_stats

    def estimate_population(self, lat, lon, radius_km=1.0):
        """Estimate population within radius of a point"""
        distances = self.haversine_distance(
            lat, lon,
            self.census_data['latitude'].values,
            self.census_data['longitude'].values
        )

        # Find tracts within radius
        within_radius = self.census_data[distances <= radius_km]

        if len(within_radius) == 0:
            return 0

        total_population = within_radius['population'].sum()
        return total_population

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Population Density Map
        ax = axes[0, 0]
        if self.density_grid is not None:
            grid_lat, grid_lon, density_grid = self.create_density_grid(resolution=80)
            contour = ax.contourf(grid_lon, grid_lat, density_grid,
                                 levels=15, cmap='YlOrRd', alpha=0.7)
            plt.colorbar(contour, ax=ax, label='Density (people/sq km)')

        scatter = ax.scatter(
            self.census_data['longitude'],
            self.census_data['latitude'],
            c=self.census_data['density_per_sqkm'],
            s=self.census_data['population'] / 50,
            cmap='YlOrRd',
            alpha=0.6,
            edgecolors='black',
            linewidths=0.5
        )
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Population Density Map', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. Density by Distance
        ax = axes[0, 1]
        ax.scatter(self.census_data['dist_from_center_km'],
                  self.census_data['density_per_sqkm'],
                  alpha=0.5, s=50, c='steelblue', edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Distance from Center (km)')
        ax.set_ylabel('Density (people/sq km)')
        ax.set_title('Density vs Distance from Center', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 3. Density Distribution
        ax = axes[0, 2]
        ax.hist(self.census_data['density_per_sqkm'], bins=30,
               color='coral', edgecolor='darkred', alpha=0.7)
        ax.axvline(self.census_data['density_per_sqkm'].mean(),
                  color='blue', linestyle='--', linewidth=2,
                  label=f'Mean: {self.census_data["density_per_sqkm"].mean():.0f}')
        ax.set_xlabel('Density (people/sq km)')
        ax.set_ylabel('Frequency')
        ax.set_title('Density Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 4. Population by Density Class
        ax = axes[1, 0]
        density_pop = self.census_data.groupby('density_class')['population'].sum()
        colors_density = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e67e22', 'Very High': '#e74c3c'}
        bars = ax.bar(range(len(density_pop)), density_pop.values,
                     color=[colors_density[dc] for dc in density_pop.index],
                     edgecolor='black')
        ax.set_xticks(range(len(density_pop)))
        ax.set_xticklabels(density_pop.index, rotation=45, ha='right')
        ax.set_ylabel('Total Population')
        ax.set_title('Population by Density Class', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 5. Cluster Map
        ax = axes[1, 1]
        if 'cluster' in self.census_data.columns:
            scatter = ax.scatter(
                self.census_data['longitude'],
                self.census_data['latitude'],
                c=self.census_data['cluster'],
                s=80,
                cmap='tab10',
                alpha=0.6,
                edgecolors='black',
                linewidths=1
            )
            plt.colorbar(scatter, ax=ax, label='Cluster ID')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Density Clusters', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 6. Income vs Density
        ax = axes[1, 2]
        ax.scatter(self.census_data['density_per_sqkm'],
                  self.census_data['median_income'],
                  alpha=0.4, s=50, c='purple', edgecolors='black', linewidths=0.5)
        ax.set_xlabel('Density (people/sq km)')
        ax.set_ylabel('Median Income ($)')
        ax.set_title('Income vs Density', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('population_density_estimation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'population_density_estimation.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("POPULATION DENSITY ESTIMATION - GEOSPATIAL ANALYSIS")
    print("="*60)

    # Initialize estimator
    estimator = PopulationDensityEstimator(city_name="Chicago")

    # Generate data
    estimator.create_sample_data()

    # Analyze patterns
    density_stats, zone_stats = estimator.analyze_density_patterns()

    # Create density grid
    grid_lat, grid_lon, density_grid = estimator.create_density_grid(resolution=100)

    # Identify clusters
    cluster_stats = estimator.identify_density_clusters(n_clusters=5)

    # Example population estimation
    test_lat, test_lon = 41.8781, -87.6298
    pop_1km = estimator.estimate_population(test_lat, test_lon, radius_km=1.0)
    print(f"\nPopulation within 1km of ({test_lat}, {test_lon}): {pop_1km:,}")

    # Visualize
    estimator.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
