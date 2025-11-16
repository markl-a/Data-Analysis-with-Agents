"""
Urban Heat Island Analysis - Geospatial Analysis
Analyze urban heat island effects and temperature patterns

Dataset: Synthetic temperature measurement data
Difficulty: ⭐⭐ Intermediate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')


class UrbanHeatAnalyzer:
    """Urban heat island effect analysis"""

    def __init__(self, city_name="Metro City"):
        self.city_name = city_name
        self.temperature_data = None
        self.land_use_data = None

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
        """Generate synthetic temperature and land use data"""
        np.random.seed(42)

        # City bounds
        center_lat, center_lon = 33.7490, -84.3880  # Atlanta-like coordinates
        lat_spread, lon_spread = 0.12, 0.16

        # Create temperature measurement points
        n_points = 300

        # Define urban core (hotter) and suburban/rural areas (cooler)
        measurements = []

        # Base temperature
        base_temp = 28.0  # Celsius

        for i in range(n_points):
            lat = center_lat + np.random.uniform(-lat_spread, lat_spread)
            lon = center_lon + np.random.uniform(-lon_spread, lon_spread)

            # Distance from city center (urban heat effect)
            dist_from_center = self.haversine_distance(lat, lon, center_lat, center_lon)

            # Temperature decreases with distance from center (urban heat island)
            urban_heat_effect = 6.0 * np.exp(-dist_from_center / 4.0)  # Up to 6°C difference

            # Land use type
            if dist_from_center < 3:
                land_use = np.random.choice(['Commercial', 'Residential', 'Industrial'],
                                           p=[0.4, 0.5, 0.1])
                vegetation_pct = np.random.uniform(5, 25)  # Low vegetation
            elif dist_from_center < 7:
                land_use = np.random.choice(['Residential', 'Park', 'Commercial'],
                                           p=[0.6, 0.2, 0.2])
                vegetation_pct = np.random.uniform(20, 50)  # Medium vegetation
            else:
                land_use = np.random.choice(['Residential', 'Park', 'Rural'],
                                           p=[0.4, 0.3, 0.3])
                vegetation_pct = np.random.uniform(40, 80)  # High vegetation

            # Vegetation cooling effect
            vegetation_effect = -(vegetation_pct / 100) * 3.0  # Up to -3°C from vegetation

            # Land use specific effects
            land_use_effects = {
                'Commercial': 2.0,    # Lots of concrete/asphalt
                'Industrial': 3.0,    # Heat from operations
                'Residential': 0.5,   # Mixed surfaces
                'Park': -2.0,         # Cooling from green space
                'Rural': -1.5         # Natural cooling
            }

            # Final temperature
            temperature = (base_temp + urban_heat_effect + vegetation_effect +
                          land_use_effects[land_use] + np.random.normal(0, 0.5))

            # Time of measurement
            hour = np.random.choice([6, 9, 12, 15, 18, 21])
            if hour in [12, 15]:  # Afternoon measurements higher
                temperature += np.random.uniform(1, 3)
            elif hour in [6, 21]:  # Early morning/night lower
                temperature -= np.random.uniform(1, 2)

            measurements.append({
                'point_id': f'TEMP{i+1:03d}',
                'latitude': lat,
                'longitude': lon,
                'temperature_c': temperature,
                'dist_from_center_km': dist_from_center,
                'land_use': land_use,
                'vegetation_pct': vegetation_pct,
                'hour': hour
            })

        self.temperature_data = pd.DataFrame(measurements)

        print(f"✓ Generated {len(self.temperature_data)} temperature measurements")
        print(f"✓ Temperature range: {self.temperature_data['temperature_c'].min():.1f}°C - {self.temperature_data['temperature_c'].max():.1f}°C")
        print(f"✓ Mean temperature: {self.temperature_data['temperature_c'].mean():.1f}°C")
        print(f"✓ Urban heat island intensity: {self.temperature_data['temperature_c'].max() - self.temperature_data['temperature_c'].min():.1f}°C")

    def analyze_heat_island_effect(self):
        """Analyze urban heat island patterns"""
        print("\n" + "="*60)
        print("URBAN HEAT ISLAND ANALYSIS")
        print("="*60)

        # Temperature by distance from center
        distance_bins = pd.cut(self.temperature_data['dist_from_center_km'],
                              bins=[0, 3, 7, 15], labels=['Urban Core', 'Suburban', 'Rural'])
        self.temperature_data['area_type'] = distance_bins

        area_stats = self.temperature_data.groupby('area_type')['temperature_c'].agg([
            'mean', 'std', 'min', 'max'
        ])

        print("\nTemperature by Area Type:")
        for area, stats in area_stats.iterrows():
            print(f"\n{area}:")
            print(f"  Mean: {stats['mean']:.2f}°C")
            print(f"  Std Dev: {stats['std']:.2f}°C")
            print(f"  Range: {stats['min']:.2f}°C - {stats['max']:.2f}°C")

        # Heat island intensity
        urban_mean = area_stats.loc['Urban Core', 'mean']
        rural_mean = area_stats.loc['Rural', 'mean']
        heat_island_intensity = urban_mean - rural_mean

        print(f"\nUrban Heat Island Intensity: {heat_island_intensity:.2f}°C")
        print(f"  (Urban Core - Rural temperature difference)")

        # Temperature by land use
        land_use_stats = self.temperature_data.groupby('land_use')['temperature_c'].agg([
            'mean', 'count'
        ]).sort_values('mean', ascending=False)

        print("\nTemperature by Land Use:")
        for land_use, stats in land_use_stats.iterrows():
            print(f"  {land_use}: {stats['mean']:.2f}°C (n={int(stats['count'])})")

        # Vegetation effect
        correlation = self.temperature_data[['vegetation_pct', 'temperature_c']].corr().iloc[0, 1]
        print(f"\nVegetation-Temperature Correlation: {correlation:.3f}")

        return area_stats, land_use_stats

    def create_heat_map(self, resolution=100):
        """Create interpolated heat map"""
        print("\n" + "="*60)
        print("CREATING HEAT MAP")
        print("="*60)

        # Get bounds
        lat_min, lat_max = self.temperature_data['latitude'].min(), self.temperature_data['latitude'].max()
        lon_min, lon_max = self.temperature_data['longitude'].min(), self.temperature_data['longitude'].max()

        # Create grid
        grid_lat = np.linspace(lat_min, lat_max, resolution)
        grid_lon = np.linspace(lon_min, lon_max, resolution)
        grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

        # Interpolate temperature
        points = self.temperature_data[['longitude', 'latitude']].values
        values = self.temperature_data['temperature_c'].values

        grid_temp = griddata(points, values, (grid_lon_mesh, grid_lat_mesh), method='cubic')

        # Apply smoothing
        grid_temp_smooth = gaussian_filter(grid_temp, sigma=2)

        print(f"✓ Created {resolution}x{resolution} heat map grid")

        return grid_lat, grid_lon, grid_temp_smooth

    def identify_hot_spots(self, threshold_percentile=90):
        """Identify extreme heat locations"""
        print("\n" + "="*60)
        print("HOT SPOT IDENTIFICATION")
        print("="*60)

        threshold = self.temperature_data['temperature_c'].quantile(threshold_percentile / 100)

        hot_spots = self.temperature_data[
            self.temperature_data['temperature_c'] >= threshold
        ].copy()

        hot_spots = hot_spots.sort_values('temperature_c', ascending=False)

        print(f"\nThreshold (P{threshold_percentile}): {threshold:.2f}°C")
        print(f"Number of hot spots: {len(hot_spots)}")

        print("\nTop 5 Hottest Locations:")
        for _, spot in hot_spots.head(5).iterrows():
            print(f"\n{spot['point_id']}:")
            print(f"  Temperature: {spot['temperature_c']:.2f}°C")
            print(f"  Location: ({spot['latitude']:.4f}, {spot['longitude']:.4f})")
            print(f"  Land Use: {spot['land_use']}")
            print(f"  Vegetation: {spot['vegetation_pct']:.1f}%")

        return hot_spots

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Temperature Distribution Map
        ax1 = fig.add_subplot(gs[0:2, 0:2])

        # Create heat map
        grid_lat, grid_lon, grid_temp = self.create_heat_map(resolution=80)

        # Plot heat map
        contour = ax1.contourf(grid_lon, grid_lat, grid_temp,
                              levels=15, cmap='RdYlBu_r', alpha=0.7)
        plt.colorbar(contour, ax=ax1, label='Temperature (°C)')

        # Overlay measurement points
        scatter = ax1.scatter(
            self.temperature_data['longitude'],
            self.temperature_data['latitude'],
            c=self.temperature_data['temperature_c'],
            s=30,
            cmap='RdYlBu_r',
            edgecolors='black',
            linewidths=0.5,
            alpha=0.8
        )

        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Urban Heat Island Map', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Temperature by Distance
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.scatter(self.temperature_data['dist_from_center_km'],
                   self.temperature_data['temperature_c'],
                   alpha=0.5, s=30, c='coral', edgecolors='black', linewidths=0.5)
        ax2.set_xlabel('Distance from Center (km)')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Temperature vs Distance', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Temperature by Land Use
        ax3 = fig.add_subplot(gs[1, 2])
        land_use_data = self.temperature_data.groupby('land_use')['temperature_c'].mean().sort_values()
        colors_lu = {'Industrial': '#e74c3c', 'Commercial': '#e67e22', 'Residential': '#f39c12',
                     'Park': '#2ecc71', 'Rural': '#27ae60'}
        bars = ax3.barh(range(len(land_use_data)), land_use_data.values,
                       color=[colors_lu.get(lu, 'gray') for lu in land_use_data.index],
                       edgecolor='black')
        ax3.set_yticks(range(len(land_use_data)))
        ax3.set_yticklabels(land_use_data.index, fontsize=9)
        ax3.set_xlabel('Avg Temperature (°C)')
        ax3.set_title('Temperature by Land Use', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. Vegetation Effect
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.scatter(self.temperature_data['vegetation_pct'],
                   self.temperature_data['temperature_c'],
                   alpha=0.4, s=30, c='green', edgecolors='darkgreen', linewidths=0.5)

        # Add trend line
        z = np.polyfit(self.temperature_data['vegetation_pct'],
                      self.temperature_data['temperature_c'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(self.temperature_data['vegetation_pct'].min(),
                             self.temperature_data['vegetation_pct'].max(), 100)
        ax4.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.1f}')

        ax4.set_xlabel('Vegetation Coverage (%)')
        ax4.set_ylabel('Temperature (°C)')
        ax4.set_title('Vegetation Cooling Effect', fontsize=11, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Area Type Comparison
        ax5 = fig.add_subplot(gs[2, 1])
        area_means = self.temperature_data.groupby('area_type')['temperature_c'].mean()
        area_order = ['Urban Core', 'Suburban', 'Rural']
        area_means = area_means.reindex(area_order)
        colors_area = ['#e74c3c', '#f39c12', '#2ecc71']

        bars = ax5.bar(range(len(area_means)), area_means.values,
                      color=colors_area, edgecolor='black')
        ax5.set_xticks(range(len(area_means)))
        ax5.set_xticklabels(area_means.index, rotation=45, ha='right')
        ax5.set_ylabel('Avg Temperature (°C)')
        ax5.set_title('Temperature by Area Type', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}°C',
                    ha='center', va='bottom', fontweight='bold')

        # 6. Temperature Distribution
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.hist(self.temperature_data['temperature_c'], bins=30,
                color='steelblue', edgecolor='navy', alpha=0.7)
        ax6.axvline(self.temperature_data['temperature_c'].mean(),
                   color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {self.temperature_data["temperature_c"].mean():.1f}°C')
        ax6.set_xlabel('Temperature (°C)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Temperature Distribution', fontsize=11, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

        plt.savefig('urban_heat_island_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'urban_heat_island_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("URBAN HEAT ISLAND ANALYSIS - GEOSPATIAL ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = UrbanHeatAnalyzer(city_name="Atlanta")

    # Generate data
    analyzer.create_sample_data()

    # Analyze heat island effect
    area_stats, land_use_stats = analyzer.analyze_heat_island_effect()

    # Identify hot spots
    hot_spots = analyzer.identify_hot_spots(threshold_percentile=90)

    # Visualize
    analyzer.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
