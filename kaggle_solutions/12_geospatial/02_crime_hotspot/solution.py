"""
Crime Hotspot Mapping - Geospatial Analysis
Identify crime hotspots and patterns using spatial clustering

Dataset: Synthetic crime incident data
Difficulty: ⭐⭐ Intermediate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from scipy.stats import gaussian_kde
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CrimeHotspotAnalyzer:
    """Crime hotspot detection and analysis using geospatial methods"""

    def __init__(self, city_name="Metro City"):
        self.city_name = city_name
        self.crime_data = None
        self.hotspots = None

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
        """Generate synthetic crime incident data"""
        np.random.seed(42)

        # City bounds
        center_lat, center_lon = 34.0522, -118.2437  # LA-like coordinates
        lat_spread, lon_spread = 0.15, 0.20

        # Define crime hotspot centers
        n_hotspots = 6
        hotspot_lats = center_lat + np.random.uniform(-lat_spread*0.7, lat_spread*0.7, n_hotspots)
        hotspot_lons = center_lon + np.random.uniform(-lon_spread*0.7, lon_spread*0.7, n_hotspots)
        hotspot_intensity = np.random.uniform(0.3, 1.0, n_hotspots)

        # Generate crime incidents
        n_incidents = 2000
        crime_types = ['Theft', 'Assault', 'Burglary', 'Robbery', 'Vandalism',
                       'Drug Offense', 'Auto Theft', 'Fraud']

        incidents = []
        start_date = datetime(2024, 1, 1)

        for i in range(n_incidents):
            # Select hotspot (biased distribution)
            hotspot_idx = np.random.choice(n_hotspots, p=hotspot_intensity/hotspot_intensity.sum())

            # Generate location near hotspot
            lat = hotspot_lats[hotspot_idx] + np.random.normal(0, 0.01)
            lon = hotspot_lons[hotspot_idx] + np.random.normal(0, 0.015)

            # Add some random crimes outside hotspots
            if np.random.random() < 0.15:
                lat = center_lat + np.random.uniform(-lat_spread, lat_spread)
                lon = center_lon + np.random.uniform(-lon_spread, lon_spread)

            # Generate timestamp (more crimes at night and weekends)
            days_offset = np.random.randint(0, 365)
            hour = np.random.choice(24, p=self._get_hourly_distribution())
            timestamp = start_date + timedelta(days=days_offset, hours=hour)

            # Select crime type
            crime_type = np.random.choice(crime_types)
            severity = np.random.choice(['Low', 'Medium', 'High'], p=[0.5, 0.35, 0.15])

            incidents.append({
                'incident_id': f'INC{i+1:05d}',
                'latitude': lat,
                'longitude': lon,
                'crime_type': crime_type,
                'severity': severity,
                'timestamp': timestamp,
                'day_of_week': timestamp.strftime('%A'),
                'hour': timestamp.hour,
                'month': timestamp.month
            })

        self.crime_data = pd.DataFrame(incidents)

        # Add derived features
        self.crime_data['is_night'] = self.crime_data['hour'].apply(lambda x: 1 if x >= 20 or x < 6 else 0)
        self.crime_data['is_weekend'] = self.crime_data['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

        print(f"✓ Generated {len(self.crime_data)} crime incidents")
        print(f"✓ Date range: {self.crime_data['timestamp'].min()} to {self.crime_data['timestamp'].max()}")
        print(f"✓ Crime types: {self.crime_data['crime_type'].nunique()}")

    def _get_hourly_distribution(self):
        """Get realistic hourly crime distribution"""
        # More crimes during evening/night
        hourly_weights = np.array([
            0.02, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5 AM
            0.03, 0.04, 0.04, 0.03, 0.03, 0.04,  # 6-11 AM
            0.05, 0.05, 0.04, 0.05, 0.06, 0.07,  # 12-5 PM
            0.08, 0.09, 0.10, 0.08, 0.06, 0.03   # 6-11 PM
        ])
        return hourly_weights / hourly_weights.sum()

    def detect_hotspots_dbscan(self, eps_km=0.5, min_samples=20):
        """Detect crime hotspots using DBSCAN clustering"""
        print("\n" + "="*60)
        print("HOTSPOT DETECTION - DBSCAN CLUSTERING")
        print("="*60)

        # Prepare coordinates
        coords = self.crime_data[['latitude', 'longitude']].values

        # Convert eps from km to degrees (approximate)
        eps_degrees = eps_km / 111.0  # 1 degree ≈ 111 km

        # Apply DBSCAN
        clustering = DBSCAN(eps=eps_degrees, min_samples=min_samples, metric='euclidean')
        self.crime_data['cluster'] = clustering.fit_predict(coords)

        # Analyze clusters
        n_clusters = len(set(self.crime_data['cluster'])) - (1 if -1 in self.crime_data['cluster'] else 0)
        n_noise = (self.crime_data['cluster'] == -1).sum()

        print(f"\n✓ Detected {n_clusters} crime hotspots")
        print(f"✓ Noise points (isolated crimes): {n_noise}")

        # Calculate hotspot statistics
        hotspot_stats = []

        for cluster_id in range(n_clusters):
            cluster_data = self.crime_data[self.crime_data['cluster'] == cluster_id]

            hotspot_stats.append({
                'hotspot_id': cluster_id,
                'n_incidents': len(cluster_data),
                'center_lat': cluster_data['latitude'].mean(),
                'center_lon': cluster_data['longitude'].mean(),
                'severity_high_pct': (cluster_data['severity'] == 'High').sum() / len(cluster_data) * 100,
                'night_crime_pct': cluster_data['is_night'].mean() * 100,
                'weekend_crime_pct': cluster_data['is_weekend'].mean() * 100,
                'most_common_crime': cluster_data['crime_type'].mode()[0]
            })

        self.hotspots = pd.DataFrame(hotspot_stats).sort_values('n_incidents', ascending=False)

        print("\nTop Hotspots:")
        for _, hotspot in self.hotspots.head(5).iterrows():
            print(f"\nHotspot {hotspot['hotspot_id']}:")
            print(f"  Location: ({hotspot['center_lat']:.4f}, {hotspot['center_lon']:.4f})")
            print(f"  Incidents: {hotspot['n_incidents']}")
            print(f"  Most Common: {hotspot['most_common_crime']}")
            print(f"  High Severity: {hotspot['severity_high_pct']:.1f}%")
            print(f"  Night Crimes: {hotspot['night_crime_pct']:.1f}%")

        return self.hotspots

    def temporal_analysis(self):
        """Analyze temporal patterns in crime data"""
        print("\n" + "="*60)
        print("TEMPORAL PATTERN ANALYSIS")
        print("="*60)

        # Hourly patterns
        hourly_crimes = self.crime_data.groupby('hour').size()
        peak_hour = hourly_crimes.idxmax()

        # Daily patterns
        daily_crimes = self.crime_data.groupby('day_of_week').size()

        # Monthly patterns
        monthly_crimes = self.crime_data.groupby('month').size()

        print(f"\nPeak Crime Hour: {peak_hour}:00 ({hourly_crimes[peak_hour]} incidents)")
        print(f"Night Crimes: {self.crime_data['is_night'].sum()} ({100*self.crime_data['is_night'].mean():.1f}%)")
        print(f"Weekend Crimes: {self.crime_data['is_weekend'].sum()} ({100*self.crime_data['is_weekend'].mean():.1f}%)")

        print("\nCrimes by Type:")
        for crime_type, count in self.crime_data['crime_type'].value_counts().head(5).items():
            print(f"  {crime_type}: {count} ({100*count/len(self.crime_data):.1f}%)")

    def calculate_risk_scores(self, grid_resolution=50):
        """Calculate risk scores for geographic grid"""
        print("\n" + "="*60)
        print("RISK SCORE CALCULATION")
        print("="*60)

        lat_min, lat_max = self.crime_data['latitude'].min(), self.crime_data['latitude'].max()
        lon_min, lon_max = self.crime_data['longitude'].min(), self.crime_data['longitude'].max()

        # Create grid
        lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
        lon_grid = np.linspace(lon_min, lon_max, grid_resolution)

        risk_grid = np.zeros((grid_resolution, grid_resolution))

        # Calculate risk for each grid cell
        for i, lat in enumerate(lat_grid):
            for j, lon in enumerate(lon_grid):
                # Count crimes within radius
                distances = self.haversine_distance(
                    lat, lon,
                    self.crime_data['latitude'].values,
                    self.crime_data['longitude'].values
                )

                # Weighted by proximity and severity
                weights = np.where(self.crime_data['severity'] == 'High', 3,
                                 np.where(self.crime_data['severity'] == 'Medium', 2, 1))
                risk_grid[i, j] = (weights / (distances + 0.1)).sum()

        print(f"✓ Calculated risk scores for {grid_resolution}x{grid_resolution} grid")
        print(f"✓ Max risk score: {risk_grid.max():.2f}")
        print(f"✓ Mean risk score: {risk_grid.mean():.2f}")

        return lat_grid, lon_grid, risk_grid

    def visualize_results(self):
        """Create comprehensive crime analysis visualizations"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Crime Density Map with Hotspots
        ax1 = fig.add_subplot(gs[0:2, 0:2])

        # Plot all crimes
        ax1.scatter(self.crime_data['longitude'], self.crime_data['latitude'],
                   c='lightgray', s=10, alpha=0.3, label='All Crimes')

        # Plot hotspot crimes
        if 'cluster' in self.crime_data.columns:
            clustered = self.crime_data[self.crime_data['cluster'] != -1]
            scatter = ax1.scatter(clustered['longitude'], clustered['latitude'],
                                c=clustered['cluster'], s=30, alpha=0.6,
                                cmap='tab10', edgecolors='black', linewidths=0.5)
            plt.colorbar(scatter, ax=ax1, label='Hotspot ID')

        # Mark hotspot centers
        if self.hotspots is not None:
            ax1.scatter(self.hotspots['center_lon'], self.hotspots['center_lat'],
                       c='red', s=500, marker='*', edgecolors='darkred',
                       linewidths=2, label='Hotspot Centers', zorder=5)

        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Crime Hotspot Map', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Hourly Crime Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        hourly = self.crime_data.groupby('hour').size()
        ax2.plot(hourly.index, hourly.values, marker='o', linewidth=2, color='steelblue')
        ax2.fill_between(hourly.index, hourly.values, alpha=0.3)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Number of Crimes')
        ax2.set_title('Hourly Crime Pattern', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Crime by Type
        ax3 = fig.add_subplot(gs[1, 2])
        crime_counts = self.crime_data['crime_type'].value_counts().head(6)
        ax3.barh(range(len(crime_counts)), crime_counts.values, color='coral', edgecolor='darkred')
        ax3.set_yticks(range(len(crime_counts)))
        ax3.set_yticklabels(crime_counts.index, fontsize=9)
        ax3.set_xlabel('Count')
        ax3.set_title('Top Crime Types', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. Severity Distribution by Hotspot
        ax4 = fig.add_subplot(gs[2, 0])
        if self.hotspots is not None and len(self.hotspots) > 0:
            top_hotspots = self.hotspots.head(5)
            x_pos = np.arange(len(top_hotspots))
            ax4.bar(x_pos, top_hotspots['severity_high_pct'],
                   color='crimson', alpha=0.7, edgecolor='darkred')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f"H{h}" for h in top_hotspots['hotspot_id']])
            ax4.set_ylabel('High Severity (%)')
            ax4.set_title('High Severity Crimes by Hotspot', fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')

        # 5. Day of Week Pattern
        ax5 = fig.add_subplot(gs[2, 1])
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily = self.crime_data.groupby('day_of_week').size().reindex(day_order)
        colors = ['steelblue'] * 5 + ['coral', 'coral']
        ax5.bar(range(7), daily.values, color=colors, edgecolor='black')
        ax5.set_xticks(range(7))
        ax5.set_xticklabels([d[:3] for d in day_order], rotation=45)
        ax5.set_ylabel('Number of Crimes')
        ax5.set_title('Crimes by Day of Week', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Incidents per Hotspot
        ax6 = fig.add_subplot(gs[2, 2])
        if self.hotspots is not None and len(self.hotspots) > 0:
            top_hotspots = self.hotspots.head(8)
            ax6.bar(range(len(top_hotspots)), top_hotspots['n_incidents'],
                   color='#3498db', edgecolor='navy')
            ax6.set_xticks(range(len(top_hotspots)))
            ax6.set_xticklabels([f"H{h}" for h in top_hotspots['hotspot_id']])
            ax6.set_ylabel('Number of Incidents')
            ax6.set_title('Incidents per Hotspot', fontsize=11, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')

        plt.savefig('crime_hotspot_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'crime_hotspot_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("CRIME HOTSPOT MAPPING - GEOSPATIAL ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = CrimeHotspotAnalyzer(city_name="Metro City")

    # Generate data
    analyzer.create_sample_data()

    # Detect hotspots
    hotspots = analyzer.detect_hotspots_dbscan(eps_km=0.5, min_samples=25)

    # Temporal analysis
    analyzer.temporal_analysis()

    # Calculate risk scores
    lat_grid, lon_grid, risk_grid = analyzer.calculate_risk_scores(grid_resolution=40)

    # Visualize results
    analyzer.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
