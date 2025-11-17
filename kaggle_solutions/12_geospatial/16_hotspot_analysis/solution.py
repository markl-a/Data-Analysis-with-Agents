"""
Hot Spot Analysis - Getis-Ord Gi* and Cluster Detection
Identify statistically significant hot spots and cold spots in spatial data

Dataset: Synthetic crime or disease incident data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')


class HotSpotAnalyzer:
    """Detect and analyze spatial hot spots"""

    def __init__(self):
        self.incidents = None
        self.grid = None
        self.hotspots = None
        self.metrics = {}

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate haversine distance in km"""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def euclidean_distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance"""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def manhattan_distance(self, x1, y1, x2, y2):
        """Calculate Manhattan distance"""
        return abs(x2 - x1) + abs(y2 - y1)

    def generate_incident_data(self, n_incidents=1000, n_hotspots=5):
        """Generate synthetic incident data with known hot spots"""
        print("="*60)
        print("GENERATING INCIDENT DATA")
        print("="*60)

        np.random.seed(42)

        incidents = []

        # Generate hot spot centers
        hotspot_centers = []
        for i in range(n_hotspots):
            center_x = np.random.uniform(10, 90)
            center_y = np.random.uniform(10, 90)
            intensity = np.random.uniform(0.4, 0.8)
            hotspot_centers.append((center_x, center_y, intensity))

        # Generate incidents around hot spots and background
        for i in range(n_incidents):
            # Decide if incident is from hotspot or background
            if np.random.random() < 0.7:  # 70% from hotspots
                # Choose random hotspot
                center_x, center_y, intensity = hotspot_centers[np.random.randint(n_hotspots)]
                x = center_x + np.random.normal(0, 3)
                y = center_y + np.random.normal(0, 3)
                severity = np.random.choice(['Low', 'Medium', 'High'],
                                           p=[0.2, 0.4, 0.4])
            else:  # 30% background noise
                x = np.random.uniform(0, 100)
                y = np.random.uniform(0, 100)
                severity = np.random.choice(['Low', 'Medium', 'High'],
                                           p=[0.6, 0.3, 0.1])

            incidents.append({
                'incident_id': i,
                'x': np.clip(x, 0, 100),
                'y': np.clip(y, 0, 100),
                'severity': severity,
                'severity_score': {'Low': 1, 'Medium': 2, 'High': 3}[severity],
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            })

        self.incidents = pd.DataFrame(incidents)

        print(f"✓ Generated {len(self.incidents)} incidents")
        print(f"✓ Number of hot spots: {n_hotspots}")
        print(f"✓ Severity distribution:")
        print(self.incidents['severity'].value_counts())

        return self.incidents

    def create_density_grid(self, cell_size=5):
        """Create grid and calculate incident density"""
        print("\n" + "="*60)
        print("CREATING DENSITY GRID")
        print("="*60)

        # Create grid
        x_bins = np.arange(0, 100 + cell_size, cell_size)
        y_bins = np.arange(0, 100 + cell_size, cell_size)

        grid_data = []

        for i in range(len(x_bins) - 1):
            for j in range(len(y_bins) - 1):
                x_min, x_max = x_bins[i], x_bins[i + 1]
                y_min, y_max = y_bins[j], y_bins[j + 1]

                # Count incidents in cell
                mask = ((self.incidents['x'] >= x_min) &
                       (self.incidents['x'] < x_max) &
                       (self.incidents['y'] >= y_min) &
                       (self.incidents['y'] < y_max))

                cell_incidents = self.incidents[mask]
                count = len(cell_incidents)
                severity_sum = cell_incidents['severity_score'].sum() if count > 0 else 0

                grid_data.append({
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'x_center': (x_min + x_max) / 2,
                    'y_center': (y_min + y_max) / 2,
                    'count': count,
                    'severity_sum': severity_sum,
                    'density': count / (cell_size ** 2)
                })

        self.grid = pd.DataFrame(grid_data)

        print(f"✓ Created {len(self.grid)} grid cells")
        print(f"✓ Cell size: {cell_size} x {cell_size}")
        print(f"✓ Mean incidents per cell: {self.grid['count'].mean():.2f}")
        print(f"✓ Max incidents in cell: {self.grid['count'].max()}")

        return self.grid

    def getis_ord_gi_star(self, distance_threshold=10):
        """Calculate Getis-Ord Gi* statistic for each grid cell"""
        print("\n" + "="*60)
        print("CALCULATING GETIS-ORD GI* STATISTIC")
        print("="*60)

        n = len(self.grid)
        coords = self.grid[['x_center', 'y_center']].values
        values = self.grid['count'].values

        # Calculate distance matrix
        distances = cdist(coords, coords, metric='euclidean')

        # Create weights matrix (include self)
        weights = (distances <= distance_threshold).astype(float)

        # Calculate Gi* for each cell
        gi_star_scores = []
        z_scores = []

        for i in range(n):
            # Neighbors including self
            neighbors = weights[i] > 0
            n_neighbors = neighbors.sum()

            if n_neighbors > 0:
                # Sum of values in neighborhood
                local_sum = values[neighbors].sum()

                # Global statistics
                global_mean = values.mean()
                global_std = values.std()

                # Expected value
                expected = global_mean * n_neighbors

                # Variance
                variance = global_std ** 2 * n_neighbors

                # Z-score
                if variance > 0:
                    z = (local_sum - expected) / np.sqrt(variance)
                else:
                    z = 0

                gi_star_scores.append(local_sum)
                z_scores.append(z)
            else:
                gi_star_scores.append(0)
                z_scores.append(0)

        self.grid['gi_star'] = gi_star_scores
        self.grid['gi_star_z'] = z_scores

        # Classify hot spots
        classifications = []
        for z in z_scores:
            if z > 2.58:
                classifications.append('Hot Spot (99%)')
            elif z > 1.96:
                classifications.append('Hot Spot (95%)')
            elif z > 1.65:
                classifications.append('Hot Spot (90%)')
            elif z < -2.58:
                classifications.append('Cold Spot (99%)')
            elif z < -1.96:
                classifications.append('Cold Spot (95%)')
            elif z < -1.65:
                classifications.append('Cold Spot (90%)')
            else:
                classifications.append('Not Significant')

        self.grid['classification'] = classifications

        # Summary
        class_counts = pd.Series(classifications).value_counts()

        print(f"\nGi* Statistics:")
        print(f"  Mean Z-score: {np.mean(z_scores):.4f}")
        print(f"  Std Z-score: {np.std(z_scores):.4f}")
        print(f"  Min/Max Z-score: {np.min(z_scores):.4f}/{np.max(z_scores):.4f}")

        print(f"\nClassification Distribution:")
        for classification in class_counts.index:
            count = class_counts[classification]
            pct = 100 * count / n
            print(f"  {classification}: {count} ({pct:.1f}%)")

        self.metrics['gi_star_mean'] = np.mean(z_scores)
        self.metrics['classification_counts'] = class_counts.to_dict()

        return z_scores, classifications

    def identify_hotspot_clusters(self, min_z_score=1.96):
        """Identify spatially contiguous hot spot clusters"""
        print("\n" + "="*60)
        print("IDENTIFYING HOT SPOT CLUSTERS")
        print("="*60)

        # Filter significant hot spots
        hotspot_cells = self.grid[self.grid['gi_star_z'] > min_z_score].copy()

        if len(hotspot_cells) == 0:
            print("No significant hot spots found")
            return None

        # Use DBSCAN to cluster hot spot cells
        coords = hotspot_cells[['x_center', 'y_center']].values
        clustering = DBSCAN(eps=7, min_samples=2).fit(coords)
        hotspot_cells['cluster'] = clustering.labels_

        # Analyze clusters
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)

        print(f"\nIdentified {n_clusters} hot spot clusters")

        cluster_summary = []

        for cluster_id in range(n_clusters):
            cluster_mask = hotspot_cells['cluster'] == cluster_id
            cluster_cells = hotspot_cells[cluster_mask]

            cluster_summary.append({
                'cluster_id': cluster_id,
                'n_cells': len(cluster_cells),
                'total_incidents': cluster_cells['count'].sum(),
                'avg_severity': cluster_cells['severity_sum'].mean(),
                'mean_z_score': cluster_cells['gi_star_z'].mean(),
                'centroid_x': cluster_cells['x_center'].mean(),
                'centroid_y': cluster_cells['y_center'].mean()
            })

            print(f"\nCluster {cluster_id}:")
            print(f"  Cells: {len(cluster_cells)}")
            print(f"  Total incidents: {cluster_cells['count'].sum()}")
            print(f"  Mean Z-score: {cluster_cells['gi_star_z'].mean():.2f}")
            print(f"  Centroid: ({cluster_cells['x_center'].mean():.2f}, {cluster_cells['y_center'].mean():.2f})")

        self.hotspots = pd.DataFrame(cluster_summary)
        self.metrics['n_hotspot_clusters'] = n_clusters

        return self.hotspots

    def kernel_density_estimation(self, bandwidth=5, grid_size=50):
        """Calculate kernel density estimation"""
        print("\n" + "="*60)
        print("CALCULATING KERNEL DENSITY ESTIMATION")
        print("="*60)

        # Create evaluation grid
        x_grid = np.linspace(0, 100, grid_size)
        y_grid = np.linspace(0, 100, grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Calculate density at each grid point
        incident_coords = self.incidents[['x', 'y']].values

        densities = np.zeros(len(grid_points))

        for i, point in enumerate(grid_points):
            # Calculate distances to all incidents
            distances = np.sqrt(((incident_coords - point) ** 2).sum(axis=1))

            # Apply Gaussian kernel
            kernel_values = np.exp(-0.5 * (distances / bandwidth) ** 2)
            densities[i] = kernel_values.sum()

        # Normalize
        densities = densities / densities.sum()

        # Reshape to grid
        density_grid = densities.reshape(grid_size, grid_size)

        print(f"✓ KDE calculated on {grid_size}x{grid_size} grid")
        print(f"✓ Bandwidth: {bandwidth}")
        print(f"✓ Max density: {densities.max():.6f}")

        self.metrics['kde_bandwidth'] = bandwidth
        self.metrics['kde_max_density'] = densities.max()

        return xx, yy, density_grid

    def temporal_hotspot_analysis(self, time_window_days=30):
        """Analyze how hot spots change over time"""
        print("\n" + "="*60)
        print("TEMPORAL HOT SPOT ANALYSIS")
        print("="*60)

        # Split data into time windows
        min_date = self.incidents['timestamp'].min()
        max_date = self.incidents['timestamp'].max()

        time_windows = pd.date_range(min_date, max_date, freq=f'{time_window_days}D')

        temporal_stats = []

        for i in range(len(time_windows) - 1):
            start_date = time_windows[i]
            end_date = time_windows[i + 1]

            # Filter incidents in window
            window_incidents = self.incidents[
                (self.incidents['timestamp'] >= start_date) &
                (self.incidents['timestamp'] < end_date)
            ]

            temporal_stats.append({
                'period': i,
                'start_date': start_date,
                'end_date': end_date,
                'n_incidents': len(window_incidents),
                'mean_x': window_incidents['x'].mean() if len(window_incidents) > 0 else np.nan,
                'mean_y': window_incidents['y'].mean() if len(window_incidents) > 0 else np.nan
            })

        temporal_df = pd.DataFrame(temporal_stats)

        print(f"\nTemporal Analysis:")
        print(f"  Time windows: {len(time_windows) - 1}")
        print(f"  Window size: {time_window_days} days")
        print(f"  Mean incidents per window: {temporal_df['n_incidents'].mean():.1f}")
        print(f"  Std incidents per window: {temporal_df['n_incidents'].std():.1f}")

        return temporal_df

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(18, 12))

        # 1. Incident points
        ax1 = plt.subplot(2, 3, 1)
        severity_colors = {'Low': 'yellow', 'Medium': 'orange', 'High': 'red'}
        for severity, color in severity_colors.items():
            mask = self.incidents['severity'] == severity
            ax1.scatter(self.incidents.loc[mask, 'x'],
                       self.incidents.loc[mask, 'y'],
                       c=color, s=20, alpha=0.6, label=severity,
                       edgecolors='black', linewidths=0.3)

        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Incident Distribution by Severity', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)

        # 2. Gi* hot spot map
        ax2 = plt.subplot(2, 3, 2)

        # Draw grid cells colored by Z-score
        for _, cell in self.grid.iterrows():
            z = cell['gi_star_z']
            if z > 1.96:
                color = plt.cm.Reds(min((z - 1.96) / 3, 1))
            elif z < -1.96:
                color = plt.cm.Blues(min((-z - 1.96) / 3, 1))
            else:
                color = 'lightgray'

            rect = plt.Rectangle((cell['x_min'], cell['y_min']),
                                cell['x_max'] - cell['x_min'],
                                cell['y_max'] - cell['y_min'],
                                facecolor=color, edgecolor='black',
                                linewidth=0.3, alpha=0.7)
            ax2.add_patch(rect)

        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title('Getis-Ord Gi* Hot Spot Map', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)

        # 3. KDE heatmap
        ax3 = plt.subplot(2, 3, 3)
        xx, yy, density_grid = self.kernel_density_estimation()
        contour = ax3.contourf(xx, yy, density_grid, levels=15, cmap='YlOrRd', alpha=0.8)
        plt.colorbar(contour, ax=ax3, label='Density')
        ax3.scatter(self.incidents['x'], self.incidents['y'],
                   c='black', s=5, alpha=0.2)
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_title('Kernel Density Estimation', fontsize=12, fontweight='bold')

        # 4. Z-score distribution
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(self.grid['gi_star_z'], bins=30, color='steelblue',
                edgecolor='black', alpha=0.7)
        ax4.axvline(1.96, color='red', linestyle='--', linewidth=2, label='95% CI')
        ax4.axvline(-1.96, color='red', linestyle='--', linewidth=2)
        ax4.axvline(2.58, color='darkred', linestyle='--', linewidth=2, label='99% CI')
        ax4.axvline(-2.58, color='darkred', linestyle='--', linewidth=2)
        ax4.set_xlabel('Gi* Z-score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Gi* Z-score Distribution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Classification distribution
        ax5 = plt.subplot(2, 3, 5)
        class_counts = self.grid['classification'].value_counts()
        colors_class = []
        for label in class_counts.index:
            if 'Hot Spot' in label:
                colors_class.append('#d62728')
            elif 'Cold Spot' in label:
                colors_class.append('#1f77b4')
            else:
                colors_class.append('#7f7f7f')

        bars = ax5.barh(range(len(class_counts)), class_counts.values,
                       color=colors_class, edgecolor='black')
        ax5.set_yticks(range(len(class_counts)))
        ax5.set_yticklabels(class_counts.index, fontsize=9)
        ax5.set_xlabel('Number of Grid Cells')
        ax5.set_title('Hot Spot Classification', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        # 6. Hotspot cluster summary
        ax6 = plt.subplot(2, 3, 6)
        if self.hotspots is not None and len(self.hotspots) > 0:
            bars = ax6.bar(self.hotspots['cluster_id'].astype(str),
                          self.hotspots['total_incidents'],
                          color='#e74c3c', edgecolor='black')
            ax6.set_xlabel('Cluster ID')
            ax6.set_ylabel('Total Incidents')
            ax6.set_title('Incidents per Hot Spot Cluster', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')

            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No significant\nhotspot clusters',
                    ha='center', va='center', fontsize=14)
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig('hotspot_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'hotspot_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("HOT SPOT ANALYSIS - GETIS-ORD GI*")
    print("="*60)

    # Initialize analyzer
    analyzer = HotSpotAnalyzer()

    # Generate data
    analyzer.generate_incident_data(n_incidents=1000, n_hotspots=5)

    # Create density grid
    analyzer.create_density_grid(cell_size=5)

    # Calculate Gi*
    analyzer.getis_ord_gi_star(distance_threshold=10)

    # Identify clusters
    analyzer.identify_hotspot_clusters(min_z_score=1.96)

    # Temporal analysis
    analyzer.temporal_hotspot_analysis(time_window_days=60)

    # Visualize results
    analyzer.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
