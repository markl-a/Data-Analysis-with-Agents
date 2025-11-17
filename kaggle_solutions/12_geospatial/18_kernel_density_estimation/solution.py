"""
Kernel Density Estimation for Spatial Data - Geospatial Analysis
Implement kernel density estimation for point patterns and crime mapping

Dataset: Synthetic point events (crimes, accidents, etc.)
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
from sklearn.neighbors import KernelDensity
import warnings
warnings.filterwarnings('ignore')


class KernelDensityAnalyzer:
    """Analyze spatial point patterns using kernel density estimation"""

    def __init__(self):
        self.events = None
        self.density_grid = None
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

    def generate_event_data(self, n_events=2000, pattern='clustered'):
        """Generate synthetic spatial event data"""
        print("="*60)
        print("GENERATING EVENT DATA")
        print("="*60)

        np.random.seed(42)

        events = []

        if pattern == 'clustered':
            # Generate clustered events (hotspots)
            n_clusters = 6
            cluster_centers = []

            for i in range(n_clusters):
                center_x = np.random.uniform(10, 90)
                center_y = np.random.uniform(10, 90)
                intensity = np.random.uniform(0.3, 0.7)
                cluster_centers.append((center_x, center_y, intensity))

            for i in range(n_events):
                if np.random.random() < 0.75:  # 75% in clusters
                    center_x, center_y, intensity = cluster_centers[np.random.randint(n_clusters)]
                    std = np.random.uniform(2, 5)
                    x = center_x + np.random.normal(0, std)
                    y = center_y + np.random.normal(0, std)
                else:  # 25% background
                    x = np.random.uniform(0, 100)
                    y = np.random.uniform(0, 100)

                events.append({
                    'event_id': i,
                    'x': np.clip(x, 0, 100),
                    'y': np.clip(y, 0, 100),
                    'weight': np.random.uniform(0.5, 1.5),
                    'type': np.random.choice(['Type A', 'Type B', 'Type C'])
                })

        elif pattern == 'uniform':
            for i in range(n_events):
                events.append({
                    'event_id': i,
                    'x': np.random.uniform(0, 100),
                    'y': np.random.uniform(0, 100),
                    'weight': 1.0,
                    'type': 'Type A'
                })

        elif pattern == 'regular':
            # Regular grid with jitter
            grid_size = int(np.sqrt(n_events))
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(events) >= n_events:
                        break
                    x = (i + 0.5 + np.random.normal(0, 0.2)) * (100 / grid_size)
                    y = (j + 0.5 + np.random.normal(0, 0.2)) * (100 / grid_size)

                    events.append({
                        'event_id': len(events),
                        'x': np.clip(x, 0, 100),
                        'y': np.clip(y, 0, 100),
                        'weight': 1.0,
                        'type': 'Type A'
                    })

        self.events = pd.DataFrame(events)

        print(f"✓ Generated {len(self.events)} events")
        print(f"✓ Pattern type: {pattern}")
        print(f"✓ Spatial extent: 100 x 100 units")

        return self.events

    def gaussian_kde_2d(self, bandwidth='scott', grid_size=100):
        """2D Gaussian kernel density estimation"""
        print("\n" + "="*60)
        print(f"GAUSSIAN KDE (bandwidth={bandwidth})")
        print("="*60)

        # Create grid
        x = np.linspace(0, 100, grid_size)
        y = np.linspace(0, 100, grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Prepare data
        event_coords = self.events[['x', 'y']].values.T

        # Compute KDE
        kde = gaussian_kde(event_coords, bw_method=bandwidth)
        density = kde(grid_points.T)

        # Reshape to grid
        density_grid = density.reshape(grid_size, grid_size)

        # Store results
        self.density_grid = density_grid

        print(f"✓ KDE computed on {grid_size}x{grid_size} grid")
        print(f"✓ Max density: {density.max():.6f}")
        print(f"✓ Mean density: {density.mean():.6f}")

        self.metrics['kde_max'] = density.max()
        self.metrics['kde_mean'] = density.mean()
        self.metrics['bandwidth'] = bandwidth

        return xx, yy, density_grid

    def sklearn_kde(self, bandwidth=5, kernel='gaussian', grid_size=100):
        """KDE using sklearn with various kernels"""
        print("\n" + "="*60)
        print(f"SKLEARN KDE (kernel={kernel}, bandwidth={bandwidth})")
        print("="*60)

        # Create grid
        x = np.linspace(0, 100, grid_size)
        y = np.linspace(0, 100, grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Prepare data
        event_coords = self.events[['x', 'y']].values

        # Compute KDE
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(event_coords)

        # Evaluate on grid
        log_density = kde.score_samples(grid_points)
        density = np.exp(log_density)

        # Reshape to grid
        density_grid = density.reshape(grid_size, grid_size)

        print(f"✓ KDE computed with {kernel} kernel")
        print(f"✓ Max density: {density.max():.6f}")
        print(f"✓ Mean density: {density.mean():.6f}")

        return xx, yy, density_grid

    def adaptive_kde(self, pilot_bandwidth=5, grid_size=100):
        """Adaptive KDE with varying bandwidth"""
        print("\n" + "="*60)
        print("ADAPTIVE KDE")
        print("="*60)

        # Create grid
        x = np.linspace(0, 100, grid_size)
        y = np.linspace(0, 100, grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Event coordinates
        event_coords = self.events[['x', 'y']].values

        # Compute pilot density at event locations
        kde_pilot = KernelDensity(bandwidth=pilot_bandwidth)
        kde_pilot.fit(event_coords)
        pilot_density = np.exp(kde_pilot.score_samples(event_coords))

        # Compute local bandwidths (inverse of pilot density)
        pilot_mean = pilot_density.mean()
        local_bandwidths = pilot_bandwidth * np.sqrt(pilot_mean / (pilot_density + 1e-10))

        # Compute adaptive KDE
        density = np.zeros(len(grid_points))

        for i, point in enumerate(grid_points):
            # Distance to all events
            distances = np.sqrt(((event_coords - point) ** 2).sum(axis=1))

            # Weighted sum of kernels with adaptive bandwidth
            kernel_values = np.exp(-0.5 * (distances / local_bandwidths) ** 2)
            kernel_values /= (local_bandwidths * np.sqrt(2 * np.pi))

            density[i] = kernel_values.sum() / len(event_coords)

        density_grid = density.reshape(grid_size, grid_size)

        print(f"✓ Adaptive KDE computed")
        print(f"✓ Bandwidth range: [{local_bandwidths.min():.2f}, {local_bandwidths.max():.2f}]")
        print(f"✓ Max density: {density.max():.6f}")

        return xx, yy, density_grid

    def weighted_kde(self, bandwidth=5, grid_size=100):
        """Weighted KDE using event weights"""
        print("\n" + "="*60)
        print("WEIGHTED KDE")
        print("="*60)

        # Create grid
        x = np.linspace(0, 100, grid_size)
        y = np.linspace(0, 100, grid_size)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Event coordinates and weights
        event_coords = self.events[['x', 'y']].values
        weights = self.events['weight'].values

        # Compute weighted KDE
        density = np.zeros(len(grid_points))

        for i, point in enumerate(grid_points):
            distances = np.sqrt(((event_coords - point) ** 2).sum(axis=1))
            kernel_values = np.exp(-0.5 * (distances / bandwidth) ** 2)
            kernel_values /= (bandwidth ** 2 * 2 * np.pi)

            # Weight by event weights
            density[i] = (kernel_values * weights).sum()

        density_grid = density.reshape(grid_size, grid_size)

        print(f"✓ Weighted KDE computed")
        print(f"✓ Max density: {density.max():.6f}")

        return xx, yy, density_grid

    def bandwidth_selection(self, bandwidths=None, cv_folds=5):
        """Cross-validation for bandwidth selection"""
        print("\n" + "="*60)
        print("BANDWIDTH SELECTION (CROSS-VALIDATION)")
        print("="*60)

        if bandwidths is None:
            bandwidths = [1, 2, 5, 10, 15, 20]

        event_coords = self.events[['x', 'y']].values
        n = len(event_coords)

        scores = []

        for bw in bandwidths:
            fold_scores = []

            for fold in range(cv_folds):
                # Split data
                test_size = n // cv_folds
                test_start = fold * test_size
                test_end = test_start + test_size

                test_indices = list(range(test_start, test_end))
                train_indices = [i for i in range(n) if i not in test_indices]

                train_data = event_coords[train_indices]
                test_data = event_coords[test_indices]

                # Fit KDE on training data
                kde = KernelDensity(bandwidth=bw)
                kde.fit(train_data)

                # Evaluate on test data
                score = kde.score(test_data)
                fold_scores.append(score)

            mean_score = np.mean(fold_scores)
            scores.append(mean_score)

            print(f"  Bandwidth {bw:5.1f}: CV score = {mean_score:.4f}")

        best_idx = np.argmax(scores)
        best_bw = bandwidths[best_idx]

        print(f"\n✓ Best bandwidth: {best_bw}")

        self.metrics['best_bandwidth'] = best_bw
        self.metrics['cv_scores'] = dict(zip(bandwidths, scores))

        return best_bw, scores

    def identify_hotspots(self, threshold_percentile=95):
        """Identify hotspot regions from density grid"""
        print("\n" + "="*60)
        print(f"IDENTIFYING HOTSPOTS (>{threshold_percentile}th percentile)")
        print("="*60)

        if self.density_grid is None:
            print("No density grid available")
            return None

        threshold = np.percentile(self.density_grid, threshold_percentile)

        hotspot_mask = self.density_grid > threshold
        hotspot_area = hotspot_mask.sum() / self.density_grid.size

        print(f"✓ Density threshold: {threshold:.6f}")
        print(f"✓ Hotspot area: {hotspot_area*100:.2f}% of total area")
        print(f"✓ Number of hotspot cells: {hotspot_mask.sum()}")

        self.metrics['hotspot_threshold'] = threshold
        self.metrics['hotspot_area_pct'] = hotspot_area * 100

        return hotspot_mask, threshold

    def nearest_neighbor_distance_distribution(self):
        """Analyze nearest neighbor distances"""
        print("\n" + "="*60)
        print("NEAREST NEIGHBOR DISTANCE ANALYSIS")
        print("="*60)

        event_coords = self.events[['x', 'y']].values
        n = len(event_coords)

        # Compute all pairwise distances
        distances = cdist(event_coords, event_coords, metric='euclidean')

        # Set diagonal to infinity
        np.fill_diagonal(distances, np.inf)

        # Find nearest neighbor for each event
        nn_distances = distances.min(axis=1)

        # Calculate metrics
        mean_nn = nn_distances.mean()
        expected_nn = 0.5 / np.sqrt(n / (100 * 100))  # Expected for random pattern

        # Nearest Neighbor Index (Clark-Evans)
        nn_index = mean_nn / expected_nn

        print(f"\nNearest Neighbor Statistics:")
        print(f"  Mean NN distance: {mean_nn:.4f}")
        print(f"  Expected NN (random): {expected_nn:.4f}")
        print(f"  NN Index: {nn_index:.4f}")

        if nn_index < 1:
            pattern = "clustered (aggregated)"
        elif nn_index > 1:
            pattern = "dispersed (regular)"
        else:
            pattern = "random"

        print(f"  Interpretation: {pattern}")

        self.metrics['nn_index'] = nn_index
        self.metrics['mean_nn_distance'] = mean_nn

        return nn_distances, nn_index

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(18, 12))

        # 1. Event points
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(self.events['x'], self.events['y'], c='blue', s=10, alpha=0.5)
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title(f'Event Points (n={len(self.events)})', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)

        # 2. Gaussian KDE
        ax2 = plt.subplot(2, 3, 2)
        xx, yy, density_grid = self.gaussian_kde_2d(bandwidth='scott', grid_size=100)
        contour = ax2.contourf(xx, yy, density_grid, levels=20, cmap='YlOrRd', alpha=0.8)
        plt.colorbar(contour, ax=ax2, label='Density')
        ax2.scatter(self.events['x'], self.events['y'], c='black', s=5, alpha=0.2)
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title('Gaussian KDE Heatmap', fontsize=12, fontweight='bold')

        # 3. Hotspot identification
        ax3 = plt.subplot(2, 3, 3)
        hotspot_mask, threshold = self.identify_hotspots(threshold_percentile=95)
        ax3.contourf(xx, yy, density_grid, levels=20, cmap='YlOrRd', alpha=0.5)
        ax3.contour(xx, yy, hotspot_mask.astype(int), levels=[0.5], colors='red',
                   linewidths=2, linestyles='--')
        ax3.scatter(self.events['x'], self.events['y'], c='black', s=5, alpha=0.2)
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_title('Hotspot Regions (>95th percentile)', fontsize=12, fontweight='bold')

        # 4. Bandwidth comparison
        ax4 = plt.subplot(2, 3, 4)
        if 'cv_scores' in self.metrics:
            bandwidths = list(self.metrics['cv_scores'].keys())
            scores = list(self.metrics['cv_scores'].values())
            ax4.plot(bandwidths, scores, marker='o', linewidth=2, markersize=8)
            ax4.axvline(self.metrics['best_bandwidth'], color='red',
                       linestyle='--', linewidth=2, label=f"Best BW = {self.metrics['best_bandwidth']}")
            ax4.set_xlabel('Bandwidth')
            ax4.set_ylabel('CV Score')
            ax4.set_title('Bandwidth Selection (Cross-Validation)', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Nearest neighbor distances
        ax5 = plt.subplot(2, 3, 5)
        nn_distances, nn_index = self.nearest_neighbor_distance_distribution()
        ax5.hist(nn_distances, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax5.axvline(nn_distances.mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean = {nn_distances.mean():.2f}')
        ax5.set_xlabel('Nearest Neighbor Distance')
        ax5.set_ylabel('Frequency')
        ax5.set_title(f'NN Distance Distribution (Index={nn_index:.3f})',
                     fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Density profile (cross-section)
        ax6 = plt.subplot(2, 3, 6)
        if self.density_grid is not None:
            # Take cross-section at y=50
            middle_row = self.density_grid[len(self.density_grid)//2, :]
            x_vals = np.linspace(0, 100, len(middle_row))
            ax6.plot(x_vals, middle_row, linewidth=2, color='darkblue')
            ax6.fill_between(x_vals, middle_row, alpha=0.3, color='blue')
            ax6.set_xlabel('X Coordinate')
            ax6.set_ylabel('Density')
            ax6.set_title('Density Profile (y=50)', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('kernel_density_estimation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'kernel_density_estimation.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("KERNEL DENSITY ESTIMATION FOR SPATIAL DATA")
    print("="*60)

    # Initialize analyzer
    analyzer = KernelDensityAnalyzer()

    # Generate event data
    analyzer.generate_event_data(n_events=2000, pattern='clustered')

    # Bandwidth selection
    analyzer.bandwidth_selection(bandwidths=[2, 5, 8, 10, 15, 20], cv_folds=5)

    # Various KDE methods
    analyzer.gaussian_kde_2d(bandwidth='scott', grid_size=100)
    analyzer.sklearn_kde(bandwidth=5, kernel='gaussian', grid_size=100)
    analyzer.adaptive_kde(pilot_bandwidth=5, grid_size=100)
    analyzer.weighted_kde(bandwidth=5, grid_size=100)

    # Nearest neighbor analysis
    analyzer.nearest_neighbor_distance_distribution()

    # Visualize results
    analyzer.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
