"""
Spatial Autocorrelation Analysis - Moran's I and Geary's C
Analyze spatial autocorrelation patterns using Moran's I and Geary's C statistics

Dataset: Synthetic regional data with spatial dependencies
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class SpatialAutocorrelationAnalyzer:
    """Analyze spatial autocorrelation using various methods"""

    def __init__(self):
        self.regions = None
        self.weights_matrix = None
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

    def generate_spatial_data(self, n_regions=50, autocorrelation_strength=0.7):
        """Generate synthetic regional data with spatial autocorrelation"""
        print("="*60)
        print("GENERATING SPATIAL DATA")
        print("="*60)

        np.random.seed(42)

        # Generate region centers on a grid
        grid_size = int(np.ceil(np.sqrt(n_regions)))
        region_data = []

        for i in range(grid_size):
            for j in range(grid_size):
                if len(region_data) >= n_regions:
                    break

                x = (i + np.random.uniform(0.2, 0.8)) * 10
                y = (j + np.random.uniform(0.2, 0.8)) * 10

                region_data.append({
                    'region_id': len(region_data),
                    'x': x,
                    'y': y,
                    'name': f'Region_{len(region_data)}'
                })

        self.regions = pd.DataFrame(region_data)

        # Generate spatially autocorrelated variable
        # Start with random values
        base_values = np.random.normal(50, 15, n_regions)

        # Create distance matrix
        coords = self.regions[['x', 'y']].values
        distances = cdist(coords, coords, metric='euclidean')

        # Apply spatial smoothing
        for iteration in range(10):
            weights = np.exp(-distances / 10)  # Distance decay
            np.fill_diagonal(weights, 0)
            weights = weights / weights.sum(axis=1, keepdims=True)

            smoothed = weights @ base_values
            base_values = (1 - autocorrelation_strength) * base_values + autocorrelation_strength * smoothed

        self.regions['value'] = base_values
        self.regions['value_category'] = pd.cut(base_values, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

        # Generate additional variables
        self.regions['population'] = np.random.lognormal(10, 0.5, n_regions)
        self.regions['income'] = base_values * np.random.uniform(800, 1200, n_regions)
        self.regions['employment_rate'] = np.clip(base_values + np.random.normal(0, 5, n_regions), 20, 95)

        print(f"✓ Generated {len(self.regions)} regions")
        print(f"✓ Autocorrelation strength: {autocorrelation_strength}")
        print(f"✓ Value range: [{self.regions['value'].min():.2f}, {self.regions['value'].max():.2f}]")
        print(f"✓ Mean value: {self.regions['value'].mean():.2f}")
        print(f"✓ Std value: {self.regions['value'].std():.2f}")

        return self.regions

    def create_weights_matrix(self, method='distance', threshold=15):
        """Create spatial weights matrix"""
        print("\n" + "="*60)
        print(f"CREATING WEIGHTS MATRIX ({method.upper()})")
        print("="*60)

        n = len(self.regions)
        coords = self.regions[['x', 'y']].values
        distances = cdist(coords, coords, metric='euclidean')

        if method == 'distance':
            # Distance-based weights
            weights = np.exp(-distances / threshold)
            np.fill_diagonal(weights, 0)

        elif method == 'k_nearest':
            # K-nearest neighbors
            k = min(8, n - 1)
            weights = np.zeros((n, n))

            for i in range(n):
                # Find k nearest neighbors
                nearest = np.argsort(distances[i])[1:k+1]
                weights[i, nearest] = 1

        elif method == 'threshold':
            # Binary threshold
            weights = (distances < threshold).astype(float)
            np.fill_diagonal(weights, 0)

        elif method == 'contiguity':
            # Simulated contiguity based on proximity
            threshold_dist = 15
            weights = (distances < threshold_dist).astype(float)
            np.fill_diagonal(weights, 0)

        # Row-normalize weights
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        self.weights_matrix = weights / row_sums

        # Calculate weights statistics
        avg_neighbors = (weights > 0).sum(axis=1).mean()
        min_neighbors = (weights > 0).sum(axis=1).min()
        max_neighbors = (weights > 0).sum(axis=1).max()

        print(f"✓ Weights matrix created")
        print(f"✓ Average neighbors per region: {avg_neighbors:.2f}")
        print(f"✓ Min/Max neighbors: {min_neighbors}/{max_neighbors}")
        print(f"✓ Matrix density: {(weights > 0).sum() / (n * n):.2%}")

        self.metrics['avg_neighbors'] = avg_neighbors
        self.metrics['weights_method'] = method

        return self.weights_matrix

    def calculate_morans_i(self, variable='value'):
        """Calculate Moran's I statistic"""
        print("\n" + "="*60)
        print("CALCULATING MORAN'S I")
        print("="*60)

        x = self.regions[variable].values
        n = len(x)
        x_mean = x.mean()

        # Deviations from mean
        z = x - x_mean

        # Numerator: sum of weighted cross-products
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += self.weights_matrix[i, j] * z[i] * z[j]

        # Denominator: sum of squared deviations
        denominator = (z ** 2).sum()

        # Sum of all weights
        S0 = self.weights_matrix.sum()

        # Moran's I
        I = (n / S0) * (numerator / denominator)

        # Expected value under null hypothesis (no autocorrelation)
        E_I = -1 / (n - 1)

        # Variance calculation
        S1 = 0.5 * ((self.weights_matrix + self.weights_matrix.T) ** 2).sum()
        S2 = ((self.weights_matrix.sum(axis=0) + self.weights_matrix.sum(axis=1)) ** 2).sum()

        b2 = (n * ((z ** 4).sum())) / (((z ** 2).sum()) ** 2)

        var_I = ((n * ((n**2 - 3*n + 3) * S1 - n*S2 + 3*(S0**2))) -
                 (b2 * ((n**2 - n) * S1 - 2*n*S2 + 6*(S0**2))))
        var_I = var_I / (((n - 1) * (n - 2) * (n - 3) * (S0**2)))

        # Z-score and p-value
        z_score = (I - E_I) / np.sqrt(var_I)
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        print(f"\nMoran's I Results:")
        print(f"  Moran's I: {I:.4f}")
        print(f"  Expected I: {E_I:.4f}")
        print(f"  Variance: {var_I:.6f}")
        print(f"  Z-score: {z_score:.4f}")
        print(f"  P-value: {p_value:.4f}")

        if p_value < 0.01:
            significance = "highly significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "significant (p < 0.05)"
        elif p_value < 0.10:
            significance = "marginally significant (p < 0.10)"
        else:
            significance = "not significant"

        if I > E_I:
            pattern = "positive spatial autocorrelation (clustering)"
        else:
            pattern = "negative spatial autocorrelation (dispersion)"

        print(f"\nInterpretation: {pattern}, {significance}")

        self.metrics['morans_i'] = I
        self.metrics['morans_i_z'] = z_score
        self.metrics['morans_i_p'] = p_value
        self.metrics['morans_i_expected'] = E_I

        return I, z_score, p_value

    def calculate_gearys_c(self, variable='value'):
        """Calculate Geary's C statistic"""
        print("\n" + "="*60)
        print("CALCULATING GEARY'S C")
        print("="*60)

        x = self.regions[variable].values
        n = len(x)
        x_mean = x.mean()

        # Numerator: sum of weighted squared differences
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += self.weights_matrix[i, j] * (x[i] - x[j]) ** 2

        # Denominator: sum of squared deviations
        denominator = 2 * ((x - x_mean) ** 2).sum()

        # Sum of all weights
        S0 = self.weights_matrix.sum()

        # Geary's C
        C = ((n - 1) / S0) * (numerator / denominator)

        # Expected value under null hypothesis
        E_C = 1.0

        print(f"\nGeary's C Results:")
        print(f"  Geary's C: {C:.4f}")
        print(f"  Expected C: {E_C:.4f}")

        if C < 1:
            pattern = "positive spatial autocorrelation (similar values clustered)"
        elif C > 1:
            pattern = "negative spatial autocorrelation (dissimilar values clustered)"
        else:
            pattern = "random spatial pattern"

        print(f"\nInterpretation: {pattern}")

        self.metrics['gearys_c'] = C
        self.metrics['gearys_c_expected'] = E_C

        return C

    def local_morans_i(self, variable='value'):
        """Calculate Local Moran's I (LISA)"""
        print("\n" + "="*60)
        print("CALCULATING LOCAL MORAN'S I (LISA)")
        print("="*60)

        x = self.regions[variable].values
        n = len(x)
        x_mean = x.mean()
        x_std = x.std()

        # Standardized values
        z = (x - x_mean) / x_std

        # Local Moran's I for each region
        local_I = np.zeros(n)

        for i in range(n):
            # Weighted sum of neighboring values
            neighbors_z = (self.weights_matrix[i] * z).sum()
            local_I[i] = z[i] * neighbors_z

        # Classify into quadrants
        quadrants = []
        for i in range(n):
            neighbors_z = (self.weights_matrix[i] * z).sum()

            if z[i] > 0 and neighbors_z > 0:
                quadrants.append('HH')  # High-High
            elif z[i] < 0 and neighbors_z < 0:
                quadrants.append('LL')  # Low-Low
            elif z[i] > 0 and neighbors_z < 0:
                quadrants.append('HL')  # High-Low
            else:
                quadrants.append('LH')  # Low-High

        self.regions['local_morans_i'] = local_I
        self.regions['lisa_quadrant'] = quadrants

        # Summary statistics
        quadrant_counts = pd.Series(quadrants).value_counts()

        print(f"\nLocal Moran's I Summary:")
        print(f"  Mean Local I: {local_I.mean():.4f}")
        print(f"  Std Local I: {local_I.std():.4f}")
        print(f"  Min/Max Local I: {local_I.min():.4f}/{local_I.max():.4f}")

        print(f"\nLISA Quadrant Distribution:")
        for quadrant in ['HH', 'LL', 'HL', 'LH']:
            count = quadrant_counts.get(quadrant, 0)
            pct = 100 * count / n
            print(f"  {quadrant}: {count} regions ({pct:.1f}%)")

        self.metrics['local_morans_i_mean'] = local_I.mean()
        self.metrics['quadrant_counts'] = quadrant_counts.to_dict()

        return local_I, quadrants

    def getis_ord_gi_star(self, variable='value'):
        """Calculate Getis-Ord Gi* hot spot statistic"""
        print("\n" + "="*60)
        print("CALCULATING GETIS-ORD GI* STATISTIC")
        print("="*60)

        x = self.regions[variable].values
        n = len(x)
        x_mean = x.mean()
        x_std = x.std()

        # Include self in weights for Gi*
        weights_star = self.weights_matrix.copy()
        np.fill_diagonal(weights_star, 1)

        # Row-normalize
        row_sums = weights_star.sum(axis=1, keepdims=True)
        weights_star = weights_star / row_sums

        # Calculate Gi* for each region
        gi_star = np.zeros(n)
        z_scores = np.zeros(n)

        for i in range(n):
            # Sum of weighted values
            weighted_sum = (weights_star[i] * x).sum()

            # Number of neighbors (including self)
            W_i = weights_star[i].sum()

            # Expected value and variance
            E_gi = x_mean
            var_gi = (x_std ** 2) * ((n * W_i - W_i ** 2) / (n - 1))

            if var_gi > 0:
                z_scores[i] = (weighted_sum - E_gi) / np.sqrt(var_gi)
                gi_star[i] = weighted_sum
            else:
                z_scores[i] = 0
                gi_star[i] = weighted_sum

        # Classify hot spots and cold spots
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

        self.regions['gi_star'] = gi_star
        self.regions['gi_star_z'] = z_scores
        self.regions['hotspot_classification'] = classifications

        # Summary
        class_counts = pd.Series(classifications).value_counts()

        print(f"\nGi* Statistics:")
        print(f"  Mean Gi* Z-score: {z_scores.mean():.4f}")
        print(f"  Min/Max Z-score: {z_scores.min():.4f}/{z_scores.max():.4f}")

        print(f"\nHot Spot Classification:")
        for classification in class_counts.index:
            count = class_counts[classification]
            pct = 100 * count / n
            print(f"  {classification}: {count} ({pct:.1f}%)")

        self.metrics['gi_star_mean'] = z_scores.mean()
        self.metrics['hotspot_counts'] = class_counts.to_dict()

        return gi_star, z_scores, classifications

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(18, 14))

        # 1. Spatial distribution of values
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(self.regions['x'], self.regions['y'],
                            c=self.regions['value'], s=200,
                            cmap='RdYlBu_r', edgecolors='black', linewidths=1)
        plt.colorbar(scatter, ax=ax1, label='Value')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Spatial Distribution of Values', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. LISA quadrants
        ax2 = plt.subplot(2, 3, 2)
        colors_lisa = {'HH': 'red', 'LL': 'blue', 'HL': 'pink', 'LH': 'lightblue'}
        for quadrant, color in colors_lisa.items():
            mask = self.regions['lisa_quadrant'] == quadrant
            ax2.scatter(self.regions.loc[mask, 'x'], self.regions.loc[mask, 'y'],
                       c=color, s=200, label=quadrant, edgecolors='black', linewidths=1)
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title('LISA Cluster Map', fontsize=12, fontweight='bold')
        ax2.legend(title='Quadrant')
        ax2.grid(True, alpha=0.3)

        # 3. Hot spot analysis
        ax3 = plt.subplot(2, 3, 3)
        scatter = ax3.scatter(self.regions['x'], self.regions['y'],
                            c=self.regions['gi_star_z'], s=200,
                            cmap='RdBu_r', edgecolors='black', linewidths=1,
                            vmin=-3, vmax=3)
        plt.colorbar(scatter, ax=ax3, label='Gi* Z-score')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_title('Getis-Ord Gi* Hot Spot Analysis', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Moran's I scatterplot
        ax4 = plt.subplot(2, 3, 4)
        x_std = (self.regions['value'] - self.regions['value'].mean()) / self.regions['value'].std()
        lag_std = self.weights_matrix @ x_std

        ax4.scatter(x_std, lag_std, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
        ax4.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax4.axvline(0, color='red', linestyle='--', alpha=0.5)

        # Add regression line
        z = np.polyfit(x_std, lag_std, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_std.min(), x_std.max(), 100)
        ax4.plot(x_line, p(x_line), "b-", linewidth=2, alpha=0.8,
                label=f"Slope = {z[0]:.3f}")

        ax4.set_xlabel('Standardized Value')
        ax4.set_ylabel('Spatial Lag (Standardized)')
        ax4.set_title(f"Moran's I Scatterplot (I={self.metrics['morans_i']:.3f})",
                     fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Autocorrelation statistics
        ax5 = plt.subplot(2, 3, 5)
        stats = {
            "Moran's I": self.metrics['morans_i'],
            "Geary's C": self.metrics['gearys_c'],
            "Local I\n(mean)": self.metrics['local_morans_i_mean']
        }
        bars = ax5.bar(range(len(stats)), list(stats.values()),
                      color=['#3498db', '#e74c3c', '#2ecc71'], edgecolor='black')
        ax5.set_xticks(range(len(stats)))
        ax5.set_xticklabels(stats.keys())
        ax5.set_ylabel('Statistic Value')
        ax5.set_title('Spatial Autocorrelation Statistics', fontsize=12, fontweight='bold')
        ax5.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax5.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center',
                    va='bottom' if height >= 0 else 'top', fontweight='bold')

        # 6. Hot spot classification distribution
        ax6 = plt.subplot(2, 3, 6)
        hotspot_counts = self.regions['hotspot_classification'].value_counts()
        colors_hot = []
        for label in hotspot_counts.index:
            if 'Hot Spot' in label:
                colors_hot.append('#d62728')
            elif 'Cold Spot' in label:
                colors_hot.append('#1f77b4')
            else:
                colors_hot.append('#7f7f7f')

        bars = ax6.barh(range(len(hotspot_counts)), hotspot_counts.values,
                       color=colors_hot, edgecolor='black')
        ax6.set_yticks(range(len(hotspot_counts)))
        ax6.set_yticklabels(hotspot_counts.index, fontsize=9)
        ax6.set_xlabel('Number of Regions')
        ax6.set_title('Hot Spot Classification Distribution', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='x')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax6.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig('spatial_autocorrelation_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'spatial_autocorrelation_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("SPATIAL AUTOCORRELATION ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = SpatialAutocorrelationAnalyzer()

    # Generate data
    analyzer.generate_spatial_data(n_regions=50, autocorrelation_strength=0.7)

    # Create weights matrix
    analyzer.create_weights_matrix(method='distance', threshold=15)

    # Calculate Moran's I
    analyzer.calculate_morans_i(variable='value')

    # Calculate Geary's C
    analyzer.calculate_gearys_c(variable='value')

    # Calculate Local Moran's I
    analyzer.local_morans_i(variable='value')

    # Calculate Getis-Ord Gi*
    analyzer.getis_ord_gi_star(variable='value')

    # Visualize results
    analyzer.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
