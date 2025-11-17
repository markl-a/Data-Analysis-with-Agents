"""
Spatial Interpolation Methods - IDW, Kriging, and Splines
Implement various spatial interpolation techniques for predicting values at unsampled locations

Dataset: Synthetic environmental measurements (temperature, pollution, etc.)
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.interpolate import Rbf, griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
warnings.filterwarnings('ignore')


class SpatialInterpolator:
    """Implement various spatial interpolation methods"""

    def __init__(self):
        self.sample_points = None
        self.grid_x = None
        self.grid_y = None
        self.interpolated = {}
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

    def generate_sample_data(self, n_samples=100):
        """Generate synthetic sample points with spatial variation"""
        print("="*60)
        print("GENERATING SAMPLE DATA")
        print("="*60)

        np.random.seed(42)

        # Generate sample locations
        sample_data = []

        # Stratified random sampling for better coverage
        for i in range(int(np.sqrt(n_samples))):
            for j in range(int(np.sqrt(n_samples))):
                if len(sample_data) >= n_samples:
                    break

                x = i * (100 / np.sqrt(n_samples)) + np.random.uniform(0, 100 / np.sqrt(n_samples))
                y = j * (100 / np.sqrt(n_samples)) + np.random.uniform(0, 100 / np.sqrt(n_samples))

                # Create spatially varying field with multiple components
                # Trend component
                trend = 20 + 0.3 * x + 0.2 * y

                # Sinusoidal component (large scale variation)
                sine_component = 10 * np.sin(x / 15) * np.cos(y / 20)

                # Local variation
                local_var = np.random.normal(0, 2)

                value = trend + sine_component + local_var

                sample_data.append({
                    'x': x,
                    'y': y,
                    'value': value,
                    'uncertainty': np.random.uniform(0.5, 2.0)
                })

        self.sample_points = pd.DataFrame(sample_data)

        print(f"✓ Generated {len(self.sample_points)} sample points")
        print(f"✓ Value range: [{self.sample_points['value'].min():.2f}, {self.sample_points['value'].max():.2f}]")
        print(f"✓ Mean value: {self.sample_points['value'].mean():.2f}")
        print(f"✓ Std value: {self.sample_points['value'].std():.2f}")

        return self.sample_points

    def create_interpolation_grid(self, grid_size=50):
        """Create regular grid for interpolation"""
        print("\n" + "="*60)
        print("CREATING INTERPOLATION GRID")
        print("="*60)

        x = np.linspace(0, 100, grid_size)
        y = np.linspace(0, 100, grid_size)
        self.grid_x, self.grid_y = np.meshgrid(x, y)

        print(f"✓ Created {grid_size}x{grid_size} interpolation grid")
        print(f"✓ Total grid points: {grid_size * grid_size}")

        return self.grid_x, self.grid_y

    def inverse_distance_weighting(self, power=2, max_distance=None):
        """Inverse Distance Weighting (IDW) interpolation"""
        print("\n" + "="*60)
        print(f"INVERSE DISTANCE WEIGHTING (power={power})")
        print("="*60)

        sample_coords = self.sample_points[['x', 'y']].values
        sample_values = self.sample_points['value'].values

        grid_points = np.c_[self.grid_x.ravel(), self.grid_y.ravel()]

        # Calculate distances from each grid point to all sample points
        distances = cdist(grid_points, sample_coords, metric='euclidean')

        # Apply distance weighting
        weights = 1 / (distances ** power + 1e-10)  # Add small value to avoid division by zero

        # Apply max distance threshold if specified
        if max_distance is not None:
            weights[distances > max_distance] = 0

        # Normalize weights
        weights_sum = weights.sum(axis=1, keepdims=True)
        weights_norm = weights / (weights_sum + 1e-10)

        # Interpolate
        interpolated_values = (weights_norm * sample_values).sum(axis=1)
        interpolated_grid = interpolated_values.reshape(self.grid_x.shape)

        self.interpolated['idw'] = interpolated_grid

        print(f"✓ IDW interpolation complete")
        print(f"✓ Interpolated range: [{interpolated_grid.min():.2f}, {interpolated_grid.max():.2f}]")

        return interpolated_grid

    def radial_basis_function(self, function='multiquadric', epsilon=None):
        """Radial Basis Function (RBF) interpolation"""
        print("\n" + "="*60)
        print(f"RADIAL BASIS FUNCTION ({function})")
        print("="*60)

        x = self.sample_points['x'].values
        y = self.sample_points['y'].values
        values = self.sample_points['value'].values

        # Create RBF interpolator
        rbf = Rbf(x, y, values, function=function, epsilon=epsilon)

        # Interpolate on grid
        interpolated_grid = rbf(self.grid_x, self.grid_y)

        self.interpolated['rbf'] = interpolated_grid

        print(f"✓ RBF interpolation complete")
        print(f"✓ Interpolated range: [{interpolated_grid.min():.2f}, {interpolated_grid.max():.2f}]")

        return interpolated_grid

    def kriging_interpolation(self, nugget=0.1, sill=1.0, range_param=20):
        """Ordinary Kriging using Gaussian Process"""
        print("\n" + "="*60)
        print("KRIGING INTERPOLATION")
        print("="*60)

        X_train = self.sample_points[['x', 'y']].values
        y_train = self.sample_points['value'].values

        # Define kernel (variogram model)
        kernel = C(sill, (1e-3, 1e3)) * RBF(range_param, (1e-2, 1e2))

        # Create Gaussian Process
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                     alpha=nugget, normalize_y=True)

        # Fit model
        gp.fit(X_train, y_train)

        # Predict on grid
        grid_points = np.c_[self.grid_x.ravel(), self.grid_y.ravel()]
        interpolated_values, std = gp.predict(grid_points, return_std=True)

        interpolated_grid = interpolated_values.reshape(self.grid_x.shape)
        std_grid = std.reshape(self.grid_x.shape)

        self.interpolated['kriging'] = interpolated_grid
        self.interpolated['kriging_std'] = std_grid

        print(f"✓ Kriging interpolation complete")
        print(f"✓ Interpolated range: [{interpolated_grid.min():.2f}, {interpolated_grid.max():.2f}]")
        print(f"✓ Mean prediction uncertainty: {std.mean():.2f}")
        print(f"✓ Fitted kernel: {gp.kernel_}")

        return interpolated_grid, std_grid

    def natural_neighbor_interpolation(self):
        """Natural Neighbor interpolation"""
        print("\n" + "="*60)
        print("NATURAL NEIGHBOR INTERPOLATION")
        print("="*60)

        points = self.sample_points[['x', 'y']].values
        values = self.sample_points['value'].values

        grid_points = np.c_[self.grid_x.ravel(), self.grid_y.ravel()]

        # Use scipy's griddata with 'linear' method (natural neighbor approximation)
        interpolated_values = griddata(points, values, grid_points, method='linear')

        # Fill NaN values with nearest neighbor
        nan_mask = np.isnan(interpolated_values)
        if nan_mask.any():
            interpolated_values[nan_mask] = griddata(
                points, values, grid_points[nan_mask], method='nearest'
            )

        interpolated_grid = interpolated_values.reshape(self.grid_x.shape)

        self.interpolated['natural_neighbor'] = interpolated_grid

        print(f"✓ Natural neighbor interpolation complete")
        print(f"✓ Interpolated range: [{interpolated_grid.min():.2f}, {interpolated_grid.max():.2f}]")

        return interpolated_grid

    def spline_interpolation(self, smoothing=0):
        """Thin-plate spline interpolation"""
        print("\n" + "="*60)
        print("THIN-PLATE SPLINE INTERPOLATION")
        print("="*60)

        x = self.sample_points['x'].values
        y = self.sample_points['y'].values
        values = self.sample_points['value'].values

        # Use RBF with thin_plate function
        rbf = Rbf(x, y, values, function='thin_plate', smooth=smoothing)

        # Interpolate on grid
        interpolated_grid = rbf(self.grid_x, self.grid_y)

        self.interpolated['spline'] = interpolated_grid

        print(f"✓ Spline interpolation complete")
        print(f"✓ Interpolated range: [{interpolated_grid.min():.2f}, {interpolated_grid.max():.2f}]")

        return interpolated_grid

    def cross_validation(self, method='idw', k_folds=5):
        """Perform k-fold cross-validation"""
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION ({method.upper()}, {k_folds}-fold)")
        print("="*60)

        n_samples = len(self.sample_points)
        fold_size = n_samples // k_folds

        errors = []
        mae_scores = []
        rmse_scores = []

        for fold in range(k_folds):
            # Split data
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < k_folds - 1 else n_samples

            test_indices = list(range(test_start, test_end))
            train_indices = [i for i in range(n_samples) if i not in test_indices]

            train_data = self.sample_points.iloc[train_indices]
            test_data = self.sample_points.iloc[test_indices]

            # Perform interpolation on test points
            test_coords = test_data[['x', 'y']].values
            test_values = test_data['value'].values

            if method == 'idw':
                # IDW prediction
                train_coords = train_data[['x', 'y']].values
                train_values = train_data['value'].values

                distances = cdist(test_coords, train_coords, metric='euclidean')
                weights = 1 / (distances ** 2 + 1e-10)
                weights_norm = weights / weights.sum(axis=1, keepdims=True)
                predictions = (weights_norm * train_values).sum(axis=1)

            elif method == 'kriging':
                # Kriging prediction
                kernel = C(1.0) * RBF(20)
                gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, normalize_y=True)
                gp.fit(train_data[['x', 'y']].values, train_data['value'].values)
                predictions, _ = gp.predict(test_coords, return_std=True)

            # Calculate errors
            fold_errors = predictions - test_values
            errors.extend(fold_errors.tolist())

            mae = np.abs(fold_errors).mean()
            rmse = np.sqrt((fold_errors ** 2).mean())

            mae_scores.append(mae)
            rmse_scores.append(rmse)

        mean_mae = np.mean(mae_scores)
        mean_rmse = np.mean(rmse_scores)
        r2 = 1 - (np.var(errors) / np.var(self.sample_points['value']))

        print(f"\nCross-validation Results:")
        print(f"  Mean Absolute Error (MAE): {mean_mae:.4f}")
        print(f"  Root Mean Square Error (RMSE): {mean_rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")

        self.metrics[f'{method}_mae'] = mean_mae
        self.metrics[f'{method}_rmse'] = mean_rmse
        self.metrics[f'{method}_r2'] = r2

        return mean_mae, mean_rmse, r2

    def compare_methods(self):
        """Compare all interpolation methods"""
        print("\n" + "="*60)
        print("COMPARING INTERPOLATION METHODS")
        print("="*60)

        methods = ['idw', 'rbf', 'kriging', 'natural_neighbor', 'spline']

        for method in methods:
            if method in self.interpolated:
                grid = self.interpolated[method]
                print(f"\n{method.upper()}:")
                print(f"  Min: {grid.min():.2f}")
                print(f"  Max: {grid.max():.2f}")
                print(f"  Mean: {grid.mean():.2f}")
                print(f"  Std: {grid.std():.2f}")

        return self.interpolated

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(18, 14))

        # 1. Sample points
        ax1 = plt.subplot(2, 4, 1)
        scatter = ax1.scatter(self.sample_points['x'], self.sample_points['y'],
                            c=self.sample_points['value'], s=50,
                            cmap='RdYlBu_r', edgecolors='black', linewidths=1)
        plt.colorbar(scatter, ax=ax1, label='Value')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Sample Points', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. IDW
        ax2 = plt.subplot(2, 4, 2)
        if 'idw' in self.interpolated:
            contour = ax2.contourf(self.grid_x, self.grid_y, self.interpolated['idw'],
                                  levels=15, cmap='RdYlBu_r')
            plt.colorbar(contour, ax=ax2, label='Value')
            ax2.scatter(self.sample_points['x'], self.sample_points['y'],
                       c='black', s=10, alpha=0.5)
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title('IDW Interpolation', fontsize=12, fontweight='bold')

        # 3. RBF
        ax3 = plt.subplot(2, 4, 3)
        if 'rbf' in self.interpolated:
            contour = ax3.contourf(self.grid_x, self.grid_y, self.interpolated['rbf'],
                                  levels=15, cmap='RdYlBu_r')
            plt.colorbar(contour, ax=ax3, label='Value')
            ax3.scatter(self.sample_points['x'], self.sample_points['y'],
                       c='black', s=10, alpha=0.5)
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_title('RBF Interpolation', fontsize=12, fontweight='bold')

        # 4. Kriging
        ax4 = plt.subplot(2, 4, 4)
        if 'kriging' in self.interpolated:
            contour = ax4.contourf(self.grid_x, self.grid_y, self.interpolated['kriging'],
                                  levels=15, cmap='RdYlBu_r')
            plt.colorbar(contour, ax=ax4, label='Value')
            ax4.scatter(self.sample_points['x'], self.sample_points['y'],
                       c='black', s=10, alpha=0.5)
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Y Coordinate')
        ax4.set_title('Kriging Interpolation', fontsize=12, fontweight='bold')

        # 5. Natural Neighbor
        ax5 = plt.subplot(2, 4, 5)
        if 'natural_neighbor' in self.interpolated:
            contour = ax5.contourf(self.grid_x, self.grid_y, self.interpolated['natural_neighbor'],
                                  levels=15, cmap='RdYlBu_r')
            plt.colorbar(contour, ax=ax5, label='Value')
            ax5.scatter(self.sample_points['x'], self.sample_points['y'],
                       c='black', s=10, alpha=0.5)
        ax5.set_xlabel('X Coordinate')
        ax5.set_ylabel('Y Coordinate')
        ax5.set_title('Natural Neighbor', fontsize=12, fontweight='bold')

        # 6. Spline
        ax6 = plt.subplot(2, 4, 6)
        if 'spline' in self.interpolated:
            contour = ax6.contourf(self.grid_x, self.grid_y, self.interpolated['spline'],
                                  levels=15, cmap='RdYlBu_r')
            plt.colorbar(contour, ax=ax6, label='Value')
            ax6.scatter(self.sample_points['x'], self.sample_points['y'],
                       c='black', s=10, alpha=0.5)
        ax6.set_xlabel('X Coordinate')
        ax6.set_ylabel('Y Coordinate')
        ax6.set_title('Spline Interpolation', fontsize=12, fontweight='bold')

        # 7. Kriging uncertainty
        ax7 = plt.subplot(2, 4, 7)
        if 'kriging_std' in self.interpolated:
            contour = ax7.contourf(self.grid_x, self.grid_y, self.interpolated['kriging_std'],
                                  levels=15, cmap='YlOrRd')
            plt.colorbar(contour, ax=ax7, label='Std Dev')
            ax7.scatter(self.sample_points['x'], self.sample_points['y'],
                       c='black', s=10, alpha=0.5)
        ax7.set_xlabel('X Coordinate')
        ax7.set_ylabel('Y Coordinate')
        ax7.set_title('Kriging Uncertainty', fontsize=12, fontweight='bold')

        # 8. Cross-validation comparison
        ax8 = plt.subplot(2, 4, 8)
        methods = []
        rmse_values = []
        for method in ['idw', 'kriging']:
            if f'{method}_rmse' in self.metrics:
                methods.append(method.upper())
                rmse_values.append(self.metrics[f'{method}_rmse'])

        if methods:
            bars = ax8.bar(methods, rmse_values, color=['#3498db', '#2ecc71'],
                          edgecolor='black')
            ax8.set_ylabel('RMSE')
            ax8.set_title('Cross-Validation Comparison', fontsize=12, fontweight='bold')
            ax8.grid(True, alpha=0.3, axis='y')

            for bar in bars:
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('spatial_interpolation_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'spatial_interpolation_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("SPATIAL INTERPOLATION METHODS ANALYSIS")
    print("="*60)

    # Initialize interpolator
    interpolator = SpatialInterpolator()

    # Generate sample data
    interpolator.generate_sample_data(n_samples=100)

    # Create interpolation grid
    interpolator.create_interpolation_grid(grid_size=50)

    # Apply various interpolation methods
    interpolator.inverse_distance_weighting(power=2)
    interpolator.radial_basis_function(function='multiquadric')
    interpolator.kriging_interpolation(nugget=0.1, sill=1.0, range_param=20)
    interpolator.natural_neighbor_interpolation()
    interpolator.spline_interpolation(smoothing=0)

    # Cross-validation
    interpolator.cross_validation(method='idw', k_folds=5)
    interpolator.cross_validation(method='kriging', k_folds=5)

    # Compare methods
    interpolator.compare_methods()

    # Visualize results
    interpolator.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
