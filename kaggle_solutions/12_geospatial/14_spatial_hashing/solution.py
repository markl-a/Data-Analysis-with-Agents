"""
Spatial Hashing and Grid-Based Indexing - Geospatial Analysis
Implement spatial hashing for efficient collision detection and proximity queries

Dataset: Synthetic moving objects and static points
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class SpatialHash:
    """Spatial hash grid for efficient spatial queries"""

    def __init__(self, cell_size=5.0, bounds=(0, 0, 100, 100)):
        self.cell_size = cell_size
        self.bounds = bounds
        self.grid = defaultdict(list)
        self.object_cells = {}  # Track which cells each object is in

    def _get_cell_coords(self, x, y):
        """Convert world coordinates to cell coordinates"""
        cell_x = int(x / self.cell_size)
        cell_y = int(y / self.cell_size)
        return (cell_x, cell_y)

    def _get_cells_for_region(self, x_min, y_min, x_max, y_max):
        """Get all cells that intersect with a region"""
        cell_x_min = int(x_min / self.cell_size)
        cell_y_min = int(y_min / self.cell_size)
        cell_x_max = int(x_max / self.cell_size)
        cell_y_max = int(y_max / self.cell_size)

        cells = []
        for cx in range(cell_x_min, cell_x_max + 1):
            for cy in range(cell_y_min, cell_y_max + 1):
                cells.append((cx, cy))

        return cells

    def insert(self, obj_id, x, y, radius=0):
        """Insert object into spatial hash"""
        # For objects with radius, insert into all overlapping cells
        if radius > 0:
            cells = self._get_cells_for_region(
                x - radius, y - radius,
                x + radius, y + radius
            )
        else:
            cells = [self._get_cell_coords(x, y)]

        self.object_cells[obj_id] = cells

        for cell in cells:
            self.grid[cell].append({
                'id': obj_id,
                'x': x,
                'y': y,
                'radius': radius
            })

    def remove(self, obj_id):
        """Remove object from spatial hash"""
        if obj_id in self.object_cells:
            for cell in self.object_cells[obj_id]:
                self.grid[cell] = [obj for obj in self.grid[cell] if obj['id'] != obj_id]
            del self.object_cells[obj_id]

    def update(self, obj_id, x, y, radius=0):
        """Update object position"""
        self.remove(obj_id)
        self.insert(obj_id, x, y, radius)

    def query_point(self, x, y):
        """Query all objects in the same cell as point"""
        cell = self._get_cell_coords(x, y)
        return self.grid.get(cell, [])

    def query_range(self, x_min, y_min, x_max, y_max):
        """Query all objects in a rectangular region"""
        cells = self._get_cells_for_region(x_min, y_min, x_max, y_max)

        results = []
        seen = set()

        for cell in cells:
            for obj in self.grid.get(cell, []):
                if obj['id'] not in seen:
                    # Check if object actually intersects the region
                    if (x_min <= obj['x'] + obj['radius'] and
                        obj['x'] - obj['radius'] <= x_max and
                        y_min <= obj['y'] + obj['radius'] and
                        obj['y'] - obj['radius'] <= y_max):

                        results.append(obj)
                        seen.add(obj['id'])

        return results

    def query_radius(self, x, y, radius):
        """Query all objects within radius of point"""
        # Query a square region first
        candidates = self.query_range(
            x - radius, y - radius,
            x + radius, y + radius
        )

        # Filter to actual radius
        results = []
        for obj in candidates:
            dist_sq = (obj['x'] - x)**2 + (obj['y'] - y)**2
            if dist_sq <= radius**2:
                results.append(obj)

        return results

    def detect_collisions(self):
        """Detect all colliding object pairs"""
        collisions = []
        checked_pairs = set()

        for cell, objects in self.grid.items():
            # Check all pairs within this cell
            for i, obj1 in enumerate(objects):
                for j in range(i + 1, len(objects)):
                    obj2 = objects[j]

                    pair = tuple(sorted([obj1['id'], obj2['id']]))
                    if pair in checked_pairs:
                        continue

                    checked_pairs.add(pair)

                    # Check collision
                    dist_sq = (obj1['x'] - obj2['x'])**2 + (obj1['y'] - obj2['y'])**2
                    min_dist = obj1['radius'] + obj2['radius']

                    if dist_sq <= min_dist**2:
                        collisions.append((obj1['id'], obj2['id'], np.sqrt(dist_sq)))

        return collisions

    def get_grid_stats(self):
        """Get grid statistics"""
        total_cells = len(self.grid)
        occupied_cells = sum(1 for cell in self.grid.values() if len(cell) > 0)
        objects_per_cell = [len(cell) for cell in self.grid.values()]

        return {
            'total_cells': total_cells,
            'occupied_cells': occupied_cells,
            'avg_objects_per_cell': np.mean(objects_per_cell) if objects_per_cell else 0,
            'max_objects_per_cell': max(objects_per_cell) if objects_per_cell else 0,
            'total_objects': len(self.object_cells)
        }


class SpatialHashAnalyzer:
    """Analyze spatial hashing performance"""

    def __init__(self):
        self.points = None
        self.spatial_hash = None
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

    def generate_spatial_data(self, n_points=1000):
        """Generate synthetic spatial data"""
        print("="*60)
        print("GENERATING SPATIAL DATA")
        print("="*60)

        np.random.seed(42)

        data = []

        # Generate clustered points
        n_clusters = 8
        points_per_cluster = n_points // n_clusters

        for i in range(n_clusters):
            center_x = np.random.uniform(10, 90)
            center_y = np.random.uniform(10, 90)

            for j in range(points_per_cluster):
                x = center_x + np.random.normal(0, 5)
                y = center_y + np.random.normal(0, 5)

                data.append({
                    'id': len(data),
                    'x': np.clip(x, 0, 100),
                    'y': np.clip(y, 0, 100),
                    'radius': np.random.uniform(0.5, 2.5),
                    'cluster': i,
                    'velocity_x': np.random.uniform(-2, 2),
                    'velocity_y': np.random.uniform(-2, 2)
                })

        self.points = pd.DataFrame(data)

        print(f"✓ Generated {len(self.points)} spatial objects")
        print(f"✓ Number of clusters: {n_clusters}")
        print(f"✓ Average radius: {self.points['radius'].mean():.2f}")
        print(f"✓ Spatial extent: 100 x 100 units")

        return self.points

    def build_spatial_hash(self, cell_size=5.0):
        """Build spatial hash grid"""
        print("\n" + "="*60)
        print("BUILDING SPATIAL HASH")
        print("="*60)

        start_time = time.time()

        self.spatial_hash = SpatialHash(cell_size=cell_size, bounds=(0, 0, 100, 100))

        for _, row in self.points.iterrows():
            self.spatial_hash.insert(
                obj_id=row['id'],
                x=row['x'],
                y=row['y'],
                radius=row['radius']
            )

        build_time = time.time() - start_time

        stats = self.spatial_hash.get_grid_stats()

        print(f"✓ Spatial hash built in {build_time:.4f} seconds")
        print(f"✓ Cell size: {cell_size}")
        print(f"✓ Occupied cells: {stats['occupied_cells']}")
        print(f"✓ Average objects per cell: {stats['avg_objects_per_cell']:.2f}")
        print(f"✓ Max objects per cell: {stats['max_objects_per_cell']}")

        self.metrics['build_time'] = build_time
        self.metrics['cell_size'] = cell_size
        self.metrics.update(stats)

        return self.spatial_hash

    def benchmark_range_queries(self, n_queries=100):
        """Benchmark range query performance"""
        print("\n" + "="*60)
        print("BENCHMARKING RANGE QUERIES")
        print("="*60)

        np.random.seed(123)

        # Generate random query ranges
        query_ranges = []
        for _ in range(n_queries):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            width = np.random.uniform(5, 15)
            height = np.random.uniform(5, 15)

            query_ranges.append((
                x - width/2, y - height/2,
                x + width/2, y + height/2
            ))

        # Benchmark spatial hash
        start_time = time.time()
        hash_results = []
        for x_min, y_min, x_max, y_max in query_ranges:
            results = self.spatial_hash.query_range(x_min, y_min, x_max, y_max)
            hash_results.append(len(results))
        hash_time = time.time() - start_time

        # Benchmark brute force
        start_time = time.time()
        brute_results = []
        for x_min, y_min, x_max, y_max in query_ranges:
            results = self.points[
                (self.points['x'] - self.points['radius'] <= x_max) &
                (self.points['x'] + self.points['radius'] >= x_min) &
                (self.points['y'] - self.points['radius'] <= y_max) &
                (self.points['y'] + self.points['radius'] >= y_min)
            ]
            brute_results.append(len(results))
        brute_time = time.time() - start_time

        print(f"\nSpatial hash range queries:")
        print(f"  Total time: {hash_time:.4f} seconds")
        print(f"  Average per query: {hash_time/n_queries*1000:.2f} ms")
        print(f"  Average results: {np.mean(hash_results):.1f} objects")

        print(f"\nBrute force range queries:")
        print(f"  Total time: {brute_time:.4f} seconds")
        print(f"  Average per query: {brute_time/n_queries*1000:.2f} ms")
        print(f"  Speedup: {brute_time/hash_time:.2f}x")

        self.metrics['hash_range_time'] = hash_time
        self.metrics['brute_range_time'] = brute_time
        self.metrics['range_speedup'] = brute_time / hash_time

        return hash_time, brute_time

    def benchmark_radius_queries(self, n_queries=100, radius=10):
        """Benchmark radius query performance"""
        print("\n" + "="*60)
        print("BENCHMARKING RADIUS QUERIES")
        print("="*60)

        np.random.seed(456)

        # Generate random query points
        query_points = [(np.random.uniform(0, 100), np.random.uniform(0, 100))
                       for _ in range(n_queries)]

        # Benchmark spatial hash
        start_time = time.time()
        hash_results = []
        for x, y in query_points:
            results = self.spatial_hash.query_radius(x, y, radius)
            hash_results.append(len(results))
        hash_time = time.time() - start_time

        # Benchmark brute force
        start_time = time.time()
        brute_results = []
        for x, y in query_points:
            distances = np.sqrt(
                (self.points['x'] - x)**2 +
                (self.points['y'] - y)**2
            )
            results = self.points[distances <= radius]
            brute_results.append(len(results))
        brute_time = time.time() - start_time

        print(f"\nSpatial hash radius queries (r={radius}):")
        print(f"  Total time: {hash_time:.4f} seconds")
        print(f"  Average per query: {hash_time/n_queries*1000:.2f} ms")
        print(f"  Average results: {np.mean(hash_results):.1f} objects")

        print(f"\nBrute force radius queries:")
        print(f"  Total time: {brute_time:.4f} seconds")
        print(f"  Average per query: {brute_time/n_queries*1000:.2f} ms")
        print(f"  Speedup: {brute_time/hash_time:.2f}x")

        self.metrics['hash_radius_time'] = hash_time
        self.metrics['brute_radius_time'] = brute_time
        self.metrics['radius_speedup'] = brute_time / hash_time

        return hash_time, brute_time

    def detect_and_analyze_collisions(self):
        """Detect and analyze collisions"""
        print("\n" + "="*60)
        print("COLLISION DETECTION")
        print("="*60)

        start_time = time.time()
        collisions = self.spatial_hash.detect_collisions()
        hash_time = time.time() - start_time

        print(f"\nSpatial hash collision detection:")
        print(f"  Time: {hash_time:.4f} seconds")
        print(f"  Collisions found: {len(collisions)}")

        if collisions:
            distances = [c[2] for c in collisions]
            print(f"  Average overlap distance: {np.mean(distances):.2f}")
            print(f"  Min/Max overlap: {np.min(distances):.2f}/{np.max(distances):.2f}")

        # Brute force comparison
        start_time = time.time()
        brute_collisions = []
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                obj1 = self.points.iloc[i]
                obj2 = self.points.iloc[j]

                dist = self.euclidean_distance(obj1['x'], obj1['y'], obj2['x'], obj2['y'])
                min_dist = obj1['radius'] + obj2['radius']

                if dist <= min_dist:
                    brute_collisions.append((obj1['id'], obj2['id'], dist))
        brute_time = time.time() - start_time

        print(f"\nBrute force collision detection:")
        print(f"  Time: {brute_time:.4f} seconds")
        print(f"  Collisions found: {len(brute_collisions)}")
        print(f"  Speedup: {brute_time/hash_time:.2f}x")

        self.metrics['collisions'] = len(collisions)
        self.metrics['collision_time'] = hash_time
        self.metrics['collision_speedup'] = brute_time / hash_time

        return collisions

    def simulate_movement(self, n_steps=10):
        """Simulate object movement and track updates"""
        print("\n" + "="*60)
        print("SIMULATING OBJECT MOVEMENT")
        print("="*60)

        update_times = []

        for step in range(n_steps):
            start_time = time.time()

            # Update each object position
            for idx, row in self.points.iterrows():
                new_x = row['x'] + row['velocity_x']
                new_y = row['y'] + row['velocity_y']

                # Bounce off boundaries
                if new_x < 0 or new_x > 100:
                    self.points.at[idx, 'velocity_x'] *= -1
                    new_x = np.clip(new_x, 0, 100)

                if new_y < 0 or new_y > 100:
                    self.points.at[idx, 'velocity_y'] *= -1
                    new_y = np.clip(new_y, 0, 100)

                self.points.at[idx, 'x'] = new_x
                self.points.at[idx, 'y'] = new_y

                # Update spatial hash
                self.spatial_hash.update(row['id'], new_x, new_y, row['radius'])

            update_time = time.time() - start_time
            update_times.append(update_time)

        print(f"\nMovement simulation ({n_steps} steps):")
        print(f"  Average update time: {np.mean(update_times)*1000:.2f} ms/step")
        print(f"  Total simulation time: {sum(update_times):.4f} seconds")

        self.metrics['avg_update_time'] = np.mean(update_times)

        return update_times

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(18, 12))

        # 1. Spatial hash grid visualization
        ax1 = plt.subplot(2, 3, 1)

        # Draw grid
        cell_size = self.metrics['cell_size']
        for x in np.arange(0, 100, cell_size):
            ax1.axvline(x, color='gray', linewidth=0.5, alpha=0.3)
        for y in np.arange(0, 100, cell_size):
            ax1.axhline(y, color='gray', linewidth=0.5, alpha=0.3)

        # Draw objects
        for _, row in self.points.iterrows():
            circle = plt.Circle((row['x'], row['y']), row['radius'],
                              color='blue', alpha=0.5, edgecolor='black', linewidth=0.5)
            ax1.add_patch(circle)

        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title(f'Spatial Hash Grid (cell size={cell_size})', fontsize=12, fontweight='bold')
        ax1.set_aspect('equal')

        # 2. Cell occupancy heatmap
        ax2 = plt.subplot(2, 3, 2)

        # Count objects per cell
        max_cells_x = int(100 / cell_size) + 1
        max_cells_y = int(100 / cell_size) + 1
        grid_counts = np.zeros((max_cells_y, max_cells_x))

        for cell, objects in self.spatial_hash.grid.items():
            if 0 <= cell[0] < max_cells_x and 0 <= cell[1] < max_cells_y:
                grid_counts[cell[1], cell[0]] = len(objects)

        im = ax2.imshow(grid_counts, cmap='YlOrRd', origin='lower',
                       extent=[0, 100, 0, 100], interpolation='nearest')
        plt.colorbar(im, ax=ax2, label='Objects per Cell')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title('Cell Occupancy Heatmap', fontsize=12, fontweight='bold')

        # 3. Range query performance
        ax3 = plt.subplot(2, 3, 3)
        methods = ['Spatial Hash', 'Brute Force']
        times = [
            self.metrics['hash_range_time'] * 1000,
            self.metrics['brute_range_time'] * 1000
        ]
        bars = ax3.bar(methods, times, color=['#3498db', '#e74c3c'], edgecolor='black')
        ax3.set_ylabel('Query Time (ms)')
        ax3.set_title('Range Query Performance', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}ms', ha='center', va='bottom')

        # 4. Radius query performance
        ax4 = plt.subplot(2, 3, 4)
        times = [
            self.metrics['hash_radius_time'] * 1000,
            self.metrics['brute_radius_time'] * 1000
        ]
        bars = ax4.bar(methods, times, color=['#2ecc71', '#e67e22'], edgecolor='black')
        ax4.set_ylabel('Query Time (ms)')
        ax4.set_title('Radius Query Performance', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}ms', ha='center', va='bottom')

        # 5. Speedup comparison
        ax5 = plt.subplot(2, 3, 5)
        speedups = {
            'Range\nQueries': self.metrics['range_speedup'],
            'Radius\nQueries': self.metrics['radius_speedup'],
            'Collision\nDetection': self.metrics['collision_speedup']
        }
        bars = ax5.bar(range(len(speedups)), list(speedups.values()),
                      color=['#9b59b6', '#1abc9c', '#f39c12'], edgecolor='black')
        ax5.set_xticks(range(len(speedups)))
        ax5.set_xticklabels(speedups.keys())
        ax5.set_ylabel('Speedup Factor (vs Brute Force)')
        ax5.set_title('Performance Speedup Analysis', fontsize=12, fontweight='bold')
        ax5.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax5.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}x', ha='center', va='bottom', fontweight='bold')

        # 6. Grid statistics
        ax6 = plt.subplot(2, 3, 6)
        stats = {
            'Occupied\nCells': self.metrics['occupied_cells'],
            'Avg Objects\nper Cell': self.metrics['avg_objects_per_cell'],
            'Max Objects\nper Cell': self.metrics['max_objects_per_cell']
        }
        bars = ax6.bar(range(len(stats)), list(stats.values()),
                      color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
        ax6.set_xticks(range(len(stats)))
        ax6.set_xticklabels(stats.keys(), fontsize=9)
        ax6.set_ylabel('Count')
        ax6.set_title('Grid Statistics', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('spatial_hashing_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'spatial_hashing_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("SPATIAL HASHING AND GRID-BASED INDEXING ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = SpatialHashAnalyzer()

    # Generate data
    analyzer.generate_spatial_data(n_points=1000)

    # Build spatial hash
    analyzer.build_spatial_hash(cell_size=5.0)

    # Benchmark queries
    analyzer.benchmark_range_queries(n_queries=100)
    analyzer.benchmark_radius_queries(n_queries=100, radius=10)

    # Collision detection
    analyzer.detect_and_analyze_collisions()

    # Simulate movement
    analyzer.simulate_movement(n_steps=10)

    # Visualize results
    analyzer.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
