"""
Quadtree and Octree Spatial Partitioning - Geospatial Analysis
Implement quadtree for 2D spatial partitioning and octree concepts for hierarchical data

Dataset: Synthetic spatial points with varying density
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class QuadTreeNode:
    """Node in a quadtree structure"""

    def __init__(self, x_min, y_min, x_max, y_max, capacity=4):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.nw = None  # Northwest
        self.ne = None  # Northeast
        self.sw = None  # Southwest
        self.se = None  # Southeast

    def contains(self, x, y):
        """Check if point is within this node's boundary"""
        return (self.x_min <= x < self.x_max and
                self.y_min <= y < self.y_max)

    def subdivide(self):
        """Subdivide this node into four quadrants"""
        x_mid = (self.x_min + self.x_max) / 2
        y_mid = (self.y_min + self.y_max) / 2

        self.nw = QuadTreeNode(self.x_min, y_mid, x_mid, self.y_max, self.capacity)
        self.ne = QuadTreeNode(x_mid, y_mid, self.x_max, self.y_max, self.capacity)
        self.sw = QuadTreeNode(self.x_min, self.y_min, x_mid, y_mid, self.capacity)
        self.se = QuadTreeNode(x_mid, self.y_min, self.x_max, y_mid, self.capacity)

        self.divided = True

    def insert(self, x, y, data=None):
        """Insert a point into the quadtree"""
        if not self.contains(x, y):
            return False

        if len(self.points) < self.capacity and not self.divided:
            self.points.append({'x': x, 'y': y, 'data': data})
            return True

        if not self.divided:
            self.subdivide()

            # Redistribute existing points
            for point in self.points:
                self._insert_to_children(point['x'], point['y'], point['data'])
            self.points = []

        return self._insert_to_children(x, y, data)

    def _insert_to_children(self, x, y, data):
        """Insert point to appropriate child"""
        if self.nw.insert(x, y, data):
            return True
        if self.ne.insert(x, y, data):
            return True
        if self.sw.insert(x, y, data):
            return True
        if self.se.insert(x, y, data):
            return True
        return False

    def query_range(self, x_min, y_min, x_max, y_max):
        """Query all points within a range"""
        results = []

        # Check if range intersects this node
        if not self._intersects(x_min, y_min, x_max, y_max):
            return results

        # Add points in this node
        for point in self.points:
            if x_min <= point['x'] < x_max and y_min <= point['y'] < y_max:
                results.append(point)

        # Recursively query children
        if self.divided:
            results.extend(self.nw.query_range(x_min, y_min, x_max, y_max))
            results.extend(self.ne.query_range(x_min, y_min, x_max, y_max))
            results.extend(self.sw.query_range(x_min, y_min, x_max, y_max))
            results.extend(self.se.query_range(x_min, y_min, x_max, y_max))

        return results

    def _intersects(self, x_min, y_min, x_max, y_max):
        """Check if range intersects this node"""
        return not (x_max <= self.x_min or x_min >= self.x_max or
                   y_max <= self.y_min or y_min >= self.y_max)

    def get_all_nodes(self):
        """Get all nodes for visualization"""
        nodes = [self]
        if self.divided:
            nodes.extend(self.nw.get_all_nodes())
            nodes.extend(self.ne.get_all_nodes())
            nodes.extend(self.sw.get_all_nodes())
            nodes.extend(self.se.get_all_nodes())
        return nodes

    def count_points(self):
        """Count total points in this subtree"""
        count = len(self.points)
        if self.divided:
            count += self.nw.count_points()
            count += self.ne.count_points()
            count += self.sw.count_points()
            count += self.se.count_points()
        return count

    def get_depth(self):
        """Get maximum depth of this subtree"""
        if not self.divided:
            return 1
        return 1 + max(
            self.nw.get_depth(),
            self.ne.get_depth(),
            self.sw.get_depth(),
            self.se.get_depth()
        )


class QuadTree:
    """Quadtree spatial index"""

    def __init__(self, x_min, y_min, x_max, y_max, capacity=4):
        self.root = QuadTreeNode(x_min, y_min, x_max, y_max, capacity)
        self.size = 0

    def insert(self, x, y, data=None):
        """Insert point into quadtree"""
        if self.root.insert(x, y, data):
            self.size += 1
            return True
        return False

    def query_range(self, x_min, y_min, x_max, y_max):
        """Query points in range"""
        return self.root.query_range(x_min, y_min, x_max, y_max)

    def get_depth(self):
        """Get tree depth"""
        return self.root.get_depth()

    def get_all_nodes(self):
        """Get all nodes"""
        return self.root.get_all_nodes()


class SpatialPartitionAnalyzer:
    """Analyze spatial partitioning using quadtrees"""

    def __init__(self):
        self.points = None
        self.quadtree = None
        self.performance_metrics = {}

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

    def generate_spatial_data(self, n_points=2000, distribution='clustered'):
        """Generate synthetic spatial data"""
        print("="*60)
        print("GENERATING SPATIAL DATA")
        print("="*60)

        np.random.seed(42)

        data = []

        if distribution == 'clustered':
            # Generate clustered points
            n_clusters = 10
            points_per_cluster = n_points // n_clusters

            for i in range(n_clusters):
                center_x = np.random.uniform(0, 100)
                center_y = np.random.uniform(0, 100)
                cluster_std = np.random.uniform(2, 8)

                cluster_x = center_x + np.random.normal(0, cluster_std, points_per_cluster)
                cluster_y = center_y + np.random.normal(0, cluster_std, points_per_cluster)

                for x, y in zip(cluster_x, cluster_y):
                    if 0 <= x <= 100 and 0 <= y <= 100:
                        data.append({
                            'x': x,
                            'y': y,
                            'cluster': i,
                            'density': 1.0 / (cluster_std + 1)
                        })

        elif distribution == 'uniform':
            # Uniform random distribution
            for _ in range(n_points):
                data.append({
                    'x': np.random.uniform(0, 100),
                    'y': np.random.uniform(0, 100),
                    'cluster': -1,
                    'density': 1.0
                })

        elif distribution == 'gradient':
            # Density gradient
            for _ in range(n_points):
                x = np.random.uniform(0, 100)
                y = np.random.uniform(0, 100)

                # Higher density in bottom-left
                accept_prob = (100 - x) * (100 - y) / 10000
                if np.random.random() < accept_prob or len(data) < n_points // 4:
                    data.append({
                        'x': x,
                        'y': y,
                        'cluster': -1,
                        'density': accept_prob
                    })

        self.points = pd.DataFrame(data)

        print(f"✓ Generated {len(self.points)} spatial points")
        print(f"✓ Distribution type: {distribution}")
        print(f"✓ Spatial extent: X=[{self.points['x'].min():.2f}, {self.points['x'].max():.2f}], "
              f"Y=[{self.points['y'].min():.2f}, {self.points['y'].max():.2f}]")

        return self.points

    def build_quadtree(self, capacity=4):
        """Build quadtree index"""
        print("\n" + "="*60)
        print("BUILDING QUADTREE INDEX")
        print("="*60)

        start_time = time.time()

        # Create quadtree with slight padding
        x_min, x_max = self.points['x'].min() - 1, self.points['x'].max() + 1
        y_min, y_max = self.points['y'].min() - 1, self.points['y'].max() + 1

        self.quadtree = QuadTree(x_min, y_min, x_max, y_max, capacity=capacity)

        # Insert all points
        for idx, row in self.points.iterrows():
            self.quadtree.insert(row['x'], row['y'], {'id': idx, 'cluster': row['cluster']})

        build_time = time.time() - start_time

        print(f"✓ Quadtree built in {build_time:.4f} seconds")
        print(f"✓ Tree depth: {self.quadtree.get_depth()}")
        print(f"✓ Node capacity: {capacity}")
        print(f"✓ Total nodes: {len(self.quadtree.get_all_nodes())}")
        print(f"✓ Total points indexed: {self.quadtree.size}")

        self.performance_metrics['build_time'] = build_time
        self.performance_metrics['tree_depth'] = self.quadtree.get_depth()
        self.performance_metrics['num_nodes'] = len(self.quadtree.get_all_nodes())

        return self.quadtree

    def analyze_spatial_density(self):
        """Analyze spatial density using quadtree"""
        print("\n" + "="*60)
        print("ANALYZING SPATIAL DENSITY")
        print("="*60)

        nodes = self.quadtree.get_all_nodes()
        leaf_nodes = [n for n in nodes if not n.divided]

        densities = []
        areas = []

        for node in leaf_nodes:
            area = (node.x_max - node.x_min) * (node.y_max - node.y_min)
            point_count = node.count_points()
            density = point_count / area if area > 0 else 0

            densities.append(density)
            areas.append(area)

        densities = np.array(densities)
        areas = np.array(areas)

        print(f"\nLeaf Node Statistics:")
        print(f"  Number of leaf nodes: {len(leaf_nodes)}")
        print(f"  Average points per leaf: {np.mean([n.count_points() for n in leaf_nodes]):.2f}")
        print(f"  Min/Max points per leaf: {min([n.count_points() for n in leaf_nodes])}/{max([n.count_points() for n in leaf_nodes])}")
        print(f"\nDensity Statistics:")
        print(f"  Mean density: {densities.mean():.4f} points/unit²")
        print(f"  Std density: {densities.std():.4f}")
        print(f"  Min/Max density: {densities.min():.4f}/{densities.max():.4f}")

        self.performance_metrics['mean_density'] = densities.mean()
        self.performance_metrics['std_density'] = densities.std()

        return densities, areas

    def benchmark_range_queries(self, n_queries=100):
        """Benchmark range query performance"""
        print("\n" + "="*60)
        print("BENCHMARKING RANGE QUERIES")
        print("="*60)

        np.random.seed(123)

        # Generate random query ranges
        query_ranges = []
        for _ in range(n_queries):
            x_center = np.random.uniform(0, 100)
            y_center = np.random.uniform(0, 100)
            width = np.random.uniform(5, 20)
            height = np.random.uniform(5, 20)

            query_ranges.append((
                x_center - width/2,
                y_center - height/2,
                x_center + width/2,
                y_center + height/2
            ))

        # Benchmark quadtree
        start_time = time.time()
        quadtree_results = []
        for x_min, y_min, x_max, y_max in query_ranges:
            results = self.quadtree.query_range(x_min, y_min, x_max, y_max)
            quadtree_results.append(len(results))
        quadtree_time = time.time() - start_time

        # Benchmark brute force
        start_time = time.time()
        brute_results = []
        for x_min, y_min, x_max, y_max in query_ranges:
            results = self.points[
                (self.points['x'] >= x_min) &
                (self.points['x'] < x_max) &
                (self.points['y'] >= y_min) &
                (self.points['y'] < y_max)
            ]
            brute_results.append(len(results))
        brute_time = time.time() - start_time

        print(f"\nQuadtree range queries:")
        print(f"  Total time: {quadtree_time:.4f} seconds")
        print(f"  Average per query: {quadtree_time/n_queries*1000:.2f} ms")
        print(f"  Average results: {np.mean(quadtree_results):.1f} points")

        print(f"\nBrute force range queries:")
        print(f"  Total time: {brute_time:.4f} seconds")
        print(f"  Average per query: {brute_time/n_queries*1000:.2f} ms")
        print(f"  Speedup: {brute_time/quadtree_time:.2f}x")

        # Verify correctness
        matches = sum(1 for q, b in zip(quadtree_results, brute_results) if q == b)
        print(f"\nCorrectness: {matches}/{n_queries} queries matched ({100*matches/n_queries:.1f}%)")

        self.performance_metrics['quadtree_time'] = quadtree_time
        self.performance_metrics['brute_time'] = brute_time
        self.performance_metrics['speedup'] = brute_time / quadtree_time

        return quadtree_time, brute_time

    def adaptive_subdivision_analysis(self):
        """Analyze adaptive subdivision behavior"""
        print("\n" + "="*60)
        print("ADAPTIVE SUBDIVISION ANALYSIS")
        print("="*60)

        nodes = self.quadtree.get_all_nodes()

        depth_distribution = {}
        for node in nodes:
            depth = 0
            current = node
            parent = getattr(current, 'parent', None)
            while parent is not None:
                depth += 1
                current = parent
                parent = getattr(current, 'parent', None)

            depth_distribution[depth] = depth_distribution.get(depth, 0) + 1

        print("\nNode Distribution by Depth:")
        for depth in sorted(depth_distribution.keys()):
            print(f"  Depth {depth}: {depth_distribution[depth]} nodes")

        # Analyze subdivision patterns
        leaf_nodes = [n for n in nodes if not n.divided]
        subdivided_nodes = [n for n in nodes if n.divided]

        print(f"\nSubdivision Statistics:")
        print(f"  Total nodes: {len(nodes)}")
        print(f"  Leaf nodes: {len(leaf_nodes)} ({100*len(leaf_nodes)/len(nodes):.1f}%)")
        print(f"  Subdivided nodes: {len(subdivided_nodes)} ({100*len(subdivided_nodes)/len(nodes):.1f}%)")

        return depth_distribution

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(18, 12))

        # 1. Points with quadtree overlay
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(self.points['x'], self.points['y'], c='blue', s=5, alpha=0.6, zorder=5)

        # Draw quadtree boundaries
        nodes = self.quadtree.get_all_nodes()
        for node in nodes:
            if not node.divided:
                rect = Rectangle((node.x_min, node.y_min),
                               node.x_max - node.x_min,
                               node.y_max - node.y_min,
                               fill=False, edgecolor='red', linewidth=0.5, alpha=0.7)
                ax1.add_patch(rect)

        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Quadtree Partitioning', fontsize=12, fontweight='bold')
        ax1.set_xlim(self.points['x'].min() - 5, self.points['x'].max() + 5)
        ax1.set_ylim(self.points['y'].min() - 5, self.points['y'].max() + 5)

        # 2. Density heatmap
        ax2 = plt.subplot(2, 3, 2)
        nodes = [n for n in self.quadtree.get_all_nodes() if not n.divided]

        for node in nodes:
            area = (node.x_max - node.x_min) * (node.y_max - node.y_min)
            density = node.count_points() / area if area > 0 else 0

            color_intensity = min(density / 5.0, 1.0)  # Normalize
            rect = Rectangle((node.x_min, node.y_min),
                           node.x_max - node.x_min,
                           node.y_max - node.y_min,
                           fill=True, facecolor=plt.cm.YlOrRd(color_intensity),
                           edgecolor='black', linewidth=0.5, alpha=0.7)
            ax2.add_patch(rect)

        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title('Spatial Density Heatmap', fontsize=12, fontweight='bold')
        ax2.set_xlim(self.points['x'].min() - 5, self.points['x'].max() + 5)
        ax2.set_ylim(self.points['y'].min() - 5, self.points['y'].max() + 5)

        # 3. Sample range query
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(self.points['x'], self.points['y'], c='lightgray', s=5, alpha=0.4)

        # Execute sample query
        query_x_min, query_y_min = 30, 30
        query_x_max, query_y_max = 60, 60
        query_results = self.quadtree.query_range(query_x_min, query_y_min, query_x_max, query_y_max)

        rect = Rectangle((query_x_min, query_y_min),
                        query_x_max - query_x_min,
                        query_y_max - query_y_min,
                        fill=False, edgecolor='red', linewidth=2)
        ax3.add_patch(rect)

        if query_results:
            result_x = [r['x'] for r in query_results]
            result_y = [r['y'] for r in query_results]
            ax3.scatter(result_x, result_y, c='red', s=20, zorder=5,
                       label=f'{len(query_results)} points found')

        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_title('Range Query Example', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.set_xlim(0, 100)
        ax3.set_ylim(0, 100)

        # 4. Performance comparison
        ax4 = plt.subplot(2, 3, 4)
        methods = ['Quadtree', 'Brute Force']
        times = [
            self.performance_metrics['quadtree_time'] * 1000,
            self.performance_metrics['brute_time'] * 1000
        ]
        bars = ax4.bar(methods, times, color=['#3498db', '#e74c3c'], edgecolor='black')
        ax4.set_ylabel('Query Time (ms)')
        ax4.set_title('Range Query Performance', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}ms', ha='center', va='bottom')

        # 5. Tree statistics
        ax5 = plt.subplot(2, 3, 5)
        stats = {
            'Tree Depth': self.performance_metrics['tree_depth'],
            'Num Nodes': self.performance_metrics['num_nodes'],
            'Build Time (ms)': self.performance_metrics['build_time'] * 1000
        }

        y_pos = np.arange(len(stats))
        bars = ax5.barh(y_pos, list(stats.values()), color=['#2ecc71', '#9b59b6', '#f39c12'],
                       edgecolor='black')
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(stats.keys())
        ax5.set_xlabel('Value')
        ax5.set_title('Quadtree Statistics', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='x')

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax5.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}', ha='left', va='center', fontweight='bold')

        # 6. Speedup analysis
        ax6 = plt.subplot(2, 3, 6)
        speedup = self.performance_metrics['speedup']
        bars = ax6.bar(['Speedup Factor'], [speedup], color='#1abc9c', edgecolor='black')
        ax6.set_ylabel('Speedup (vs Brute Force)')
        ax6.set_title('Performance Improvement', fontsize=12, fontweight='bold')
        ax6.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=14)

        plt.tight_layout()
        plt.savefig('quadtree_spatial_partitioning.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'quadtree_spatial_partitioning.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("QUADTREE SPATIAL PARTITIONING ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = SpatialPartitionAnalyzer()

    # Generate spatial data
    analyzer.generate_spatial_data(n_points=2000, distribution='clustered')

    # Build quadtree
    analyzer.build_quadtree(capacity=4)

    # Analyze density
    analyzer.analyze_spatial_density()

    # Benchmark queries
    analyzer.benchmark_range_queries(n_queries=100)

    # Adaptive subdivision analysis
    analyzer.adaptive_subdivision_analysis()

    # Visualize results
    analyzer.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
