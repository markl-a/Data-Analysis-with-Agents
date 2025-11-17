"""
R-tree Spatial Indexing - Geospatial Analysis
Implement R-tree spatial indexing for efficient spatial queries and nearest neighbor search

Dataset: Synthetic geospatial data with points of interest
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.neighbors import KDTree
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class RTreeNode:
    """Node in R-tree structure"""

    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.entries = []  # (bbox, data/child_node)
        self.parent = None

    def get_bbox(self):
        """Calculate bounding box of all entries"""
        if not self.entries:
            return None

        min_x = min(e[0][0] for e in self.entries)
        min_y = min(e[0][1] for e in self.entries)
        max_x = max(e[0][2] for e in self.entries)
        max_y = max(e[0][3] for e in self.entries)

        return (min_x, min_y, max_x, max_y)


class RTree:
    """R-tree spatial index implementation"""

    def __init__(self, max_entries=4, min_entries=2):
        self.root = RTreeNode(is_leaf=True)
        self.max_entries = max_entries
        self.min_entries = min_entries
        self.size = 0

    def insert(self, point, data):
        """Insert a point into the R-tree"""
        x, y = point
        bbox = (x, y, x, y)

        # Find leaf to insert into
        leaf = self._choose_leaf(self.root, bbox)

        # Add entry to leaf
        leaf.entries.append((bbox, data))
        self.size += 1

        # Split if necessary
        if len(leaf.entries) > self.max_entries:
            self._split_node(leaf)

    def _choose_leaf(self, node, bbox):
        """Choose leaf node for insertion"""
        if node.is_leaf:
            return node

        # Choose child with minimum area enlargement
        best_child = None
        min_enlargement = float('inf')

        for entry_bbox, child in node.entries:
            enlarged = self._enlarge_bbox(entry_bbox, bbox)
            enlargement = self._bbox_area(enlarged) - self._bbox_area(entry_bbox)

            if enlargement < min_enlargement:
                min_enlargement = enlargement
                best_child = child

        return self._choose_leaf(best_child, bbox)

    def _split_node(self, node):
        """Split node using quadratic split algorithm"""
        entries = node.entries

        # Find two seed entries with maximum separation
        max_waste = -1
        seed1_idx, seed2_idx = 0, 1

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                bbox1, bbox2 = entries[i][0], entries[j][0]
                combined = self._enlarge_bbox(bbox1, bbox2)
                waste = (self._bbox_area(combined) -
                        self._bbox_area(bbox1) -
                        self._bbox_area(bbox2))

                if waste > max_waste:
                    max_waste = waste
                    seed1_idx, seed2_idx = i, j

        # Create two new nodes
        node1 = RTreeNode(is_leaf=node.is_leaf)
        node2 = RTreeNode(is_leaf=node.is_leaf)

        node1.entries = [entries[seed1_idx]]
        node2.entries = [entries[seed2_idx]]

        # Distribute remaining entries
        remaining = [entries[i] for i in range(len(entries))
                    if i not in [seed1_idx, seed2_idx]]

        for entry in remaining:
            bbox1 = node1.get_bbox()
            bbox2 = node2.get_bbox()

            enlargement1 = (self._bbox_area(self._enlarge_bbox(bbox1, entry[0])) -
                          self._bbox_area(bbox1))
            enlargement2 = (self._bbox_area(self._enlarge_bbox(bbox2, entry[0])) -
                          self._bbox_area(bbox2))

            if enlargement1 < enlargement2:
                node1.entries.append(entry)
            else:
                node2.entries.append(entry)

        # Update parent
        if node == self.root:
            new_root = RTreeNode(is_leaf=False)
            new_root.entries = [
                (node1.get_bbox(), node1),
                (node2.get_bbox(), node2)
            ]
            self.root = new_root
        else:
            parent = node.parent
            parent.entries.remove((node.get_bbox(), node))
            parent.entries.append((node1.get_bbox(), node1))
            parent.entries.append((node2.get_bbox(), node2))

            if len(parent.entries) > self.max_entries:
                self._split_node(parent)

    def range_query(self, query_bbox):
        """Find all points within a bounding box"""
        results = []
        self._range_search(self.root, query_bbox, results)
        return results

    def _range_search(self, node, query_bbox, results):
        """Recursive range search"""
        if node.is_leaf:
            for bbox, data in node.entries:
                if self._bbox_intersects(bbox, query_bbox):
                    results.append(data)
        else:
            for bbox, child in node.entries:
                if self._bbox_intersects(bbox, query_bbox):
                    self._range_search(child, query_bbox, results)

    def nearest_neighbor(self, point, k=1):
        """Find k nearest neighbors using best-first search"""
        x, y = point

        # Priority queue: (distance, node/entry)
        queue = [(0, self.root)]
        results = []

        while queue and len(results) < k:
            dist, item = queue.pop(0)

            if isinstance(item, RTreeNode):
                if item.is_leaf:
                    for bbox, data in item.entries:
                        pt = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                        d = np.sqrt((pt[0] - x)**2 + (pt[1] - y)**2)
                        queue.append((d, data))
                else:
                    for bbox, child in item.entries:
                        d = self._point_to_bbox_distance(point, bbox)
                        queue.append((d, child))
            else:
                results.append((dist, item))

            queue.sort(key=lambda x: x[0])

        return results[:k]

    def _enlarge_bbox(self, bbox1, bbox2):
        """Calculate enlarged bounding box"""
        return (
            min(bbox1[0], bbox2[0]),
            min(bbox1[1], bbox2[1]),
            max(bbox1[2], bbox2[2]),
            max(bbox1[3], bbox2[3])
        )

    def _bbox_area(self, bbox):
        """Calculate bounding box area"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def _bbox_intersects(self, bbox1, bbox2):
        """Check if two bounding boxes intersect"""
        return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                   bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

    def _point_to_bbox_distance(self, point, bbox):
        """Calculate minimum distance from point to bbox"""
        x, y = point
        dx = max(bbox[0] - x, 0, x - bbox[2])
        dy = max(bbox[1] - y, 0, y - bbox[3])
        return np.sqrt(dx**2 + dy**2)

    def get_height(self):
        """Get tree height"""
        if not self.root:
            return 0

        height = 0
        node = self.root
        while not node.is_leaf:
            height += 1
            if node.entries:
                node = node.entries[0][1]
            else:
                break
        return height + 1


class SpatialIndexAnalyzer:
    """Analyze and compare spatial indexing methods"""

    def __init__(self):
        self.points = None
        self.rtree = None
        self.kdtree = None
        self.performance_results = {}

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

        # Generate clustered points
        n_clusters = 8
        points_per_cluster = n_points // n_clusters

        data = []

        for i in range(n_clusters):
            center_x = np.random.uniform(-10, 10)
            center_y = np.random.uniform(-10, 10)

            cluster_points_x = center_x + np.random.normal(0, 1.5, points_per_cluster)
            cluster_points_y = center_y + np.random.normal(0, 1.5, points_per_cluster)

            for x, y in zip(cluster_points_x, cluster_points_y):
                data.append({
                    'x': x,
                    'y': y,
                    'cluster': i,
                    'poi_type': np.random.choice(['restaurant', 'shop', 'park', 'school']),
                    'importance': np.random.uniform(0.1, 1.0)
                })

        self.points = pd.DataFrame(data)

        print(f"✓ Generated {len(self.points)} spatial points")
        print(f"✓ Number of clusters: {n_clusters}")
        print(f"✓ Spatial extent: X=[{self.points['x'].min():.2f}, {self.points['x'].max():.2f}], "
              f"Y=[{self.points['y'].min():.2f}, {self.points['y'].max():.2f}]")

        return self.points

    def build_rtree_index(self):
        """Build R-tree spatial index"""
        print("\n" + "="*60)
        print("BUILDING R-TREE INDEX")
        print("="*60)

        start_time = time.time()

        self.rtree = RTree(max_entries=4, min_entries=2)

        for idx, row in self.points.iterrows():
            self.rtree.insert((row['x'], row['y']), {
                'id': idx,
                'x': row['x'],
                'y': row['y'],
                'type': row['poi_type']
            })

        build_time = time.time() - start_time

        print(f"✓ R-tree built in {build_time:.4f} seconds")
        print(f"✓ Tree height: {self.rtree.get_height()}")
        print(f"✓ Number of indexed points: {self.rtree.size}")

        self.performance_results['rtree_build_time'] = build_time

        return self.rtree

    def build_kdtree_index(self):
        """Build KD-tree index for comparison"""
        print("\n" + "="*60)
        print("BUILDING KD-TREE INDEX")
        print("="*60)

        start_time = time.time()

        points_array = self.points[['x', 'y']].values
        self.kdtree = KDTree(points_array)

        build_time = time.time() - start_time

        print(f"✓ KD-tree built in {build_time:.4f} seconds")

        self.performance_results['kdtree_build_time'] = build_time

        return self.kdtree

    def benchmark_range_queries(self, n_queries=100):
        """Benchmark range query performance"""
        print("\n" + "="*60)
        print("BENCHMARKING RANGE QUERIES")
        print("="*60)

        np.random.seed(123)

        # Generate random query boxes
        query_boxes = []
        for _ in range(n_queries):
            center_x = np.random.uniform(self.points['x'].min(), self.points['x'].max())
            center_y = np.random.uniform(self.points['y'].min(), self.points['y'].max())
            width = np.random.uniform(1, 5)
            height = np.random.uniform(1, 5)

            query_boxes.append((
                center_x - width/2,
                center_y - height/2,
                center_x + width/2,
                center_y + height/2
            ))

        # Benchmark R-tree
        start_time = time.time()
        rtree_results = []
        for bbox in query_boxes:
            results = self.rtree.range_query(bbox)
            rtree_results.append(len(results))
        rtree_time = time.time() - start_time

        # Benchmark brute force
        start_time = time.time()
        brute_results = []
        for bbox in query_boxes:
            results = self.points[
                (self.points['x'] >= bbox[0]) &
                (self.points['x'] <= bbox[2]) &
                (self.points['y'] >= bbox[1]) &
                (self.points['y'] <= bbox[3])
            ]
            brute_results.append(len(results))
        brute_time = time.time() - start_time

        print(f"\nR-tree range queries:")
        print(f"  Total time: {rtree_time:.4f} seconds")
        print(f"  Average per query: {rtree_time/n_queries*1000:.2f} ms")
        print(f"  Average results per query: {np.mean(rtree_results):.1f}")

        print(f"\nBrute force range queries:")
        print(f"  Total time: {brute_time:.4f} seconds")
        print(f"  Average per query: {brute_time/n_queries*1000:.2f} ms")
        print(f"  Speedup: {brute_time/rtree_time:.2f}x")

        self.performance_results['rtree_range_time'] = rtree_time
        self.performance_results['brute_range_time'] = brute_time
        self.performance_results['range_speedup'] = brute_time / rtree_time

        return rtree_time, brute_time

    def benchmark_nearest_neighbor(self, n_queries=100, k=10):
        """Benchmark nearest neighbor queries"""
        print("\n" + "="*60)
        print("BENCHMARKING NEAREST NEIGHBOR QUERIES")
        print("="*60)

        np.random.seed(456)

        # Generate random query points
        query_points = []
        for _ in range(n_queries):
            x = np.random.uniform(self.points['x'].min(), self.points['x'].max())
            y = np.random.uniform(self.points['y'].min(), self.points['y'].max())
            query_points.append((x, y))

        # Benchmark R-tree
        start_time = time.time()
        for point in query_points:
            results = self.rtree.nearest_neighbor(point, k=k)
        rtree_time = time.time() - start_time

        # Benchmark KD-tree
        start_time = time.time()
        for point in query_points:
            results = self.kdtree.query([point], k=k)
        kdtree_time = time.time() - start_time

        # Benchmark brute force
        start_time = time.time()
        for point in query_points:
            distances = np.sqrt(
                (self.points['x'] - point[0])**2 +
                (self.points['y'] - point[1])**2
            )
            nearest = np.argsort(distances)[:k]
        brute_time = time.time() - start_time

        print(f"\nR-tree NN queries (k={k}):")
        print(f"  Total time: {rtree_time:.4f} seconds")
        print(f"  Average per query: {rtree_time/n_queries*1000:.2f} ms")

        print(f"\nKD-tree NN queries (k={k}):")
        print(f"  Total time: {kdtree_time:.4f} seconds")
        print(f"  Average per query: {kdtree_time/n_queries*1000:.2f} ms")

        print(f"\nBrute force NN queries:")
        print(f"  Total time: {brute_time:.4f} seconds")
        print(f"  Average per query: {brute_time/n_queries*1000:.2f} ms")

        print(f"\nSpeedups vs brute force:")
        print(f"  R-tree: {brute_time/rtree_time:.2f}x")
        print(f"  KD-tree: {brute_time/kdtree_time:.2f}x")

        self.performance_results['rtree_nn_time'] = rtree_time
        self.performance_results['kdtree_nn_time'] = kdtree_time
        self.performance_results['brute_nn_time'] = brute_time

        return rtree_time, kdtree_time, brute_time

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(18, 12))

        # 1. Spatial distribution
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(
            self.points['x'],
            self.points['y'],
            c=self.points['cluster'],
            s=self.points['importance'] * 50,
            alpha=0.6,
            cmap='tab10',
            edgecolors='black',
            linewidths=0.5
        )
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Spatial Point Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Cluster')

        # 2. R-tree visualization (sample query)
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(self.points['x'], self.points['y'], c='lightgray', s=10, alpha=0.5)

        # Draw sample query box
        query_x, query_y = 0, 0
        query_width, query_height = 4, 4
        query_bbox = (query_x - query_width/2, query_y - query_height/2,
                     query_x + query_width/2, query_y + query_height/2)

        rect = Rectangle((query_bbox[0], query_bbox[1]),
                        query_bbox[2] - query_bbox[0],
                        query_bbox[3] - query_bbox[1],
                        fill=False, edgecolor='red', linewidth=2)
        ax2.add_patch(rect)

        # Highlight results
        results = self.rtree.range_query(query_bbox)
        if results:
            result_x = [r['x'] for r in results]
            result_y = [r['y'] for r in results]
            ax2.scatter(result_x, result_y, c='red', s=50, zorder=5, label=f'{len(results)} results')

        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title('R-tree Range Query Example', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Build time comparison
        ax3 = plt.subplot(2, 3, 3)
        build_times = [
            self.performance_results['rtree_build_time'] * 1000,
            self.performance_results['kdtree_build_time'] * 1000
        ]
        bars = ax3.bar(['R-tree', 'KD-tree'], build_times, color=['#3498db', '#2ecc71'],
                      edgecolor='black')
        ax3.set_ylabel('Build Time (ms)')
        ax3.set_title('Index Build Time Comparison', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}ms', ha='center', va='bottom')

        # 4. Range query performance
        ax4 = plt.subplot(2, 3, 4)
        range_times = [
            self.performance_results['rtree_range_time'] * 1000,
            self.performance_results['brute_range_time'] * 1000
        ]
        bars = ax4.bar(['R-tree', 'Brute Force'], range_times,
                      color=['#3498db', '#e74c3c'], edgecolor='black')
        ax4.set_ylabel('Query Time (ms)')
        ax4.set_title('Range Query Performance', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}ms', ha='center', va='bottom')

        # 5. NN query performance
        ax5 = plt.subplot(2, 3, 5)
        nn_times = [
            self.performance_results['rtree_nn_time'] * 1000,
            self.performance_results['kdtree_nn_time'] * 1000,
            self.performance_results['brute_nn_time'] * 1000
        ]
        bars = ax5.bar(['R-tree', 'KD-tree', 'Brute Force'], nn_times,
                      color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
        ax5.set_ylabel('Query Time (ms)')
        ax5.set_title('Nearest Neighbor Query Performance', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}ms', ha='center', va='bottom', fontsize=9)

        # 6. Speedup comparison
        ax6 = plt.subplot(2, 3, 6)
        speedups = [
            self.performance_results['range_speedup'],
            self.performance_results['brute_nn_time'] / self.performance_results['rtree_nn_time'],
            self.performance_results['brute_nn_time'] / self.performance_results['kdtree_nn_time']
        ]
        bars = ax6.bar(['Range\n(R-tree)', 'NN\n(R-tree)', 'NN\n(KD-tree)'],
                      speedups, color=['#9b59b6', '#f39c12', '#1abc9c'],
                      edgecolor='black')
        ax6.set_ylabel('Speedup Factor (vs Brute Force)')
        ax6.set_title('Performance Speedup Analysis', fontsize=12, fontweight='bold')
        ax6.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}x', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('rtree_spatial_indexing.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'rtree_spatial_indexing.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("R-TREE SPATIAL INDEXING ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = SpatialIndexAnalyzer()

    # Generate spatial data
    analyzer.generate_spatial_data(n_points=1000)

    # Build indexes
    analyzer.build_rtree_index()
    analyzer.build_kdtree_index()

    # Benchmark queries
    analyzer.benchmark_range_queries(n_queries=100)
    analyzer.benchmark_nearest_neighbor(n_queries=100, k=10)

    # Visualize results
    analyzer.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
