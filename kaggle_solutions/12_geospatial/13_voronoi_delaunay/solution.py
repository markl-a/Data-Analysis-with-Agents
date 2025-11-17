"""
Voronoi Diagrams and Delaunay Triangulation - Geospatial Analysis
Implement Voronoi diagrams and Delaunay triangulation for spatial analysis

Dataset: Synthetic facility locations and service areas
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import Voronoi, Delaunay, voronoi_plot_2d, delaunay_plot_2d
from scipy.spatial.distance import cdist
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')


class VoronoiDelaunayAnalyzer:
    """Analyze spatial relationships using Voronoi and Delaunay"""

    def __init__(self):
        self.facilities = None
        self.customers = None
        self.voronoi = None
        self.delaunay = None
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

    def generate_spatial_data(self, n_facilities=20, n_customers=500):
        """Generate synthetic facility and customer data"""
        print("="*60)
        print("GENERATING SPATIAL DATA")
        print("="*60)

        np.random.seed(42)

        # Generate facility locations (strategic placement)
        facilities_data = []

        # Use stratified sampling for better coverage
        grid_size = int(np.sqrt(n_facilities))
        for i in range(grid_size):
            for j in range(grid_size):
                if len(facilities_data) >= n_facilities:
                    break

                # Base position in grid
                base_x = (i + 0.5) * (100 / grid_size)
                base_y = (j + 0.5) * (100 / grid_size)

                # Add random offset
                x = base_x + np.random.normal(0, 5)
                y = base_y + np.random.normal(0, 5)

                facilities_data.append({
                    'x': np.clip(x, 0, 100),
                    'y': np.clip(y, 0, 100),
                    'facility_id': len(facilities_data),
                    'capacity': np.random.randint(50, 200),
                    'service_quality': np.random.uniform(0.5, 1.0)
                })

        self.facilities = pd.DataFrame(facilities_data)

        # Generate customer locations (clustered around facilities)
        customers_data = []

        for _ in range(n_customers):
            # Choose random facility as attraction point
            facility = self.facilities.sample(1).iloc[0]

            # Generate customer near facility with some spread
            x = facility['x'] + np.random.normal(0, 8)
            y = facility['y'] + np.random.normal(0, 8)

            customers_data.append({
                'x': np.clip(x, 0, 100),
                'y': np.clip(y, 0, 100),
                'customer_id': len(customers_data),
                'demand': np.random.randint(1, 10)
            })

        self.customers = pd.DataFrame(customers_data)

        print(f"✓ Generated {len(self.facilities)} facilities")
        print(f"✓ Generated {len(self.customers)} customers")
        print(f"✓ Spatial extent: 100 x 100 units")
        print(f"✓ Total customer demand: {self.customers['demand'].sum()}")
        print(f"✓ Total facility capacity: {self.facilities['capacity'].sum()}")

        return self.facilities, self.customers

    def compute_voronoi_diagram(self):
        """Compute Voronoi diagram for facilities"""
        print("\n" + "="*60)
        print("COMPUTING VORONOI DIAGRAM")
        print("="*60)

        points = self.facilities[['x', 'y']].values

        # Add boundary points to avoid infinite regions
        boundary_points = np.array([
            [-50, -50], [-50, 150], [150, -50], [150, 150],
            [50, -50], [50, 150], [-50, 50], [150, 50]
        ])
        extended_points = np.vstack([points, boundary_points])

        self.voronoi = Voronoi(extended_points)

        print(f"✓ Voronoi diagram computed")
        print(f"✓ Number of regions: {len(self.voronoi.regions)}")
        print(f"✓ Number of vertices: {len(self.voronoi.vertices)}")
        print(f"✓ Number of ridges: {len(self.voronoi.ridge_points)}")

        return self.voronoi

    def compute_delaunay_triangulation(self):
        """Compute Delaunay triangulation"""
        print("\n" + "="*60)
        print("COMPUTING DELAUNAY TRIANGULATION")
        print("="*60)

        points = self.facilities[['x', 'y']].values
        self.delaunay = Delaunay(points)

        print(f"✓ Delaunay triangulation computed")
        print(f"✓ Number of triangles: {len(self.delaunay.simplices)}")
        print(f"✓ Number of points: {len(points)}")

        return self.delaunay

    def assign_customers_to_facilities(self):
        """Assign customers to nearest facilities using Voronoi"""
        print("\n" + "="*60)
        print("ASSIGNING CUSTOMERS TO FACILITIES")
        print("="*60)

        facility_points = self.facilities[['x', 'y']].values
        customer_points = self.customers[['x', 'y']].values

        # Calculate distances
        distances = cdist(customer_points, facility_points, metric='euclidean')

        # Assign to nearest facility
        nearest_facility = np.argmin(distances, axis=1)
        nearest_distance = np.min(distances, axis=1)

        self.customers['assigned_facility'] = nearest_facility
        self.customers['distance_to_facility'] = nearest_distance

        # Calculate facility loads
        facility_loads = self.customers.groupby('assigned_facility')['demand'].sum()

        print(f"\nAssignment Statistics:")
        print(f"  Average distance to facility: {nearest_distance.mean():.2f} units")
        print(f"  Max distance to facility: {nearest_distance.max():.2f} units")
        print(f"  Min distance to facility: {nearest_distance.min():.2f} units")

        print(f"\nFacility Utilization:")
        for idx, row in self.facilities.iterrows():
            load = facility_loads.get(idx, 0)
            capacity = row['capacity']
            utilization = (load / capacity) * 100 if capacity > 0 else 0

            print(f"  Facility {idx}: {load}/{capacity} ({utilization:.1f}% utilization)")

        self.metrics['avg_distance'] = nearest_distance.mean()
        self.metrics['max_distance'] = nearest_distance.max()
        self.metrics['facility_loads'] = facility_loads.to_dict()

        return self.customers

    def analyze_service_areas(self):
        """Analyze service area characteristics"""
        print("\n" + "="*60)
        print("ANALYZING SERVICE AREAS")
        print("="*60)

        areas = []
        perimeters = []

        for idx in range(len(self.facilities)):
            region_index = self.voronoi.point_region[idx]
            region = self.voronoi.regions[region_index]

            if -1 not in region and len(region) > 0:
                vertices = self.voronoi.vertices[region]

                # Calculate area using Shoelace formula
                area = 0.5 * abs(sum(
                    vertices[i][0] * vertices[(i+1) % len(vertices)][1] -
                    vertices[(i+1) % len(vertices)][0] * vertices[i][1]
                    for i in range(len(vertices))
                ))

                # Calculate perimeter
                perimeter = sum(
                    self.euclidean_distance(
                        vertices[i][0], vertices[i][1],
                        vertices[(i+1) % len(vertices)][0],
                        vertices[(i+1) % len(vertices)][1]
                    )
                    for i in range(len(vertices))
                )

                areas.append(area)
                perimeters.append(perimeter)
            else:
                areas.append(np.nan)
                perimeters.append(np.nan)

        valid_areas = [a for a in areas if not np.isnan(a)]
        valid_perimeters = [p for p in perimeters if not np.isnan(p)]

        print(f"\nService Area Statistics:")
        print(f"  Mean area: {np.mean(valid_areas):.2f} units²")
        print(f"  Std area: {np.std(valid_areas):.2f} units²")
        print(f"  Min/Max area: {np.min(valid_areas):.2f}/{np.max(valid_areas):.2f} units²")

        print(f"\nService Perimeter Statistics:")
        print(f"  Mean perimeter: {np.mean(valid_perimeters):.2f} units")
        print(f"  Min/Max perimeter: {np.min(valid_perimeters):.2f}/{np.max(valid_perimeters):.2f} units")

        self.metrics['service_areas'] = areas
        self.metrics['service_perimeters'] = perimeters

        return areas, perimeters

    def analyze_triangulation_properties(self):
        """Analyze Delaunay triangulation properties"""
        print("\n" + "="*60)
        print("ANALYZING TRIANGULATION PROPERTIES")
        print("="*60)

        points = self.facilities[['x', 'y']].values
        triangles = self.delaunay.simplices

        edge_lengths = []
        triangle_areas = []
        triangle_angles = []

        for tri in triangles:
            # Get triangle vertices
            p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]

            # Calculate edge lengths
            e1 = self.euclidean_distance(p1[0], p1[1], p2[0], p2[1])
            e2 = self.euclidean_distance(p2[0], p2[1], p3[0], p3[1])
            e3 = self.euclidean_distance(p3[0], p3[1], p1[0], p1[1])

            edge_lengths.extend([e1, e2, e3])

            # Calculate area using cross product
            area = 0.5 * abs(
                (p2[0] - p1[0]) * (p3[1] - p1[1]) -
                (p3[0] - p1[0]) * (p2[1] - p1[1])
            )
            triangle_areas.append(area)

            # Calculate angles using law of cosines
            angle1 = np.arccos(np.clip((e2**2 + e3**2 - e1**2) / (2*e2*e3), -1, 1))
            angle2 = np.arccos(np.clip((e1**2 + e3**2 - e2**2) / (2*e1*e3), -1, 1))
            angle3 = np.arccos(np.clip((e1**2 + e2**2 - e3**2) / (2*e1*e2), -1, 1))

            triangle_angles.extend([np.degrees(angle1), np.degrees(angle2), np.degrees(angle3)])

        edge_lengths = np.array(edge_lengths)
        triangle_areas = np.array(triangle_areas)
        triangle_angles = np.array(triangle_angles)

        print(f"\nEdge Statistics:")
        print(f"  Total edges: {len(edge_lengths)}")
        print(f"  Mean edge length: {edge_lengths.mean():.2f} units")
        print(f"  Min/Max edge length: {edge_lengths.min():.2f}/{edge_lengths.max():.2f} units")

        print(f"\nTriangle Statistics:")
        print(f"  Total triangles: {len(triangles)}")
        print(f"  Mean triangle area: {triangle_areas.mean():.2f} units²")
        print(f"  Min/Max triangle area: {triangle_areas.min():.2f}/{triangle_areas.max():.2f} units²")

        print(f"\nAngle Statistics:")
        print(f"  Mean angle: {triangle_angles.mean():.2f}°")
        print(f"  Min/Max angle: {triangle_angles.min():.2f}°/{triangle_angles.max():.2f}°")

        self.metrics['edge_lengths'] = edge_lengths
        self.metrics['triangle_areas'] = triangle_areas
        self.metrics['triangle_angles'] = triangle_angles

        return edge_lengths, triangle_areas, triangle_angles

    def find_nearest_neighbors_delaunay(self):
        """Find nearest neighbors using Delaunay triangulation"""
        print("\n" + "="*60)
        print("FINDING NEAREST NEIGHBORS")
        print("="*60)

        # Build neighbor graph from Delaunay triangulation
        n_facilities = len(self.facilities)
        neighbors = {i: set() for i in range(n_facilities)}

        for simplex in self.delaunay.simplices:
            # Each triangle creates 3 neighbor pairs
            neighbors[simplex[0]].update([simplex[1], simplex[2]])
            neighbors[simplex[1]].update([simplex[0], simplex[2]])
            neighbors[simplex[2]].update([simplex[0], simplex[1]])

        # Calculate statistics
        neighbor_counts = [len(neighbors[i]) for i in range(n_facilities)]

        print(f"\nNeighbor Statistics:")
        print(f"  Mean neighbors per facility: {np.mean(neighbor_counts):.2f}")
        print(f"  Min/Max neighbors: {np.min(neighbor_counts)}/{np.max(neighbor_counts)}")

        self.metrics['neighbors'] = neighbors
        self.metrics['neighbor_counts'] = neighbor_counts

        return neighbors

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(18, 12))

        # 1. Voronoi diagram
        ax1 = plt.subplot(2, 3, 1)
        voronoi_plot_2d(self.voronoi, ax=ax1, show_vertices=False, line_colors='red',
                       line_width=1, point_size=0)

        # Plot facilities
        ax1.scatter(self.facilities['x'], self.facilities['y'],
                   c='blue', s=100, marker='s', edgecolors='black',
                   linewidths=2, label='Facilities', zorder=5)

        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Voronoi Diagram - Service Areas', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Delaunay triangulation
        ax2 = plt.subplot(2, 3, 2)
        ax2.triplot(self.facilities['x'], self.facilities['y'],
                   self.delaunay.simplices, 'c-', linewidth=1, alpha=0.5)
        ax2.scatter(self.facilities['x'], self.facilities['y'],
                   c='red', s=100, marker='^', edgecolors='black',
                   linewidths=2, label='Facilities', zorder=5)

        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.set_title('Delaunay Triangulation', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Customer assignment
        ax3 = plt.subplot(2, 3, 3)

        # Plot customers colored by assigned facility
        scatter = ax3.scatter(self.customers['x'], self.customers['y'],
                            c=self.customers['assigned_facility'],
                            s=20, alpha=0.6, cmap='tab20')

        # Plot facilities
        ax3.scatter(self.facilities['x'], self.facilities['y'],
                   c='black', s=200, marker='*', edgecolors='white',
                   linewidths=2, label='Facilities', zorder=5)

        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_title('Customer-Facility Assignment', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Service area distribution
        ax4 = plt.subplot(2, 3, 4)
        valid_areas = [a for a in self.metrics['service_areas'] if not np.isnan(a)]
        ax4.hist(valid_areas, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax4.axvline(np.mean(valid_areas), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(valid_areas):.1f}')
        ax4.set_xlabel('Service Area (units²)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Service Area Distribution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Edge length distribution
        ax5 = plt.subplot(2, 3, 5)
        edge_lengths = self.metrics['edge_lengths']
        ax5.hist(edge_lengths, bins=20, color='orange', edgecolor='black', alpha=0.7)
        ax5.axvline(edge_lengths.mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {edge_lengths.mean():.1f}')
        ax5.set_xlabel('Edge Length (units)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Triangulation Edge Length Distribution', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Distance statistics
        ax6 = plt.subplot(2, 3, 6)
        distance_stats = {
            'Avg Customer\nDistance': self.metrics['avg_distance'],
            'Max Customer\nDistance': self.metrics['max_distance'],
            'Avg Edge\nLength': edge_lengths.mean()
        }

        bars = ax6.bar(range(len(distance_stats)), list(distance_stats.values()),
                      color=['#2ecc71', '#e74c3c', '#3498db'], edgecolor='black')
        ax6.set_xticks(range(len(distance_stats)))
        ax6.set_xticklabels(distance_stats.keys(), fontsize=9)
        ax6.set_ylabel('Distance (units)')
        ax6.set_title('Distance Metrics Summary', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('voronoi_delaunay_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'voronoi_delaunay_analysis.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("VORONOI DIAGRAMS AND DELAUNAY TRIANGULATION ANALYSIS")
    print("="*60)

    # Initialize analyzer
    analyzer = VoronoiDelaunayAnalyzer()

    # Generate data
    analyzer.generate_spatial_data(n_facilities=20, n_customers=500)

    # Compute Voronoi diagram
    analyzer.compute_voronoi_diagram()

    # Compute Delaunay triangulation
    analyzer.compute_delaunay_triangulation()

    # Assign customers
    analyzer.assign_customers_to_facilities()

    # Analyze service areas
    analyzer.analyze_service_areas()

    # Analyze triangulation
    analyzer.analyze_triangulation_properties()

    # Find neighbors
    analyzer.find_nearest_neighbors_delaunay()

    # Visualize results
    analyzer.visualize_results()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
