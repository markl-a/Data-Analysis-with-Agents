"""
Delivery Route Optimization - Geospatial Analysis
Optimize delivery routes using spatial algorithms

Dataset: Synthetic delivery location data
Difficulty: ⭐⭐⭐ Advanced
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class DeliveryRouteOptimizer:
    """Delivery route optimization using geospatial algorithms"""

    def __init__(self, city_name="Metro City"):
        self.city_name = city_name
        self.deliveries = None
        self.depot_location = None
        self.routes = None

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
        """Generate synthetic delivery locations"""
        np.random.seed(42)

        # Depot (central warehouse)
        self.depot_location = {'latitude': 40.7589, 'longitude': -73.9851, 'name': 'Main Depot'}

        # Generate delivery locations in clusters (representing neighborhoods)
        n_deliveries = 80
        n_clusters = 6

        # Cluster centers
        center_lat, center_lon = 40.7589, -73.9851
        lat_spread, lon_spread = 0.10, 0.15

        cluster_lats = center_lat + np.random.uniform(-lat_spread*0.7, lat_spread*0.7, n_clusters)
        cluster_lons = center_lon + np.random.uniform(-lon_spread*0.7, lon_spread*0.7, n_clusters)

        deliveries = []

        for i in range(n_deliveries):
            # Select cluster
            cluster_idx = np.random.randint(0, n_clusters)

            # Generate location near cluster center
            lat = cluster_lats[cluster_idx] + np.random.normal(0, 0.01)
            lon = cluster_lons[cluster_idx] + np.random.normal(0, 0.015)

            # Package details
            package_weight = np.random.uniform(0.5, 20.0)  # kg
            priority = np.random.choice(['Standard', 'Express', 'Urgent'], p=[0.6, 0.3, 0.1])
            time_window_start = np.random.choice([8, 9, 10, 11, 12])
            time_window_end = time_window_start + np.random.choice([2, 3, 4])

            # Service time (minutes)
            service_time = np.random.uniform(3, 10)

            deliveries.append({
                'delivery_id': f'DEL{i+1:03d}',
                'latitude': lat,
                'longitude': lon,
                'package_weight_kg': package_weight,
                'priority': priority,
                'time_window_start': time_window_start,
                'time_window_end': time_window_end,
                'service_time_min': service_time,
                'cluster': cluster_idx
            })

        self.deliveries = pd.DataFrame(deliveries)

        # Calculate distances from depot
        depot_distances = self.haversine_distance(
            self.depot_location['latitude'],
            self.depot_location['longitude'],
            self.deliveries['latitude'].values,
            self.deliveries['longitude'].values
        )
        self.deliveries['dist_from_depot_km'] = depot_distances

        print(f"✓ Generated {len(self.deliveries)} delivery locations")
        print(f"✓ Depot: ({self.depot_location['latitude']:.4f}, {self.depot_location['longitude']:.4f})")
        print(f"✓ Priority breakdown:")
        for priority, count in self.deliveries['priority'].value_counts().items():
            print(f"    {priority}: {count}")
        print(f"✓ Total package weight: {self.deliveries['package_weight_kg'].sum():.1f} kg")

    def nearest_neighbor_route(self, start_idx=None, max_deliveries=None):
        """Create route using nearest neighbor heuristic"""
        if max_deliveries is None:
            max_deliveries = len(self.deliveries)

        # Create distance matrix
        coords = self.deliveries[['latitude', 'longitude']].values
        n_points = len(coords)

        # Calculate pairwise distances
        distances = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    distances[i, j] = self.haversine_distance(
                        coords[i, 0], coords[i, 1],
                        coords[j, 0], coords[j, 1]
                    )

        # Nearest neighbor algorithm
        if start_idx is None:
            # Start from closest to depot
            start_idx = self.deliveries['dist_from_depot_km'].idxmin()

        route = [start_idx]
        unvisited = set(range(n_points)) - {start_idx}
        current = start_idx

        while unvisited and len(route) < max_deliveries:
            # Find nearest unvisited
            nearest = min(unvisited, key=lambda x: distances[current, x])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return route

    def optimize_routes(self, n_vehicles=4, max_capacity_kg=100):
        """Optimize delivery routes for multiple vehicles"""
        print("\n" + "="*60)
        print("ROUTE OPTIMIZATION")
        print("="*60)

        # Sort by priority and distance
        priority_order = {'Urgent': 3, 'Express': 2, 'Standard': 1}
        self.deliveries['priority_score'] = self.deliveries['priority'].map(priority_order)
        sorted_deliveries = self.deliveries.sort_values(
            ['priority_score', 'dist_from_depot_km'],
            ascending=[False, True]
        ).reset_index(drop=True)

        # Assign to vehicles
        vehicle_routes = []
        remaining_indices = set(sorted_deliveries.index)

        for vehicle_id in range(n_vehicles):
            if not remaining_indices:
                break

            route_indices = []
            current_weight = 0

            # Start with highest priority unassigned
            for idx in list(remaining_indices):
                delivery = sorted_deliveries.loc[idx]

                if current_weight + delivery['package_weight_kg'] <= max_capacity_kg:
                    route_indices.append(idx)
                    current_weight += delivery['package_weight_kg']
                    remaining_indices.remove(idx)

            if route_indices:
                # Optimize order using nearest neighbor
                if len(route_indices) > 1:
                    # Create subset deliveries
                    subset = sorted_deliveries.loc[route_indices].reset_index(drop=True)
                    optimized_order = self.nearest_neighbor_route_subset(subset)
                    route_indices = [route_indices[i] for i in optimized_order]

                # Calculate route metrics
                route_deliveries = sorted_deliveries.loc[route_indices]
                total_distance = self._calculate_route_distance(route_deliveries)
                total_time = route_deliveries['service_time_min'].sum() + total_distance * 3  # 3 min per km

                vehicle_routes.append({
                    'vehicle_id': vehicle_id,
                    'route_indices': route_indices,
                    'n_deliveries': len(route_indices),
                    'total_weight_kg': current_weight,
                    'total_distance_km': total_distance,
                    'estimated_time_min': total_time
                })

        self.routes = pd.DataFrame(vehicle_routes)

        print(f"\n✓ Optimized routes for {len(self.routes)} vehicles")
        print(f"✓ Total deliveries assigned: {sum(self.routes['n_deliveries'])}")
        print(f"✓ Unassigned deliveries: {len(remaining_indices)}")

        print("\nRoute Summary:")
        for _, route in self.routes.iterrows():
            print(f"\nVehicle {route['vehicle_id']}:")
            print(f"  Deliveries: {route['n_deliveries']}")
            print(f"  Total Weight: {route['total_weight_kg']:.1f} kg")
            print(f"  Total Distance: {route['total_distance_km']:.2f} km")
            print(f"  Estimated Time: {route['estimated_time_min']:.0f} min")

        return self.routes

    def nearest_neighbor_route_subset(self, subset_df):
        """Nearest neighbor for subset of deliveries"""
        coords = subset_df[['latitude', 'longitude']].values
        n_points = len(coords)

        if n_points <= 1:
            return list(range(n_points))

        # Distance matrix
        distances = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    distances[i, j] = self.haversine_distance(
                        coords[i, 0], coords[i, 1],
                        coords[j, 0], coords[j, 1]
                    )

        # Start from closest to depot
        start_idx = subset_df['dist_from_depot_km'].idxmin()
        start_idx = subset_df.index.get_loc(start_idx)

        route = [start_idx]
        unvisited = set(range(n_points)) - {start_idx}
        current = start_idx

        while unvisited:
            nearest = min(unvisited, key=lambda x: distances[current, x])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return route

    def _calculate_route_distance(self, route_deliveries):
        """Calculate total distance for a route"""
        if len(route_deliveries) == 0:
            return 0

        total_dist = 0

        # Depot to first delivery
        first = route_deliveries.iloc[0]
        total_dist += self.haversine_distance(
            self.depot_location['latitude'],
            self.depot_location['longitude'],
            first['latitude'],
            first['longitude']
        )

        # Between deliveries
        for i in range(len(route_deliveries) - 1):
            current = route_deliveries.iloc[i]
            next_loc = route_deliveries.iloc[i + 1]
            total_dist += self.haversine_distance(
                current['latitude'], current['longitude'],
                next_loc['latitude'], next_loc['longitude']
            )

        # Last delivery back to depot
        last = route_deliveries.iloc[-1]
        total_dist += self.haversine_distance(
            last['latitude'],
            last['longitude'],
            self.depot_location['latitude'],
            self.depot_location['longitude']
        )

        return total_dist

    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. All Delivery Locations
        ax = axes[0, 0]
        scatter = ax.scatter(
            self.deliveries['longitude'],
            self.deliveries['latitude'],
            c=self.deliveries['cluster'],
            s=100,
            alpha=0.6,
            cmap='tab10',
            edgecolors='black',
            linewidths=1
        )
        ax.scatter(self.depot_location['longitude'], self.depot_location['latitude'],
                  c='red', s=500, marker='s', edgecolors='darkred',
                  linewidths=3, label='Depot', zorder=5)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Delivery Locations', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Optimized Routes
        ax = axes[0, 1]
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']

        if self.routes is not None:
            for _, route in self.routes.iterrows():
                vehicle_id = route['vehicle_id']
                route_indices = route['route_indices']

                route_deliveries = self.deliveries.loc[route_indices]

                # Plot deliveries
                ax.scatter(route_deliveries['longitude'], route_deliveries['latitude'],
                          c=colors[vehicle_id % len(colors)], s=100, alpha=0.7,
                          label=f'Vehicle {vehicle_id}', zorder=3)

                # Plot route lines
                lats = [self.depot_location['latitude']] + route_deliveries['latitude'].tolist() + [self.depot_location['latitude']]
                lons = [self.depot_location['longitude']] + route_deliveries['longitude'].tolist() + [self.depot_location['longitude']]

                ax.plot(lons, lats, c=colors[vehicle_id % len(colors)],
                       linewidth=2, alpha=0.5, zorder=2)

        ax.scatter(self.depot_location['longitude'], self.depot_location['latitude'],
                  c='red', s=500, marker='s', edgecolors='darkred',
                  linewidths=3, label='Depot', zorder=5)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Optimized Delivery Routes', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Priority Distribution
        ax = axes[1, 0]
        priority_counts = self.deliveries['priority'].value_counts()
        colors_priority = {'Urgent': '#e74c3c', 'Express': '#f39c12', 'Standard': '#3498db'}
        bars = ax.bar(priority_counts.index,
                     priority_counts.values,
                     color=[colors_priority[p] for p in priority_counts.index],
                     edgecolor='black')
        ax.set_ylabel('Number of Deliveries')
        ax.set_title('Deliveries by Priority', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')

        # 4. Route Metrics
        ax = axes[1, 1]
        if self.routes is not None:
            x = np.arange(len(self.routes))
            width = 0.35

            ax.bar(x - width/2, self.routes['n_deliveries'],
                  width, label='Deliveries', color='steelblue', edgecolor='black')
            ax.set_xlabel('Vehicle ID')
            ax.set_ylabel('Number of Deliveries', color='steelblue')
            ax.tick_params(axis='y', labelcolor='steelblue')
            ax.set_xticks(x)
            ax.set_xticklabels([f'V{i}' for i in self.routes['vehicle_id']])

            ax2 = ax.twinx()
            ax2.bar(x + width/2, self.routes['total_distance_km'],
                   width, label='Distance (km)', color='coral', edgecolor='black')
            ax2.set_ylabel('Total Distance (km)', color='coral')
            ax2.tick_params(axis='y', labelcolor='coral')

            ax.set_title('Route Metrics by Vehicle', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('delivery_route_optimization.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'delivery_route_optimization.png'")
        plt.show()


def main():
    """Main execution function"""
    print("="*60)
    print("DELIVERY ROUTE OPTIMIZATION - GEOSPATIAL ANALYSIS")
    print("="*60)

    # Initialize optimizer
    optimizer = DeliveryRouteOptimizer(city_name="New York")

    # Generate data
    optimizer.create_sample_data()

    # Optimize routes
    routes = optimizer.optimize_routes(n_vehicles=4, max_capacity_kg=150)

    # Visualize
    optimizer.visualize_results()

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
