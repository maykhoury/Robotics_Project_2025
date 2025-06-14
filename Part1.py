import numpy as np
import csv
from DroneClient import DroneClient
from DroneTypes import Position
from shapely.geometry import Point, Polygon, LineString
import matplotlib.pyplot as plt
import time
from scipy.spatial import Voronoi
from Bug2_algorithm import *
from Vector import *
class Obstacle:
    def __init__(self, num_points: int, x: float, y: float, z: float, obs_type: str):
        self.num_points = num_points
        self.x = x
        self.y = y
        self.z = z
        self.obs_type = obs_type

    def __str__(self):
        return f"Obstacle(points={self.num_points}, x={self.x}, y={self.y}, z={self.z}, type={self.obs_type})"


class Navigation:
    def __init__(self, drone_client: DroneClient):
        self.drone = drone_client
        self.obstacles = []
        self.polygons = []
        self.prior_knowledge_bounds = [
            (-670, -700, -100),
            (-1250, -700, -100),
            (-1250, -1200, -100),
            (-670, -1200, -100)
        ]
        self.max_lidar_range = 35  # meters
        self.safety_distance = 1  # meters
        self.start_pos = None
        self.mid_pos = None
        self.load_prior_obstacles()
        self.load_polygons()
        self.phase = "midway"
        self.safe_distance = 5.0  # Safe distance from obstacles in meters
        self.goal_threshold = 3.0  # Distance threshold to consider goal reached
        self.visited_obstacles = []
        self.vertigo = 0  # Initialize vertigo counter
        self.last_heading = np.array([1, 0])  # Track last movement direction
        self.plane=None

################################################ Init ##################################################################################
    # start is (start.x, start.y), mid is (mid.x, mid.y), plane is the z value
    def set_waypoints(self, start: Position, mid: Position, plane: Position):
        """Set the waypoints for navigation"""
        self.start_pos = np.array([start[0], start[1], plane])
        self.mid_pos = np.array([mid[0], mid[1], plane])
        self.plane=plane
        print(f"Waypoints set - Start: {self.start_pos}, Mid: {self.mid_pos}")

    ################################################ Loading Obstacles #####################################################################
    
    # Check if a point [x,y,z] is within the prior knowledge bounds
    def is_point_in_bounds(self, point) -> bool:
        """Check if a point [x,y,z] is within the prior knowledge bounds"""
        x, y, z = point
        x_coords = [b[0] for b in self.prior_knowledge_bounds]
        y_coords = [b[1] for b in self.prior_knowledge_bounds]

        # Check if point is within the rectangle bounds
        return (min(x_coords) <= x <= max(x_coords) and
                min(y_coords) <= y <= max(y_coords))

    # Load polygons from CSV file
    def load_polygons(self):
        """
        Load obstacles from CSV and create polygons by grouping obstacles of the same type.
        Each polygon is a list of points ordered clockwise.
        """
        # Initialize polygons list and temporary dictionary to store points by type
        self.polygons = []
        polygons_dict = {}

        try:
            with open('obstacles_100m_above_sea_level.csv', 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header

                # First, group points by obstacle type
                for row in csv_reader:
                    if len(row) >= 5:
                        try:
                            num_points = int(row[0])
                            x = float(row[1])
                            y = float(row[2])
                            z = float(row[3])
                            obs_type = row[4]

                            # Only add if within bounds
                            if self.is_point_in_bounds([x, y, z]):
                                if obs_type not in polygons_dict:
                                    polygons_dict[obs_type] = []
                                polygons_dict[obs_type].append((x, y))

                        except ValueError as e:
                            print(f"Skipping invalid row: {row}, Error: {e}")

                # For each obstacle type, order points clockwise
                for obs_type, points in polygons_dict.items():
                    if len(points) < 3:  # Skip if not enough points for a polygon
                        continue

                    # Find centroid
                    centroid_x = sum(p[0] for p in points) / len(points)
                    centroid_y = sum(p[1] for p in points) / len(points)

                    # Sort points clockwise around centroid using atan2
                    sorted_points = sorted(points,
                                           key=lambda p: np.arctan2(p[1] - centroid_y, p[0] - centroid_x))

                    # Add sorted polygon to the list
                    self.polygons.append(sorted_points)

                print(f"Created {len(self.polygons)} polygons from obstacles")

        except Exception as e:
            print(f"Error loading polygons: {str(e)}")
            self.polygons = []  # Initialize empty list if loading fails

    # Load obstacles from CSV file for the prior knowledge area
    def load_prior_obstacles(self):
        """Load obstacles from CSV file for the prior knowledge area"""
        try:
            with open('obstacles_100m_above_sea_level.csv', 'r') as file:
                csv_reader = csv.reader(file)
                # Skip header row
                next(csv_reader)

                for row in csv_reader:
                    if len(row) >= 5:  # Make sure we have all needed values
                        try:
                            num_points = int(row[0])
                            x = float(row[1])
                            y = float(row[2])
                            z = float(row[3])
                            obs_type = row[4]

                            # Create new obstacle object
                            obstacle = Obstacle(num_points, x, y, z, obs_type)

                            # Only add if within bounds
                            if self.is_point_in_bounds([x, y, z]):
                                self.obstacles.append(obstacle)
                        except ValueError as e:
                            print(f"Skipping invalid row: {row}, Error: {e}")

                print(f"Loaded {len(self.obstacles)} obstacles within bounds")
        except Exception as e:
            print(f"Error loading obstacles: {str(e)}")

    ############################################# Helper Functions #########################################################################
    
    # Calculate the Euclidean distance between two points
    def calculate_distance(self, point1, point2):
            """Calculate Euclidean distance between two points"""
            return np.linalg.norm(np.array(point1) - np.array(point2))
      
    # Function to check if two points have line-of-sight visibility
    def path_clear(self,point1, point2):
        line = LineString([point1, point2])
        for polygon in self.polygons:
            shapely_polygon = Polygon(polygon)
            if line.crosses(shapely_polygon) or line.within(shapely_polygon):
                return False
        return True

    ############################################# Navigation algorithms ####################################################################
    
    def start_navigation(self):
        """Start navigation"""
        return self.heuristic_navigation()



    def transform_to_world_frame(self, point, pose):
        """Transform point from drone frame to world frame"""
        # Create rotation matrix from euler angles
        roll = pose.orientation.x_rad
        pitch = pose.orientation.y_rad
        yaw = pose.orientation.z_rad
        R = np.array([
            [np.cos(yaw) * np.cos(pitch),
             np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll),
             np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
            [np.sin(yaw) * np.cos(pitch),
             np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll),
             np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
            [-np.sin(pitch),
             np.cos(pitch) * np.sin(roll),
             np.cos(pitch) * np.cos(roll)]
        ])
        # Transform point to world frame
        point_world = np.dot(R, point) + np.array([pose.pos.x_m, pose.pos.y_m, pose.pos.z_m])
        return point_world
    def detect_obstacles_lidar(self):
        """Process LIDAR data to detect obstacles"""
        lidar_data = self.drone.getLidarData()
        #print(lidar_data.points)
        
        if len(lidar_data.points) == 0 or lidar_data.points[0] == 0:
            return False, []

        current_pose = self.drone.getPose()
        obs = []

        # Increase detection sensitivity
        detection_distance = self.safe_distance * 2  # Look further ahead for obstacles

        for i in range(0, len(lidar_data.points), 3):
            point = lidar_data.points[i:i + 3]
            if all(p == 0 for p in point):  # Skip invalid points
                continue

            # Convert point from drone frame to world frame
            point_world = self.transform_to_world_frame(point, current_pose)

            # Calculate distance in 3D space
            distance = np.linalg.norm(
                point_world - np.array([current_pose.pos.x_m, current_pose.pos.y_m, current_pose.pos.z_m]))

            if distance < detection_distance:
                obs.append((point_world, distance))
                # print(f"LIDAR detected obstacle at distance: {distance:.2f}m")

        return len(obs) > 0, obs
    def visualize_obstacle_detection(self, drone_pos, obstacles, target_point):
        """
        Visualize the drone, detected obstacles, and target point.
        
        Args:
            drone_pos: Current drone position (x, y)
            obstacles: List of (point_world, distance) tuples from LIDAR
            target_point: Current target point (x, y) from path
        """
        plt.figure(figsize=(10, 8))
        
        # Plot drone position
        plt.plot(drone_pos[0], drone_pos[1], 'bo', markersize=10, label='Drone')
        
        # Plot obstacles
        if obstacles:
            obstacle_points = np.array([obs[0] for obs in obstacles])
            plt.scatter(obstacle_points[:, 0], obstacle_points[:, 1], 
                       c='red', marker='x', s=100, label='Obstacles')
            
            # Draw lines from drone to obstacles
            for obs in obstacles:
                plt.plot([drone_pos[0], obs[0][0]], [drone_pos[1], obs[0][1]], 
                        'r--', alpha=0.3)
        
        # Plot target point
        plt.plot(target_point[0], target_point[1], 'g*', 
                markersize=10, label='Target Point')
        
        # Draw line from drone to target
        plt.plot([drone_pos[0], target_point[0]], 
                [drone_pos[1], target_point[1]], 'g--', 
                alpha=0.5, label='Path to Target')
        
        plt.title('Obstacle Detection Visualization')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def heuristic_navigation(self):
        """Navigate to midpoint using visibility graph and A* with dynamic replanning"""
        print("Starting navigation to midpoint...")
        
        #   VORONOI
        # self.visualize_polygons()
        voronoi_points = self.build_voronoi_diagram()
        for polygon in self.polygons:
            shapely_polygon = Polygon(polygon)
            for point in voronoi_points:
                if shapely_polygon.contains(Point(point)):
                    voronoi_points.remove(point)
        visibility_graph = self.voronoi_visibility_graph(voronoi_points)

        #   VISIBILITY
        # start_time = time.time()
        # visibility_graph = self.build_visibility_graph()
        # end_time = time.time()
        # build_time=end_time-start_time
        # # Print time taken
        # print(f"Visibility graph built in {build_time:.4f} seconds")
        # print(f"Number of nodes in graph: {len(visibility_graph)}")
        # print(f"Number of edges in graph: {sum(len(edges) for _, edges in visibility_graph.items()) // 2}")
        # print("Visibility graph calculation done")
       
        # Find shortest path using Dijkstra's
        start_time = time.time()
        start_point = tuple(self.start_pos[:2])  # Convert to tuple and take only x,y
        mid_point = tuple(self.mid_pos[:2])      # Convert to tuple and take only x,y
        path, total_distance = self.dijkstra_algorithm(visibility_graph, start_point, mid_point)
        end_time = time.time()

        build_time = end_time - start_time
        print(f"Dijkstra search completed in {build_time:.4f} seconds")
   
        if path is not None:
            print(f"Path found with length: {total_distance:.2f} meters")
            print("Path points:", path)
            
            return visibility_graph,path
        else:
            print("No path found")

    

    def voronoi_visibility_graph(self, voronoi_points):
        """
        Generate a graph from Voronoi points that follows the Voronoi diagram structure.
        Ensures connectivity between start and end points while maintaining safety.
        
        Args:
            voronoi_points: List of (x,y) tuples representing Voronoi vertices
            
        Returns:
            Dictionary where keys are nodes and values are lists of (neighbor, distance) tuples
        """
        # Create 2D points for start and mid
        start_point = (self.start_pos[0], self.start_pos[1])
        mid_point = (self.mid_pos[0], self.mid_pos[1])
        
        # Parameters for graph construction
        MAX_EDGE_LENGTH = 500  # Increased for better connectivity
        K_NEAREST = 8  # Increased number of neighbors
        
        # Combine start, mid and Voronoi points
        all_points = [start_point, mid_point] + voronoi_points
        
        # Initialize the graph
        graph = {point: [] for point in all_points}
        
        # Helper function to find K nearest neighbors
        def get_k_nearest(point, points, k):
            distances = [(p, self.calculate_distance(point, p)) for p in points if p != point]
            return sorted(distances, key=lambda x: x[1])[:k]
        
        # Connect start and mid points with direct path if possible
        if self.path_clear(start_point, mid_point):
            distance = self.calculate_distance(start_point, mid_point)
            graph[start_point].append((mid_point, distance))
            graph[mid_point].append((start_point, distance))
        
        # Connect start and mid points to nearest valid Voronoi points
        for point in [start_point, mid_point]:
            # Try to connect to more points for better connectivity
            nearest_points = get_k_nearest(point, voronoi_points, K_NEAREST * 2)
            connected = False
            
            for neighbor, distance in nearest_points:
                if distance <= MAX_EDGE_LENGTH and self.path_clear(point, neighbor):
                    graph[point].append((neighbor, distance))
                    graph[neighbor].append((point, distance))
                    connected = True
            
            # If no connections were made, try with increased distance
            if not connected:
                for neighbor, distance in nearest_points:
                    if self.path_clear(point, neighbor):
                        graph[point].append((neighbor, distance))
                        graph[neighbor].append((point, distance))
                        break
        
        # Connect Voronoi points to their nearest neighbors
        for i, point1 in enumerate(voronoi_points):
            # Get K nearest neighbors
            nearest = get_k_nearest(point1, voronoi_points, K_NEAREST)
            
            for neighbor, distance in nearest:
                # Skip if already connected
                if any(n[0] == neighbor for n in graph[point1]):
                    continue
                    
                # Check if edge meets criteria
                if distance <= MAX_EDGE_LENGTH and self.path_clear(point1, neighbor):
                    # Check if midpoint of edge is safe from obstacles
                    mid_x = (point1[0] + neighbor[0]) / 2
                    mid_y = (point1[1] + neighbor[1]) / 2
                    mid_point_safe = True
                    
                    # Check if midpoint maintains safe distance from obstacles
                    for polygon in self.polygons:
                        poly = Polygon(polygon)
                        if Point(mid_x, mid_y).distance(poly) < self.safe_distance:
                            mid_point_safe = False
                            break
                    
                    if mid_point_safe:
                        graph[point1].append((neighbor, distance))
                        graph[neighbor].append((point1, distance))
        
        # Ensure graph connectivity through additional connections if needed
        def find_connected_components():
            visited = set()
            components = []
            
            def dfs(node, component):
                visited.add(node)
                component.add(node)
                for neighbor, _ in graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor, component)
            
            for node in graph:
                if node not in visited:
                    component = set()
                    dfs(node, component)
                    components.append(component)
            
            return components
        
        # Find disconnected components
        components = find_connected_components()
        
        # If there are multiple components, try to connect them
        if len(components) > 1:
            # Sort components by size
            components.sort(key=len, reverse=True)
            
            # Try to connect smaller components to the largest one
            main_component = components[0]
            for other_component in components[1:]:
                # Find closest pair of points between components
                min_distance = float('inf')
                best_pair = None
                
                for p1 in main_component:
                    for p2 in other_component:
                        dist = self.calculate_distance(p1, p2)
                        if dist < min_distance and self.path_clear(p1, p2):
                            min_distance = dist
                            best_pair = (p1, p2)
                
                # Connect the components if a valid pair was found
                if best_pair:
                    p1, p2 = best_pair
                    graph[p1].append((p2, min_distance))
                    graph[p2].append((p1, min_distance))
        
        return graph
   
    def a_star_search(self, visibility_graph, start_point, end_point):

        # Convert numpy arrays to tuples if needed
        if isinstance(start_point, np.ndarray):
            start_point = tuple(start_point[:2])  # Take only x,y coordinates
        if isinstance(end_point, np.ndarray):
            end_point = tuple(end_point[:2])  # Take only x,y coordinates

        # Check if start and end points are in the graph
        if start_point not in visibility_graph or end_point not in visibility_graph:
            print(f"Start point {start_point} or end point {end_point} not in visibility graph")
            return None, float('inf')

        # Initialize open and close
        open_set = {start_point}
        closed_set = set()

        # Dictionary to store the cost from start to each node
        g_score = {node: float('inf') for node in visibility_graph}
        g_score[start_point] = 0

        # Dictionary to store the estimated total cost from start to goal through each node
        f_score = {node: float('inf') for node in visibility_graph}
        f_score[start_point] = self.calculate_distance(start_point, end_point)

        # Dictionary to store the previous node in the optimal path
        came_from = {}

        while open_set:
            # Get the node with the lowest f_score
            current = min(open_set, key=lambda node: f_score[node])

            # If we reached the end point, reconstruct and return the path
            if current == end_point:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_point)
                path.reverse()  # Reverse to get path from start to end
                return path, g_score[end_point]

            # Move current from open_set to closed_set
            open_set.remove(current)
            closed_set.add(current)

            # Check all neighbors of current node
            for neighbor, distance in visibility_graph[current]:
                # Skip if neighbor is already evaluated
                if neighbor in closed_set:
                    continue

                # Calculate tentative g_score
                tentative_g_score = g_score[current] + distance

                # If neighbor is not in open_set, add it
                if neighbor not in open_set:
                    open_set.add(neighbor)
                # If this path to neighbor is not better than previous one, skip
                elif tentative_g_score >= g_score[neighbor]:
                    continue

                # This path is the best so far, record it
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.calculate_distance(neighbor, end_point)

        print("No path found between the points")
        return None, float('inf')

    def build_visibility_graph(self):
        """Generate visibility graph using start and mid points from self"""
        # Create 2D points for start and mid
        start_point = (self.start_pos[0], self.start_pos[1])
        mid_point = (self.mid_pos[0], self.mid_pos[1])
        
        # Collect all vertices from polygons
        polygon_vertices = []

        for polygon in self.polygons:

            for vertex in polygon:
                polygon_vertices.append(tuple(vertex))
        
        # Combine start, mid and polygon vertices
        nodes = [start_point, mid_point] + polygon_vertices
        
        # Initialize the graph
        graph = {node: [] for node in nodes}

        # Set maximum edge length (you can adjust this value)
        MAX_EDGE_LENGTH = 500  # meters
        
        # Build connections between nodes
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1:]:
                if node1 != node2 and self.path_clear(node1, node2):
                    distance = self.calculate_distance(node1, node2)
                    if distance <= MAX_EDGE_LENGTH:
                        graph[node1].append((node2, distance))
                        graph[node2].append((node1, distance))

        return graph
    

    def dijkstra_algorithm(self, visibility_graph, start_point=None, end_point=None):
        """
        Implements Dijkstra's algorithm to find shortest path in visibility graph
        Args:
            visibility_graph: Dict of node -> [(neighbor, distance)]
            start_point: Starting point (x,y). If None, uses self.start_pos
            end_point: End point (x,y). If None, uses self.mid_pos
        Returns:
            path: List of points forming shortest path
            distance: Total distance of path
        """
        # Convert numpy arrays to tuples if needed
        if isinstance(start_point, np.ndarray):
            start_point = tuple(start_point[:2])  # Take only x,y coordinates
        if isinstance(end_point, np.ndarray):
            end_point = tuple(end_point[:2])  # Take only x,y coordinates

        # Use class start/mid points if none provided
        if start_point is None:
            start_point = (self.start_pos[0], self.start_pos[1])
        if end_point is None:
            end_point = (self.mid_pos[0], self.mid_pos[1])

        # Verify points exist in graph
        if start_point not in visibility_graph or end_point not in visibility_graph:
            print(f"Start point {start_point} or end point {end_point} not in visibility graph")
            return None, float('inf')

        # Initialize distances and previous nodes
        distances = {node: float('inf') for node in visibility_graph}
        distances[start_point] = 0
        previous = {node: None for node in visibility_graph}
        
        # Priority queue of unvisited nodes: (distance, node)
        unvisited = [(0, start_point)]
        visited = set()

        while unvisited:
            # Get node with minimum distance
            current_distance, current_node = min(unvisited)
            unvisited.remove((current_distance, current_node))
            
            # If we reached the end, reconstruct and return path
            if current_node == end_point:
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = previous[current_node]
                path.reverse()
                return path, distances[end_point]
            
            # Skip if already visited
            if current_node in visited:
                continue
            visited.add(current_node)
            
            # Check all neighbors
            for neighbor, edge_distance in visibility_graph[current_node]:
                if neighbor in visited:
                    continue
                    
                # Calculate new distance
                new_distance = distances[current_node] + edge_distance
                
                # Update if new distance is shorter
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
                    unvisited.append((new_distance, neighbor))
        
        # If we get here, no path was found
        print("No path found between points")
        return None, float('inf')
        

    ################################################ Plotting ##############################################################################
    def build_voronoi_diagram(self):
        """
        Build a Voronoi diagram from polygon vertices and return a list of Voronoi points.
        Returns a list of (x,y) points that can be used for path planning.
        """
        # Collect all vertices from polygons
        points = []
        for polygon in self.polygons:
            points.extend(polygon)
            
            # Add intermediate points along polygon edges
            for i in range(len(polygon)):
                p1 = np.array(polygon[i])
                p2 = np.array(polygon[(i + 1) % len(polygon)])
                
                # Add points every 10 meters along edges
                edge_length = np.linalg.norm(p2 - p1)
                num_points = max(1, int(edge_length / 10))
                
                for j in range(1, num_points):
                    t = j / num_points
                    intermediate_point = p1 + t * (p2 - p1)
                    points.append(tuple(intermediate_point))
        
        # Convert to numpy array
        points = np.array(points)
        
        # Add start and mid points to ensure they're considered in the Voronoi diagram
        start_point = np.array([self.start_pos[0], self.start_pos[1]])
        mid_point = np.array([self.mid_pos[0], self.mid_pos[1]])
        points = np.vstack((points, start_point, mid_point))
        
        # Add bounds points with better coverage
        bounds_size = 100  # meters beyond the prior knowledge bounds
        x_min = min(p[0] for p in self.prior_knowledge_bounds) - bounds_size
        x_max = max(p[0] for p in self.prior_knowledge_bounds) + bounds_size
        y_min = min(p[1] for p in self.prior_knowledge_bounds) - bounds_size
        y_max = max(p[1] for p in self.prior_knowledge_bounds) + bounds_size
        
        # Create a grid of boundary points
        num_boundary_points = 10
        x_boundary = np.linspace(x_min, x_max, num_boundary_points)
        y_boundary = np.linspace(y_min, y_max, num_boundary_points)
        
        boundary_points = []
        for x in x_boundary:
            boundary_points.append([x, y_min])
            boundary_points.append([x, y_max])
        for y in y_boundary:
            boundary_points.append([x_min, y])
            boundary_points.append([x_max, y])
            
        # Add corners with some offset to improve edge behavior
        corner_offset = bounds_size / 2
        corners = [
            [x_min - corner_offset, y_min - corner_offset],
            [x_min - corner_offset, y_max + corner_offset],
            [x_max + corner_offset, y_min - corner_offset],
            [x_max + corner_offset, y_max + corner_offset]
        ]
        
        # Combine all points
        all_points = np.vstack((points, boundary_points, corners))
        
        # Compute Voronoi diagram
        vor = Voronoi(all_points)
        
        # Collect Voronoi vertices with improved filtering
        voronoi_points = []
        for vertex in vor.vertices:
            x, y = vertex
            
            # Check if point is within expanded bounds (allow some margin)
            margin = 50  # meters
            if (x_min - margin <= x <= x_max + margin and 
                y_min - margin <= y <= y_max + margin):
                
                # Check if point maintains safe distance from obstacles
                point_valid = True
                min_distance = float('inf')
                
                for polygon in self.polygons:
                    poly = Polygon(polygon)
                    point = Point(x, y)
                    distance = point.distance(poly)
                    min_distance = min(min_distance, distance)
                    
                    if distance < self.safe_distance:
                        point_valid = False
                        break
                
                # Accept points that are either:
                # 1. Safe distance from obstacles
                # 2. Or strategically useful (close to start/mid points)
                if point_valid or (
                    np.linalg.norm(np.array([x, y]) - start_point) < self.safe_distance * 2 or
                    np.linalg.norm(np.array([x, y]) - mid_point) < self.safe_distance * 2
                ):
                    voronoi_points.append((x, y))
        
        # Add additional connection points near start and goal
        for point in [start_point, mid_point]:
            for angle in np.linspace(0, 2*np.pi, 8):
                # Add points in a circle around start/goal
                offset_distance = self.safe_distance * 1.5
                new_point = (
                    point[0] + offset_distance * np.cos(angle),
                    point[1] + offset_distance * np.sin(angle)
                )
                # Check if the new point is safe
                point_valid = True
                for polygon in self.polygons:
                    poly = Polygon(polygon)
                    if Point(new_point).distance(poly) < self.safe_distance:
                        point_valid = False
                        break
                if point_valid:
                    voronoi_points.append(new_point)
        
        return voronoi_points
    
    def visualize_voronoi_diagram(self, voronoi_points):
        """
        Visualize the Voronoi diagram points along with polygons.
        
        Args:
            voronoi_points: List of (x,y) points from Voronoi diagram
        """
        plt.figure(figsize=(12, 8))
        
        # Plot polygons
        for polygon in self.polygons:
            # Close the polygon for plotting
            poly = polygon + [polygon[0]]
            xs, ys = zip(*poly)
            plt.fill(xs, ys, alpha=0.2)
            plt.plot(xs, ys, 'b-', linewidth=2)
        
        # Plot Voronoi points
        if voronoi_points:
            xs = [p[0] for p in voronoi_points]
            ys = [p[1] for p in voronoi_points]
            plt.plot(xs, ys, 'r.', markersize=8, label='Voronoi Points')
        
        # Plot bounds
        x_coords = [x[0] for x in self.prior_knowledge_bounds]
        y_coords = [x[1] for x in self.prior_knowledge_bounds]
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])
        plt.plot(x_coords, y_coords, 'k--', linewidth=2, label='Bounds')
        
        plt.title('Voronoi Diagram')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    def visualize_visibility_graph(self, visibility_graph, path=None, path2=None):
        """
        Visualize the visibility graph and optionally two paths.
        Shows:
        - Polygons with different colors
        - Graph edges in blue
        - Voronoi vertices as red dots
        - Start/Mid points as larger dots
        - Planned path in green (if provided)
        - Real trajectory in pink (if provided)
        
        Args:
            visibility_graph: Dictionary where keys are nodes and values are lists of (neighbor, distance) tuples
            path: Optional list of points representing the planned path to visualize
            path2: Optional list of points representing the real drone trajectory to visualize
        """
        plt.figure(figsize=(12, 10))
        
        # Plot polygons with different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.polygons)))
        for i, polygon in enumerate(self.polygons):
            # Close the polygon for plotting
            poly = polygon + [polygon[0]]
            xs, ys = zip(*poly)
            plt.fill(xs, ys, color=colors[i], alpha=0.2)
            plt.plot(xs, ys, color=colors[i], linewidth=1)
        
        # Plot visibility graph edges
        for node, edges in visibility_graph.items():
            for neighbor, _ in edges:
                plt.plot([node[0], neighbor[0]], [node[1], neighbor[1]], 
                        color='blue', linewidth=0.3, alpha=0.2)

        # Plot Voronoi vertices (excluding start and mid points)
        start_point = (self.start_pos[0], self.start_pos[1])
        mid_point = (self.mid_pos[0], self.mid_pos[1])
        voronoi_vertices = [point for point in visibility_graph.keys() 
                          if point != start_point and point != mid_point]
        
        if voronoi_vertices:
            xs = [p[0] for p in voronoi_vertices]
            ys = [p[1] for p in voronoi_vertices]
            plt.plot(xs, ys, 'r.', markersize=8, label='Voronoi Vertices')

        # Plot the planned path if provided
        if path and len(path) > 1:
            # Extract x and y coordinates from the path
            path_xs = [point[0] for point in path]
            path_ys = [point[1] for point in path]
            
            # Plot the path line with a different color (green)
            plt.plot(path_xs, path_ys, 'g-', linewidth=3, label='Planned Path')
            
            # Plot the path points with the same color
            plt.plot(path_xs, path_ys, 'go', markersize=6)
            
            # Annotate path points with their order
            for i, (x, y) in enumerate(zip(path_xs, path_ys)):
                plt.annotate(f'{i}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontweight='bold')

        # Plot the real trajectory if provided
        if path2 and len(path2) > 1:
            # Extract x and y coordinates from path2
            path2_xs = [point[0] for point in path2]
            path2_ys = [point[1] for point in path2]
            
            # Plot the real trajectory in pink
            plt.plot(path2_xs, path2_ys, 'pink', linewidth=3, label='Real Trajectory')
            plt.plot(path2_xs, path2_ys, 'pink', marker='o', markersize=4)

        # Plot start and mid points with larger markers
        plt.plot(self.start_pos[0], self.start_pos[1], 'go', 
                markersize=12, label='Start', markeredgecolor='black')
        plt.plot(self.mid_pos[0], self.mid_pos[1], 'yo', 
                markersize=12, label='Mid', markeredgecolor='black')
        
        # Set fixed axis limits based on prior_knowledge_bounds
        x_coords = [x[0] for x in self.prior_knowledge_bounds]
        y_coords = [x[1] for x in self.prior_knowledge_bounds]
        plt.xlim(min(x_coords), max(x_coords))
        plt.ylim(min(y_coords), max(y_coords))
        
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title('Visibility Graph with Paths')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_polygons(self):
        """
        Visualize the polygons with different colors and labels.
        Each polygon will be displayed with a unique color and its vertices connected.
        """
        plt.figure(figsize=(12, 8))

        # Define a color cycle for different polygons
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.polygons)))

        # Plot each polygon
        for i, polygon in enumerate(self.polygons):
            # Convert polygon points to numpy arrays for easier plotting
            points = np.array(polygon)

            # Add first point at end to close the polygon
            points = np.vstack([points, points[0]])

            # Plot polygon edges
            plt.plot(points[:, 0], points[:, 1], '-',
                     color=colors[i], linewidth=2,
                     label=f'Polygon {i + 1}')

            # Plot vertices
            plt.plot(points[:-1, 0], points[:-1, 1], 'o',
                     color=colors[i], markersize=6)

            # Add vertex labels
            for j, (x, y) in enumerate(polygon):
                plt.annotate(f'P{j + 1}', (x, y),
                             xytext=(5, 5), textcoords='offset points')

        # Plot bounds
        x_coords = [x[0] for x in self.prior_knowledge_bounds]
        y_coords = [x[1] for x in self.prior_knowledge_bounds]
        x_coords.append(x_coords[0])  # Close the boundary
        y_coords.append(y_coords[0])
        plt.plot(x_coords, y_coords, 'k--', linewidth=2, label='Bounds')

        # Set plot properties
        plt.grid(True)
        plt.axis('equal')
        plt.title('Obstacle Polygons')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.legend()

        # Show the plot
        plt.show(block=True)

    
    