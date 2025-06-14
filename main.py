from DroneClient import DroneClient
from Vector import *
from Navigator import MidWayNavigator, GoalNavigator
import time
import matplotlib.pyplot as plt
import numpy as np

def visualize_path(graph, planned_path, trajectory_to_mid, polygons, trajectory_to_end, lidar_data):
    plt.figure(figsize=(15, 12))
    
    # Plot polygons (obstacles)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(polygons)))
    for i, polygon in enumerate(polygons):
        # Close the polygon for plotting
        poly = polygon + [polygon[0]]
        xs, ys = zip(*poly)
        plt.fill(xs, ys, color=colors[i], alpha=0.2)
        plt.plot(xs, ys, color=colors[i], linewidth=1, label=f'Obstacle {i+1}')
    
    # Plot visibility graph edges with very light blue
    for node, edges in graph.items():
        for neighbor, _ in edges:
            plt.plot([node[0], neighbor[0]], [node[1], neighbor[1]], 
                    color='lightblue', linewidth=0.3, alpha=0.2)
    
    # Plot planned path to midpoint with green dashed line
    if planned_path and len(planned_path) > 1:
        path_xs = [point[0] for point in planned_path]
        path_ys = [point[1] for point in planned_path]
        plt.plot(path_xs, path_ys, 'g--', linewidth=2, label='Planned Path')
        
        # Add arrows to show direction
        for i in range(len(path_xs)-1):
            plt.arrow(path_xs[i], path_ys[i], 
                     (path_xs[i+1] - path_xs[i])*0.1, 
                     (path_ys[i+1] - path_ys[i])*0.1,
                     head_width=5, head_length=10, fc='g', ec='g')
    
    # Plot actual trajectory to midpoint with blue line
    if trajectory_to_mid and len(trajectory_to_mid) > 1:
        traj_xs = [point[0] for point in trajectory_to_mid]
        traj_ys = [point[1] for point in trajectory_to_mid]
        plt.plot(traj_xs, traj_ys, 'b-', linewidth=2, label='Trajectory to Mid')
    
    # Plot actual trajectory to endpoint with red line
    if trajectory_to_end and len(trajectory_to_end) > 1:
        end_xs = [point[0] for point in trajectory_to_end]
        end_ys = [point[1] for point in trajectory_to_end]
        plt.plot(end_xs, end_ys, 'r-', linewidth=2, label='Trajectory to End')
    
    # Plot LIDAR points if available
    if lidar_data:
        lidar_xs = [point[0] for point in lidar_data]
        lidar_ys = [point[1] for point in lidar_data]
        plt.scatter(lidar_xs, lidar_ys, c='purple', s=10, alpha=0.5, label='LIDAR Points')
    
    # Plot start, mid, and end points with distinct markers
    if trajectory_to_mid:
        plt.plot(trajectory_to_mid[0][0], trajectory_to_mid[0][1], 'go', 
                markersize=15, label='Start', markeredgecolor='black')
        plt.plot(trajectory_to_mid[-1][0], trajectory_to_mid[-1][1], 'yo', 
                markersize=15, label='Mid', markeredgecolor='black')
    if trajectory_to_end:
        plt.plot(trajectory_to_end[-1][0], trajectory_to_end[-1][1], 'ro', 
                markersize=15, label='End', markeredgecolor='black')
    
    plt.title('Complete Navigation Visualization')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.axis('equal')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

def visualize_trajectory_and_lidar(trajectory_points, lidar_points):
    plt.figure(figsize=(10, 8))
    
    # Plot trajectory points
    if trajectory_points and len(trajectory_points) > 1:
        traj_xs = [point[0] for point in trajectory_points]
        traj_ys = [point[1] for point in trajectory_points]
        
        # Plot lines connecting trajectory points
        plt.plot(traj_xs, traj_ys, 'b-', linewidth=1.5, label='Drone Path')
        
        # Plot individual trajectory points
        plt.scatter(traj_xs, traj_ys, c='blue', s=30, alpha=0.6, label='Trajectory Points')
        
        # Mark start and end points
        plt.plot(traj_xs[0], traj_ys[0], 'go', markersize=12, label='Start')
        plt.plot(traj_xs[-1], traj_ys[-1], 'ro', markersize=12, label='End')
        
        # Add arrows to show direction
        for i in range(len(traj_xs)-1):
            if i % 5 == 0:  # Add arrow every 5 points to avoid cluttering
                plt.arrow(traj_xs[i], traj_ys[i],
                         (traj_xs[i+1] - traj_xs[i])*0.2,
                         (traj_ys[i+1] - traj_ys[i])*0.2,
                         head_width=2, head_length=4, fc='b', ec='b', alpha=0.5)
    
    # Plot LIDAR points
    if lidar_points:
        lidar_xs = [point[0] for point in lidar_points]
        lidar_ys = [point[1] for point in lidar_points]
        plt.scatter(lidar_xs, lidar_ys, c='red', s=15, alpha=0.3, label='LIDAR Points')
    
    plt.title('Drone Trajectory and LIDAR Data')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Add text box with statistics
    if trajectory_points and len(trajectory_points) > 1:
        total_distance = sum(np.sqrt((trajectory_points[i+1][0] - trajectory_points[i][0])**2 + 
                                   (trajectory_points[i+1][1] - trajectory_points[i][1])**2)
                           for i in range(len(trajectory_points)-1))
        
        stats_text = f'Total Points: {len(trajectory_points)}\n'
        stats_text += f'Distance: {total_distance:.2f}m\n'
        if lidar_points:
            stats_text += f'LIDAR Points: {len(lidar_points)}'
        
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top',
                fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    client = DroneClient()
    client.connect()
    print(client.isConnected())

    # the plane on the z axis in which all the positions are found
    plane = -70
    # find the path from each position,
    # to the next one on the list

    start = Vector_2D(-875,-1050)
    mid = Vector_2D(-700, -700)
    end = Vector_2D(-560,-590)

    client.setAtPosition(start.x, start.y, plane)
    time.sleep(5)

    # # Start -> Mid
    mn1 = MidWayNavigator((start.x, start.y), (mid.x, mid.y), plane, client)
    # pos1, polygons_not_filter, visibility_graph, path, lid1, time_start_mid = 
    graph,path,trajectory,polygons,time_start_mid =mn1.fly_to_mid_point()
    print("Arrived to MID")

    # Mid -> End
    mn2 = GoalNavigator(mid, end, plane, client)
    pos2, lid2, time_mid_end = mn2.fly_to_goal_point()
    print("Arrived to GOAL")
    #Compute total time
    total_time = time_start_mid + time_mid_end
    print(f"Total time from START to GOAL: {total_time:.2f} seconds")
    # visualize_trajectory_and_lidar(pos2, lid2)
    visualize_path(graph, path, trajectory, polygons, pos2, lid2)
    