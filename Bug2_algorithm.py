import time
from typing import Generator, List, Optional, Set, Tuple, Dict, Iterable
import numpy as np
from DroneClient import *
from Vector import *


class Bug2():

    # Drone parameters
    pos: []
    lidar_data:[]
    client: DroneClient
    plane: float

    # Safety parameters
    colision_radius: float = 3
    connection_dist: float = 9
    sensor_range: float = 35
    goal_epsilon: float = 3 # Given in part 1
    boundary_dist: float = 3
    corridor_dist: float = 9

    # Points params
    obstacle_points: Dict[Vector_2D, int] = {}
    nearby_points: List[Vector_2D] = []

    # Navigation params
    position: Vector_2D = Vector_2D(0, 0)
    orientation: float = 0
    orientation_3D: Quaternion = Quaternion(0, 0, 0, 1)

    # Path params
    goal: Vector_2D = Vector_2D(0, 0)
    cur_corridor_width: float = math.inf
    vertigo: int = 0
    start_position: Vector_2D = Vector_2D(0, 0)  # Starting pos for M-line
    boundary_start_position: Vector_2D = None  # Pos where boundary following began
    global_goal: Vector_2D = Vector_2D(0, 0)
    min_dist: float

    def __init__(self, client: DroneClient, plane: float ,start_position : Vector_2D)-> None: 
        self.client = client
        self.plane = plane
        self.pos = []
        self.lidar_data = []
        self.start_position=start_position
    
    def toBodyFrame(self, point: Vector_2D):
        return (point - self.position).rotate(-self.orientation)

    def toWorldFrame(self, point: Vector_2D):
        return point.rotate(self.orientation) + self.position

    def autoFlyTo(self, point: Vector_2D, limit: float = 10): #to check
        length = point.length()
        if abs(Vector_2D(1, 0).angle(point)) > math.pi / 6: 
            self.vertigo = 100
        velocity: float
        if length < 0.0001:
            velocity = 0
        else:
            safety_velocity = min((p.length()
                                   for p in self.nearby_points), default=math.inf)
            if self.vertigo <= 0:
                safety_velocity *= 1.5
            else:
                self.vertigo -= 1
            velocity = min(limit, safety_velocity, self.goal.length() / 2)
        world_point = self.toWorldFrame(point)
        self.client.flyToPosition(
            world_point.x, world_point.y, self.plane, velocity)

    def detectObstacles(self) -> Generator[Vector_2D, None, None]:
        points_detected = self.client.getLidarData().points
        if len(points_detected) < 3: # Not enough points to form a point
            return

        for i in range(0, len(points_detected), 3):
            point = Quaternion(
                points_detected[i], points_detected[i + 1], points_detected[i + 2], 0)
            rotated = self.orientation_3D * point * self.orientation_3D.conjugate()
            world_point = Vector_2D(rotated.x, rotated.y) + self.position
            self.lidar_data.append([world_point.x,world_point.y])
            yield world_point

    def flush_points(self):
        flushed = []
        for p, i in self.obstacle_points.items():
            if i > 250:
                flushed.append(p)
            else:
                # the point stays for another iteration
                self.obstacle_points[p] += 1
        for p in flushed:
            self.obstacle_points.pop(p, None)

    def updateEnvironment(self): #to check
        pose = self.client.getPose()
        self.orientation_3D = Quaternion.from_euler_angles(pose.orientation.x_rad,
                                                          pose.orientation.y_rad,
                                                          pose.orientation.z_rad)
        position = Vector_2D(pose.pos.x_m, pose.pos.y_m)
        self.pos.append([pose.pos.x_m, pose.pos.y_m])
        world_goal = self.toWorldFrame(self.goal)
        self.position = position
        self.orientation = pose.orientation.z_rad
        self.goal = self.toBodyFrame(world_goal)
        self.cur_corridor_width = self.find_the_tunnel_width()

        for p in self.detectObstacles():
            self.obstacle_points[p.round()] = 0

        self.flush_points()
        self.nearby_points = [self.toBodyFrame(p) for p in self.obstacle_points.keys()
                              if 1 < p.distance(self.position) < self.sensor_range]  

    def checkObstaclesInPath(self):
        return any(checkoverlapCircle(Vector_2D(0, 0), self.goal, p, self.colision_radius) for p in self.nearby_points)

    def checkPointsConnected(self, p1: Vector_2D, p2: Vector_2D):
        if(p1.distance(p2) > self.connection_dist):
            return False
        return True

    def findPath(self, goal: Vector_2D,start: Vector_2D):
        # Store starting position for M-line
        # Current position becomes start position
        
        self.goal = self.toBodyFrame(goal)
        self.updateEnvironment()
        self.start_position = start
        following_boundary = False
        boundary_following_planner = self.followBoundary()
        self.min_dist = np.inf
        last_direction = self.goal.rotate(-self.orientation).normalize()

        while True:
            self.updateEnvironment()

            if self.goal.length() <= self.goal_epsilon: # Arrived at goal
                self.autoFlyTo(Vector_2D(0, 0))
                return
            
            if following_boundary:
                limit = 9
                point = next(boundary_following_planner, None)
                
                # Add M-line check here
                if point is not None and self.is_on_M_line(point):
                    # We've hit the M-line and are closer to goal, go back to goal-seeking
                    self.min_dist = np.inf
                    following_boundary = False
                    print("start start_position",self.start_position)
                    print("global goal",self.global_goal)
                    print("current position",self.position)
                    print("current goal",self.toWorldFrame(self.goal))
                    print("Hit M line")
                elif point is None:
                    self.min_dist = np.inf
                    following_boundary = False
                else:
                    self.autoFlyTo(point, limit=limit)

            else:
                # find next point to fly to in the goal direction if none is found 
                # then this means we should follow the boundary so we switch to follow boundary mode
                # we go to point that still makes progress toward goal but still doesn't hit aany obstacle
                # this makes sure we start following boundary when we are not "that far" from the obstacle and not that close.
                point = self.go_to_goal()
                if point is None:
                    # Store position where boundary following begins (in world frame)
                    self.boundary_start_position = self.position
                    boundary_following_planner = self.followBoundary(last_direction)
                    following_boundary = True
                else:
                    self.autoFlyTo(point, 9)
                    last_direction = point.rotate(-self.orientation).normalize()

            time.sleep(0.02)

    def go_to_goal(self):
    #if path is obstructed then return the closest point to the goal that is not obstructed
            if self.checkObstaclesInPath():
                discontinuity_points = self.findDiscontinuityPoints()
                if discontinuity_points is None:
                    return

                the_closest_point_to_the_goal = min(discontinuity_points,
                                                    key=lambda p: self.calculate_total_path_distance(p))
                distance = self.calculate_total_path_distance(the_closest_point_to_the_goal)
                if self.min_dist < distance:
                    return None
                else:
                    self.min_dist = distance
                    return the_closest_point_to_the_goal
            else:
                return self.goal

    def getBlockingObstacle(self, path: Vector_2D):
        blocking_obstacle = []
        counter_clockwise_points = []
        clockwise_points = []

        self.nearby_points.sort(key=lambda p: path.angle(p))

        for point in self.nearby_points:
            if checkoverlapCircle(Vector_2D(0, 0), path, point, self.colision_radius):
                blocking_obstacle.append(point)
            elif path.angle(point) > 0:
                counter_clockwise_points.append(point)
            else:
                clockwise_points.append(point)

        for point in counter_clockwise_points:
            if any(self.checkPointsConnected(point, p) for p in blocking_obstacle):
                blocking_obstacle.append(point)

        for point in reversed(clockwise_points):
            if any(self.checkPointsConnected(point, p) for p in blocking_obstacle):
                blocking_obstacle.append(point)
        return blocking_obstacle
    
    def calculate_total_path_distance(self, point: Vector_2D):
        return point.length() + point.distance(self.goal)
    
    def findDiscontinuityPoints(self):
        obstacle = self.getBlockingObstacle(self.goal)

        cw = min(obstacle, key=lambda p: self.goal.angle(p))
        cw_avoidance_angle = getFoVCoverage(cw, self.boundary_dist)
        if cw_avoidance_angle is None:
            return None
        cw = cw.rotate(-cw_avoidance_angle)

        counter_cw = max(obstacle, key=lambda p: self.goal.angle(p))
        ccw_avoidance_angle = getFoVCoverage(counter_cw, self.boundary_dist)
        if ccw_avoidance_angle is None:
            return None
        counter_cw = counter_cw.rotate(ccw_avoidance_angle)

        return cw, counter_cw

    def getFollowedBoundary(self, followed_point: Vector_2D) -> Generator[Vector_2D, None, None]:
        corridor_ratio = self.cur_corridor_width / self.corridor_dist
        resize = min(1, 0.9 * corridor_ratio)

        for point in self.nearby_points:
            if point.distance(followed_point) < resize * self.corridor_dist:
                yield point

    def followBoundary(self, prev_path_hint: Optional[Vector_2D] = None) -> Generator[Vector_2D, None, None]:
        min_followed_distance = math.inf
        right_follow = None # true
        prev_followed_obstacle = [self.toWorldFrame(p) for p in self.getBlockingObstacle(self.goal)]
        while True:
            # Ensure that the obstacle contains only points that are currently nearby
            followed_obstacle = set(self.toBodyFrame(p).round()
                                    for p in prev_followed_obstacle)
            #why ? lanek 7reme
            followed_obstacle.intersection_update(p.round()
                                                  for p in self.nearby_points)
            # ensure that obstacles in the way to the followed obstacle are not ignored,
            followed_obstacle.update(
                p for p in self.nearby_points if p.length() < self.boundary_dist * 1.5)

            followed_point = min(followed_obstacle,
                                 key=lambda p: p.length(), default=None)
            if followed_point is None:
                return

            followed_obstacle.update(
                self.getFollowedBoundary(followed_point))
            prev_followed_obstacle = [
                self.toWorldFrame(p) for p in followed_obstacle]
            # choose only one direction MAY
            if right_follow is None:
                # helps convince pyright linter that followed point is not None in this branch
                fp = followed_point
                # if no path hint is available, choose the direction based on the path to the goal
                path_hint = prev_path_hint.rotate(
                    self.orientation) if prev_path_hint is not None else self.goal

                # find the direction to follow that is closest to the path the drone is already going towards
                right_follow = min(
                    [True, False], key=lambda b: abs(self.getNextFollowPoint(fp, b).angle(path_hint)))

            flight_direction = self.getNextFollowPoint(
                followed_point, right_follow)

            yield flight_direction

    def find_the_tunnel_width(self):
        closest_point = min(
            self.nearby_points, key=lambda p: p.length(), default=None)

        if closest_point is None:
            return math.inf

        opposing_distance = min((p.length() for p in self.nearby_points
                                 if abs(closest_point.angle(p)) > math.pi / 2), default=math.inf)
        return closest_point.length() + opposing_distance

    def is_on_M_line(self, current_position: Vector_2D, tolerance: float = 5.0) -> bool:
        # If boundary following hasn't started yet, or no start position is stored
        if self.boundary_start_position is None:
            return False
            
        # Convert points to world frame for consistent calculations
        world_current = self.toWorldFrame(current_position)
        world_goal = self.toWorldFrame(self.goal)
        
        # Calculate the distance from current_position to the M-line
        start_to_goal = world_goal - self.start_position
        start_to_current = world_current - self.start_position
        
        # Project the vector from start to current onto the start to goal vector
        if start_to_goal.length() < 0.0001:
            return False  # Avoid division by zero
        
        # Calculate the projection of start_to_current onto start_to_goal
        t = start_to_current.dot(start_to_goal) / start_to_goal.dot(start_to_goal)
        
        # Find the closest point on the line
        closest_point = self.start_position + start_to_goal * t
        
        # Calculate distance to the line
        distance_to_line = world_current.distance(closest_point)
        
        # Check if we're on the line and closer to the goal than when we started boundary following
        on_line = distance_to_line <= tolerance
        making_progress = world_current.distance(world_goal) < self.boundary_start_position.distance(world_goal)
        
        return on_line and making_progress

    def getNextFollowPoint(self, followed_point: Vector_2D, right_follow: bool) -> Vector_2D:
        angle_sign = -1

        away_point = Vector_2D(0, 0)
        max_angle = -math.inf

        for point in self.getFollowedBoundary(followed_point):
            # ensure that the distance from the boundary is small enough,
            # to avoid being closer to the other side of the corridor
            resize = min(1, 0.4 * self.cur_corridor_width)
            radius = min(resize * self.boundary_dist, 0.9999 * point.length())

            # rotate away from the obstacle to avoid colliding with it
            avoidance_angle = getFoVCoverage(point, radius)
            assert avoidance_angle is not None
            rotated = point.rotate(avoidance_angle * angle_sign)

            # find the point that would avoid all other points on the obstacle as well
            angle = angle_sign * followed_point.angle(rotated)
            if max_angle < angle:
                max_angle = angle
                away_point = rotated

        return away_point
