from DroneClient import DroneClient
from Vector import *
from Bug2_algorithm import Bug2
import time
from Part1 import Navigation , Obstacle
x_min, x_max = -1250, -670
y_min, y_max = -1200, -700


# Navigate Start -> Mid flight
class MidWayNavigator:
    start: ()
    goal: ()
    height: float
    start_time: time.time()
    client: DroneClient

    def __init__(self, start, goal, height, client):
        self.start = start
        self.goal = goal
        self.height = height
        self.client = client
        self.start_time = time.time()

    def fly_to_mid_point(self):
        navigator=Navigation(self.client)
        navigator.set_waypoints(self.start,self.goal,self.height)
        graph,path=navigator.start_navigation()
        print(path)
        start=Vector_2D(self.start[0],self.start[1])
        goal=Vector_2D(self.goal[0],self.goal[1])
        bug = Bug2(self.client, self.height,start)#,goal)
        i=1
        if x_min <= self.goal[0] <= x_max and y_min <= self.goal[1] <= y_max :
            for p in range(0,len(path)-2):
                print(f"Moving to {i}")
                start=Vector_2D(path[p][0],path[p][1])
                goal=Vector_2D(path[p+1][0],path[p+1][1])
                bug.findPath(goal,start)
                i+=1
        else:
            print("Goal is out of bounds")
            bug.findPath(Vector_2D(self.goal[0], self.goal[1]))

        mid_time = time.time()
        mid_arrival_time = mid_time - self.start_time
        print(f"Time taken from START to MID: {mid_arrival_time:.2f} seconds")
        # navigator.visualize_visibility_graph(graph,path,bug.pos)
        return graph,path,bug.pos,navigator.polygons,mid_arrival_time
        

# Navigate Mid -> End flight
class GoalNavigator:
    start: Vector_2D
    goal: Vector_2D
    height: float
    client: DroneClient
    start_time: time.time()

    def __init__(self, start, goal, height, client):
        self.start = start
        self.goal = goal
        self.height = height
        self.client = client
        self.start_time = time.time()

    def fly_to_goal_point(self):
        time.sleep(5)
        bug = Bug2(self.client, self.height, self.start)
        bug.findPath(self.goal,self.start)

        end_time = time.time()
        flight_time = end_time - self.start_time
        print(f"Time from MID to GOAL: {flight_time:.2f} seconds")

        return bug.pos, bug.lidar_data, flight_time
