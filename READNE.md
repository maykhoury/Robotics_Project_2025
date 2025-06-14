 Bug2-Based Drone Navigation Simulator
 
 _This project was originally developed and committed to a private repository under my college GitHub account as part of an academic assignment._
This project implements a **Bug2 algorithm** for autonomous obstacle-avoiding drone navigation using simulated Lidar data within a known and unknown environment. The drone is controlled using Microsoft's [AirSim](https://github.com/microsoft/AirSim) simulator.

## Features

- Connects to and controls a drone in AirSim
- Loads prior knowledge of obstacles from a CSV file
- Implements the **Bug2 path planning algorithm**:
  - First phase uses prior obstacle knowledge
  - Second phase uses real-time Lidar scanning
- Automatically detects and avoids obstacles
- Visualizes obstacle maps and drone path using matplotlib

## How It Works

1. **DroneClient.py** — Manages connection and commands to the AirSim drone.
2. **Bug2Planner.py** — Implements Bug2 algorithm, combining prior map data and Lidar input.
3. **DroneTypes.py** — Defines `Position`, `Orientation`, `Pose`, and `PointCloud` structures.
4. **main.py** — Entry point to initialize the simulation, load prior knowledge, and (optionally) run the full trajectory mission.
