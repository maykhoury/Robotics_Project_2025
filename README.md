# Bug2-Based Drone Navigation Simulator

> _This project was originally developed and committed to a private repository under my college GitHub account as part of an academic assignment. It has since been adapted and published here for demonstration and continued development._

This project implements the **Bug2 algorithm** for autonomous obstacle-avoiding drone navigation using simulated Lidar data in environments with both prior knowledge and unknown elements. The drone is controlled via Microsoft's [AirSim](https://github.com/microsoft/AirSim) simulator.

---

## Features

- Connects to and controls a drone in AirSim
- Loads prior obstacle knowledge from a CSV file
- Implements the **Bug2 path planning algorithm**:
  - Phase 1: Uses prior map data to avoid known obstacles
  - Phase 2: Uses real-time Lidar data for dynamic avoidance
- Automatically detects and avoids obstacles
- Visualizes the prior map and planned trajectory using matplotlib

---

## How It Works

- **`DroneClient.py`** — Wraps connection and control APIs for the AirSim drone
- **`Bug2Planner.py`** — Core logic implementing the Bug2 navigation algorithm
- **`DroneTypes.py`** — Data structures: `Position`, `Pose`, `Orientation`, `PointCloud`
- **`main.py`** — Orchestrates drone connection, planning, and visualization
