# MEAM 6240, UPenn

import numpy as np
from Node3D import Node3D
from Edge3D import Edge3D
from Graph3D import Graph3D
import time

def generateRandomGraph3D(N):
    """Generate a random 3D graph with N nodes"""
    G = Graph3D()

    # Create nodes with random positions and orientations
    for inode in range(N):
        n = Node3D(inode)
        
        # Random position within the bounds [-1.5, 1.5]
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-1.0, 1.0)
        z = np.random.uniform(-1.0, 1.0)
        
        # Random orientation (azimuth and elevation angles)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(-np.pi/2, np.pi/2)
        
        # Initial speed (default is 1.0, but can be randomized)
        speed = 1.0
        
        n.setState([x, y, z, theta, phi, speed])
        
        # Set flocking parameters
        n.w0 = 0.1  # Base angular velocity
        n.collision_threshold = 0.5
        n.collisionscale = 1.5
        n.velocityscale = 1.0
        n.centeringscale = 0.0
        n.angle_gain = 1.0
        n.speed_gain = 0.5
        
        # Set speed limits
        n.min_speed = 0.2
        n.max_speed = 2.0
        
        # Add random perturbation to prevent synchronization deadlock
        n.flag = True
        
        G.addNode(n)
        
    # In the vectorized version, we don't need actual Edge objects for message passing
    # But we can still create them to maintain the graph structure
    for i in range(N):
        for j in range(N):
            if i != j:
                G.addEdge(i, j, 0)
    
    return G

def addObstacles(G, num_obstacles=2):
    """Add static obstacle nodes to the graph"""
    start_idx = G.Nv
    
    for i in range(num_obstacles):
        n = Node3D(start_idx + i)
        
        # Position obstacles at specific locations
        x = np.random.uniform(-0.8, 0.8)
        y = np.random.uniform(-0.8, 0.8)
        z = np.random.uniform(-0.8, 0.8)
        
        n.setState([x, y, z, 0, 0])
        n.is_obstacle = True
        
        G.addNode(n)
    
    return G

def runSimulation(N=10, with_obstacles=True, steps=None):
    """Run the 3D flocking simulation"""
    # Generate a random graph with N nodes
    G = generateRandomGraph3D(N)
    
    # Add obstacles if requested
    if with_obstacles:
        G = addObstacles(G, num_obstacles=3)
    
    # Print out the graph descriptor
    print(G)
    for inode in G.V:
        print(inode)
    
    print("========== Starting Simulation ==========")
    print("Close the figure to stop the simulation")
    
    # Run simulation without visualization if steps are specified
    if steps is not None:
        start_time = time.time()
        G.run_simulation(steps)
        end_time = time.time()
        print(f"Simulation of {steps} steps completed in {end_time - start_time:.4f} seconds")
        G.plotPhase()
    else:
        # Set up animation for visualization
        G.setupAnimation()
    
    print("========== Simulation Complete ==========")
    return G

if __name__ == '__main__':
    # Run with 20 nodes and no fixed step count (interactive visualization)
    runSimulation(50, with_obstacles=True)
    
    # Alternatively, run a fixed number of steps without visualization
    # runSimulation(20, with_obstacles=True, steps=500) 