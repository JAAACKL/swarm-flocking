# MEAM 6240, UPenn

import numpy as np
from Node2D import Node2D
from Edge2D import Edge2D
from Graph2D import Graph2D
import time

def generateRandomGraph2D(N):
    """Generate a random 2D graph with N nodes"""
    G = Graph2D()

    # Create nodes with random positions and orientations
    for inode in range(N):
        n = Node2D(inode)
        
        # Random position within the bounds [-5.0, 5.0]
        x = np.random.uniform(-4.0, 4.0)
        y = np.random.uniform(-4.0, 4.0)
        
        # Random orientation
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Fixed constant speed (can be set per node but will remain constant during simulation)
        speed = 1.0
        
        n.setState([x, y, theta, speed])
        
        # Set flocking parameters
        n.w0 = 0.1  # Base angular velocity
        n.collision_threshold = 1.0
        n.neighbor_radius = 2.0  # Increased radius for better visibility of connections
        n.collisionscale = 5.0
        n.velocityscale = 2.0
        n.centeringscale = 2.0
        n.angle_gain = 10.0
        
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
        n = Node2D(start_idx + i)
        
        # Position obstacles at specific locations
        x = np.random.uniform(-4.0, 4.0)
        y = np.random.uniform(-4.0, 4.0)
        
        # Obstacles have zero speed
        n.setState([x, y, 0, 0])
        n.is_obstacle = True
        
        G.addNode(n)
    
    return G

def runSimulation(N=10, with_obstacles=True, steps=None):
    """Run the 2D flocking simulation"""
    # Generate a random graph with N nodes
    G = generateRandomGraph2D(N)
    
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
    runSimulation(30, with_obstacles=True)
    
    # Alternatively, run a fixed number of steps without visualization
    # runSimulation(50, with_obstacles=True, steps=500) 