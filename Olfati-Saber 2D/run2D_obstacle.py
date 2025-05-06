# MEAM 6240, UPenn

import numpy as np
from Node2D import Node2D
from Edge2D import Edge2D
from Graph2D import Graph2D
from Obstacle2D import Obstacle
import time
import matplotlib.pyplot as plt
def generateRandomGraph2D(N, flocking):
    """Generate a random 2D graph with N nodes"""
    G = Graph2D()

    # Create nodes with random positions and velocities
    for inode in range(N):
        n = Node2D(inode)
        n.is_flocking = flocking
        
        # Random position within the bounds [-5.0, 5.0]
        x = np.random.uniform(-200.0, -150.0)
        y = np.random.uniform(-200.0, 200.0)
        
        # Random velocity direction with fixed magnitude
        theta = np.random.uniform(0, 2 * np.pi)
        speed = 0.0  # Initial speed
        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)
        
        n.setState([x, y, vx, vy])
        
        # Add random perturbation to prevent synchronization deadlock
        n.flag = False

        if inode < 1:
            n.is_gamma = True
            n.state[0] = 300
            n.state[1] = 0
            n.state[2] = 0
            n.state[3] = 0
        
        G.addNode(n)
        
    # In the vectorized version, we don't need actual Edge objects for message passing
    # But we can still create them to maintain the graph structure
    for i in range(N):
        for j in range(N):
            if i != j:
                G.addEdge(i, j, 0)
    
    return G

def addObstacles(G):
    """Add static obstacle nodes: outer ring + two smaller Vs."""
    obstacles = [
        # ——— square boundaries ———
        Obstacle('hyperplane', normal=[ 0,  1], point=[ 0,  250]),  # top edge y=+500
        Obstacle('hyperplane', normal=[ 0, -1], point=[ 0, -250]),  # bottom edge y=-500
        # Obstacle('hyperplane', normal=[ 1,  0], point=[ 250,  0]),  # right edge x=+500
        Obstacle('hyperplane', normal=[-1,  0], point=[-250,  0]),  # left edge  x=-500
        Obstacle('sphere', center=[60, 30], radius=40),   # small circle
        Obstacle('sphere', center=[  -60,  160], radius=70),   # medium circle
        Obstacle('sphere', center=[  0, -120], radius=80),  # larger circle
    ]

    # attach to your graph:
    G.obstacles = obstacles

    return G

def runSimulation(N=10, with_obstacles=True, steps=None, flocking=False):
    """Run the 2D flocking simulation"""
    # Generate a random graph with N nodes
    G = generateRandomGraph2D(N, flocking)
    
    # Add obstacles if requested
    if with_obstacles:
        G = addObstacles(G)
    
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

    else:
        # Set up animation for visualization
        G.setupAnimation()
    
    print("========== Simulation Complete ==========")
    return G

if __name__ == '__main__':
    runSimulation(50, with_obstacles=True, flocking=False)
    # Run with 500 nodes and no fixed step count (interactive visualization)
    steps_taken = np.zeros((2,10))
   
    for i in range(10):
        G = runSimulation(50, with_obstacles=True, flocking=False, steps=1000)
        steps_taken[0,i] = G.step_taken_to_reach_goal
        G = runSimulation(50, with_obstacles=True, flocking=True, steps=1000)
        steps_taken[1,i] = G.step_taken_to_reach_goal
        
    # print the mean and std of the steps taken for both flocking and non-flocking
    print(f"Mean steps taken to reach goal for non-flocking: {np.mean(steps_taken[0,:])}")
    print(f"Std steps taken to reach goal for non-flocking: {np.std(steps_taken[0,:])}")
    print(f"Mean steps taken to reach goal for flocking: {np.mean(steps_taken[1,:])}")
    print(f"Std steps taken to reach goal for flocking: {np.std(steps_taken[1,:])}")

    # plot the steps taken for both flocking and non-flocking
    # plot bar chart of the steps taken for both flocking and non-flocking
    # with error bars
    plt.bar(['Non-flocking', 'Flocking'], [np.mean(steps_taken[0,:]), np.mean(steps_taken[1,:])], yerr=[np.std(steps_taken[0,:]), np.std(steps_taken[1,:])])
    plt.show()
