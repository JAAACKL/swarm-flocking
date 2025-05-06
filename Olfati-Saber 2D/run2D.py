# MEAM 6240, UPenn

import numpy as np
from Node2D import Node2D
from Edge2D import Edge2D
from Graph2D import Graph2D
from Obstacle2D import Obstacle
import time
import matplotlib.pyplot as plt
def generateRandomGraph2D(N, is_flocking):
    """Generate a random 2D graph with N nodes"""
    G = Graph2D()

    for inode in range(0, N):
        n = Node2D(inode)
        n.is_flocking = is_flocking
        # Random position within the bounds [-5.0, 5.0]
        x = np.random.uniform(-200.0, 200.0)
        y = np.random.uniform(-200.0, 200.0)
        
        
        # Random velocity direction with fixed magnitude
        theta = np.random.uniform(0, 2 * np.pi)
        speed = 10.0  # Initial speed
        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)

        n.has_obstruction = True
        
        n.setState([x, y, vx, vy])
        
        # Add random perturbation to prevent synchronization deadlock
        n.flag = False
        
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
        Obstacle('hyperplane', normal=[ 1,  0], point=[ 250,  0]),  # right edge x=+500
        Obstacle('hyperplane', normal=[-1,  0], point=[-250,  0]),  # left edge  x=-500
    ]

    # attach to your graph:
    G.obstacles = obstacles

    return G

def runSimulation(N=10, with_obstacles=True, steps=None, is_flocking=True):
    """Run the 2D flocking simulation"""
    # Generate a random graph with N nodes
    G = generateRandomGraph2D(N, is_flocking)
    
    # Add obstacles if requested
    if with_obstacles:
        G = addObstacles(G)
    
    # Food functionality:
    # - Food is implemented as a special type of non-moving agent (Node2D with is_food=True)
    # - 5 food items are spawned randomly in the environment
    # - Food is visualized as red stars
    # - Agents are attracted to food within their sensing range
    # - Each food can be eaten up to 10 times before disappearing
    # - After an agent eats a food, it won't be attracted to that food for 1000 simulation updates
    # - The simulation tracks and plots the amount of food remaining (total bites left)
    
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
        # Plot both phase coherence, alive prey count, and food remaining

    else:
        # Set up animation for visualization
        G.setupAnimation()
    
    print("========== Simulation Complete ==========")
    return G

if __name__ == '__main__':

    num_runs = 10
    steps = 400
    # Initialize arrays to collect metrics
    metrics_C = np.zeros((num_runs, steps))
    metrics_R = np.zeros((num_runs, steps))
    metrics_E = np.zeros((num_runs, steps))
    metrics_K = np.zeros((num_runs, steps))

    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}")
        G = runSimulation(50, with_obstacles=False, steps=steps)
        metrics_C[run] = G.C_history
        metrics_R[run] = G.R_history
        metrics_E[run] = G.E_history
        metrics_K[run] = G.K_history

        # Compute means and stds
    mean_C = metrics_C.mean(axis=0)
    std_C  = metrics_C.std(axis=0)
    mean_R = metrics_R.mean(axis=0)
    std_R  = metrics_R.std(axis=0)
    mean_E = metrics_E.mean(axis=0)
    std_E  = metrics_E.std(axis=0)
    mean_K = metrics_K.mean(axis=0)
    std_K  = metrics_K.std(axis=0)
    
    t = np.arange(steps)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1) Relative Connectivity
    ax = axes[0,0]
    ax.plot(t, mean_C, label='Mean C(t)')
    ax.fill_between(t, mean_C - std_C, mean_C + std_C, alpha=0.3)
    ax.set_title('Relative Connectivity')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('C(t)')
    ax.grid(True)

    # 2) Cohesion Radius
    ax = axes[0,1]
    ax.plot(t, mean_R, label='Mean R(t)')
    ax.fill_between(t, mean_R - std_R, mean_R + std_R, alpha=0.3)
    ax.set_title('Cohesion Radius')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('R(t)')
    ax.grid(True)

    # 3) Normalized Deviation Energy
    ax = axes[1,0]
    ax.plot(t, mean_E, label='Mean Ē(q)')
    ax.fill_between(t, mean_E - std_E, mean_E + std_E, alpha=0.3)
    ax.set_title('Deviation Energy')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Ē(q)')
    ax.grid(True)

    # 4) Normalized Velocity Mismatch
    ax = axes[1,1]
    ax.plot(t, mean_K, label='Mean K̄(v)')
    ax.fill_between(t, mean_K - std_K, mean_K + std_K, alpha=0.3)
    ax.set_title('Velocity Mismatch')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('K̄(v)')
    ax.grid(True)

    plt.tight_layout()
    plt.show()