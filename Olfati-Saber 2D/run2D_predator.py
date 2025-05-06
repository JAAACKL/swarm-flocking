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

    # gamma nodes
    # n = Node2D(0)
    # n.is_gamma = False
    # n.setState([0, 0, 0, 0])
    # G.addNode(n)

    

    # Create nodes with random positions and velocities
    for inode in range(0, N):
        n = Node2D(inode)
        n.is_flocking = is_flocking
        # Random position within the bounds [-5.0, 5.0]
        x = np.random.uniform(-100.0, 100.0)
        y = np.random.uniform(-100.0, 100.0)
        

        while abs(x) < 50.0 and abs(y) < 50.0:
            x = np.random.uniform(-100.0, 100.0)
            y = np.random.uniform(-100.0, 100.0)
        
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

    # create food nodes
    for i in range(N,N+3):
        n = Node2D(i)
        n.is_predator = True
        n.max_vel = 300
        n.max_accel = 1500
        n.max_rotation_rate = 2.0 * np.pi
        n.predator_strategy = 1
        x = np.random.uniform(-50.0, 50.0)
        y = np.random.uniform(-50.0, 50.0)
        n.setState([x, y, 0, 0])
        G.addNode(n)
        
    # In the vectorized version, we don't need actual Edge objects for message passing
    # But we can still create them to maintain the graph structure
    for i in range(N+3):
        for j in range(N+3):
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

    # runSimulation(50, with_obstacles=True)

    num_runs = 5
    max_steps = 1000
    N = 50

    # Arrays to store alive prey counts per timestep for each trial
    alive_nf = np.zeros((num_runs, max_steps))
    alive_f  = np.zeros((num_runs, max_steps))

    for i in range(num_runs):
        print(f"Trial {i+1}/{num_runs} — No Flocking")
        G_nf = runSimulation(N, with_obstacles=True, steps=max_steps, is_flocking=False)
        # Truncate or pad counts to max_steps
        counts_nf = G_nf.alive_prey_count
        alive_nf[i, :len(counts_nf)] = counts_nf[:max_steps]

        print(f"Trial {i+1}/{num_runs} — Flocking")
        G_f = runSimulation(N, with_obstacles=True, steps=max_steps, is_flocking=True)
        counts_f = G_f.alive_prey_count
        alive_f[i, :len(counts_f)] = counts_f[:max_steps]

    # Compute mean and std over trials
    mean_nf = alive_nf.mean(axis=0)
    std_nf  = alive_nf.std(axis=0)
    mean_f  = alive_f.mean(axis=0)
    std_f   = alive_f.std(axis=0)

    t = np.arange(1, max_steps+1)

    # Plot time series and mean ± std for No Flocking
    fig, ax = plt.subplots(figsize=(10, 4))
    # for trial in range(num_runs):
    #     ax.plot(t, alive_nf[trial], color='gray', alpha=0.3)
    ax.plot(t, mean_nf, color='blue', label='Mean No Flocking')
    ax.fill_between(t, mean_nf - std_nf, mean_nf + std_nf, color='blue', alpha=0.2)

    # Plot time series and mean ± std for Flocking
    # for trial in range(num_runs):
    #     ax.plot(t, alive_f[trial], color='brown', alpha=0.3)
    ax.plot(t, mean_f, color='green', label='Mean Flocking')
    ax.fill_between(t, mean_f - std_f, mean_f + std_f, color='green', alpha=0.2)
    ax.set_title('Alive Prey Over Time')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Number of Alive Prey')
    ax.legend()
    ax.grid(True)
    plt.show()