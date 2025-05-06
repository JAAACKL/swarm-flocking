# MEAM 6240, UPenn

from Node2D import Node2D
from Edge2D import Edge2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection

class Graph2D:
    def __init__(self, filename=None):
        """Constructor"""
        self.Nv = 0        # Number of nodes
        self.V = []        # List of nodes
        self.E = []        # List of edges
        
        # For plotting
        self.animatedt = 50  # milliseconds
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([-5.0, 5.0])
        self.ax.set_ylim([-5.0, 5.0])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('2D Flocking Simulation')
        self.ax.set_aspect('equal')
        
        # Initialize scatter plot with smaller markers
        self.scat = self.ax.scatter([], [], c='b', marker='o', s=30)
        self.quiver = None  # For velocity vectors
        self.connections = None  # For visualizing neighbor connections
        
        # For animation
        self.anim = None
        self.phaseHistory = []

    def __str__(self):
        """Printing"""
        return f"Graph2D: {self.Nv} nodes, {len(self.E)} edges"
    
    def addNode(self, n):
        """Add a node to the graph"""
        self.V.append(n)
        self.Nv += 1
    
    def addEdge(self, i, o, c=0):
        """Add an edge between two nodes"""
        e = Edge2D(i, o, c)
        self.E.append(e)
    
    def simulate(self, steps=1):
        """Run simulation for a number of steps"""
        for _ in range(steps):
            # First compute control for all nodes (vectorized)
            for node in self.V:
                node.computeControl(self.V)
            
            # Then update all node states
            for node in self.V:
                node.updateState()
    
    def run_simulation(self, total_steps=100):
        """Run the simulation for a fixed number of steps"""
        for _ in range(total_steps):
            self.simulate(1)
            # Optionally record data
            self.phaseHistory.append(self.gatherPhase())
    
    def gatherNodeLocations(self):
        """Collect state information from all the nodes"""
        positions = np.array([node.getPosition() for node in self.V])
        if len(positions) == 0:
            return [], []
        return positions[:, 0], positions[:, 1]
    
    def gatherVelocities(self):
        """Collect velocity vectors from all nodes"""
        velocities = np.array([node.getVelocityVector() for node in self.V])
        positions = np.array([node.getPosition() for node in self.V])
        return positions, velocities
    
    def gatherPhase(self):
        """Collect phase information (orientation angles)"""
        return np.array([node.getOrientation() for node in self.V])
    
    def gatherNeighborConnections(self):
        """Gather connections between neighboring nodes based on neighbor_radius"""
        connections = []
        
        # For each node, find its neighbors
        for i, node_i in enumerate(self.V):
            if node_i.is_obstacle:
                continue
                
            pos_i = node_i.getPosition()
            neighbor_radius = node_i.neighbor_radius
            
            for j, node_j in enumerate(self.V):
                if i == j or node_j.is_obstacle:
                    continue
                    
                pos_j = node_j.getPosition()
                
                # Calculate distance between nodes
                dist = np.linalg.norm(pos_i - pos_j)
                
                # If within neighbor radius, add connection
                if dist <= neighbor_radius:
                    connections.append([pos_i, pos_j])
        
        return connections
    
    def setupAnimation(self):
        """Initialize the animation"""
        self.anim = animation.FuncAnimation(
            self.fig, self.animate, 
            frames=100, interval=self.animatedt, 
            blit=False)
        plt.show()
    
    def animate(self, i):
        """Animation helper function"""
        self.simulate(1)  # Run one step of simulation
        
        # Update node positions in the plot
        x, y = self.gatherNodeLocations()
        self.scat.set_offsets(np.column_stack([x, y]))
        
        # Update velocity vectors
        positions, velocities = self.gatherVelocities()
        
        # Remove old quiver and create new one
        if self.quiver:
            self.quiver.remove()
        
        # Scale velocity vectors for better visualization
        if len(positions) > 0:
            self.quiver = self.ax.quiver(
                positions[:, 0], positions[:, 1],
                velocities[:, 0], velocities[:, 1],
                scale=50, color='r'
            )
        
        # Update neighbor connections
        if self.connections:
            self.connections.remove()
            
        # Draw connections between neighboring nodes
        neighbor_lines = self.gatherNeighborConnections()
        if neighbor_lines:
            self.connections = LineCollection(
                neighbor_lines, 
                colors='lightgray', 
                linewidths=0.5, 
                alpha=0.3, 
                zorder=0  # Draw behind nodes
            )
            self.ax.add_collection(self.connections)
        
        # Record phase information
        self.phaseHistory.append(self.gatherPhase())
        
        return self.scat, self.quiver
    
    def plotPhase(self):
        """Plot the phase coherence over time"""
        if not self.phaseHistory:
            print("No phase history to plot")
            return
        
        # Calculate phase coherence
        theta_coherence = []
        for phase_step in self.phaseHistory:
            # Calculate unit vectors on the circle
            x = np.cos(phase_step)
            y = np.sin(phase_step)
            
            # Calculate the magnitude of the average vector
            avg_vec = np.array([np.mean(x), np.mean(y)])
            magnitude = np.linalg.norm(avg_vec)
            
            theta_coherence.append(magnitude)
        
        plt.figure(figsize=(10, 6))
        plt.plot(theta_coherence)
        plt.title(f"Phase Coherence with w0 = {self.V[0].w0} and gain = {self.V[0].angle_gain}")
        plt.xlabel("Timestep")
        plt.ylabel("Phase Coherence")
        plt.grid(True)
        plt.show() 