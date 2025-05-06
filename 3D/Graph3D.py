# MEAM 6240, UPenn

from Node3D import Node3D
from Edge3D import Edge3D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

class Graph3D:
    def __init__(self, filename=None):
        """Constructor"""
        self.Nv = 0        # Number of nodes
        self.V = []        # List of nodes
        self.E = []        # List of edges
        
        # For plotting
        self.animatedt = 50  # milliseconds
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim3d([-1.5, 1.5])
        self.ax.set_ylim3d([-1.5, 1.5])
        self.ax.set_zlim3d([-1.5, 1.5])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Flocking Simulation')
        
        # Initialize scatter plot with smaller markers
        self.scat = self.ax.scatter([], [], [], c='b', marker='o', s=10)
        self.scat_obstacles = self.ax.scatter([], [], [], c='r', marker='o', s=100)
        self.quiver = None  # For velocity vectors
        
        # For animation
        self.anim = None
        self.phaseHistory = []
        
        # For auto-rotation
        self.rotation_angle = 0
        self.rotation_speed = 0.1  # degrees per frame

    def __str__(self):
        """Printing"""
        return f"Graph3D: {self.Nv} nodes, {len(self.E)} edges"
    
    def addNode(self, n):
        """Add a node to the graph"""
        self.V.append(n)
        self.Nv += 1
    
    def addEdge(self, i, o, c=0):
        """Add an edge between two nodes"""
        e = Edge3D(i, o, c)
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
        positions = np.array([node.getPosition() for node in self.V if not node.is_obstacle])
        if len(positions) == 0:
            return [], [], []
        return positions[:, 0], positions[:, 1], positions[:, 2]
    
    def gatherObstacleLocations(self):
        """Collect state information from all the obstacles"""
        positions = np.array([node.getPosition() for node in self.V if node.is_obstacle])
        if len(positions) == 0:
            return [], [], []
        return positions[:, 0], positions[:, 1], positions[:, 2]
    
    def gatherVelocities(self):
        """Collect velocity vectors from all nodes"""
        velocities = np.array([node.getVelocityVector() for node in self.V if not node.is_obstacle])
        positions = np.array([node.getPosition() for node in self.V if not node.is_obstacle])
        return positions, velocities
    
    def gatherPhase(self):
        """Collect phase information (orientation angles)"""
        return np.array([node.getOrientation() for node in self.V])
    
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
        x, y, z = self.gatherNodeLocations()
        self.scat._offsets3d = (x, y, z)

        x_obs, y_obs, z_obs = self.gatherObstacleLocations()
        self.scat_obstacles._offsets3d = (x_obs, y_obs, z_obs)
        
        # Update velocity vectors
        positions, velocities = self.gatherVelocities()
        
        # Remove old quiver and create new one
        if self.quiver:
            self.quiver.remove()
        
        # Scale velocity vectors for better visualization (reduced scale)
        scale = 0.1
        if len(positions) > 0:
            self.quiver = self.ax.quiver(
                positions[:, 0], positions[:, 1], positions[:, 2],
                velocities[:, 0], velocities[:, 1], velocities[:, 2],
                length=scale, normalize=True, color='r'
            )
        
        # Update rotation angle
        self.rotation_angle += self.rotation_speed
        self.ax.view_init(elev=20, azim=self.rotation_angle)
        
        # Record phase information
        self.phaseHistory.append(self.gatherPhase())
        
        return self.scat, self.quiver
    
    def plotPhase(self):
        """Plot the phase coherence over time"""
        if not self.phaseHistory:
            print("No phase history to plot")
            return
        
        # Calculate phase coherence for azimuthal angle (theta)
        theta_coherence = []
        for phase_step in self.phaseHistory:
            # Extract theta values
            thetas = phase_step[:, 0]
            
            # Calculate unit vectors on the circle
            x = np.cos(thetas)
            y = np.sin(thetas)
            
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