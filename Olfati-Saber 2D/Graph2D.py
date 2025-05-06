from Node2D import Node2D
from Edge2D import Edge2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
class Graph2D:
    def __init__(self, filename=None):
        """Constructor"""
        self.Nv = 0        # Number of nodes
        self.V = []        # List of nodes
        self.E = []        # List of edges
        self.obstacles = [] # List of obstacles
        self.obs_patches = []     # will hold the matplotlib artists
        
        # Food-related properties
        self.food_count = 5  # Number of food items to maintain
        self.food_remaining_history = []  # Track food left over time
        
        # Track alive prey count over time
        self.alive_prey_count = []
        self.sum_avg_speed = 0.0 # Sum of average speeds per step
        self.steps_for_speed = 0 # Number of steps counted for speed

        # For plotting
        self.animatedt = 10  # milliseconds
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim([-300.0, 300.0])
        self.ax.set_ylim([-300.0, 300.0])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('2D Flocking Simulation')
        self.ax.set_aspect('equal')
        
        # Initialize scatter plot with smaller markers
        self.scat = self.ax.scatter([], [], c='b', marker='o', s=30)
        self.predator_scat = self.ax.scatter([], [], c='orange', marker='o', s=80)
        self.gamma_scat = self.ax.scatter([], [], c='green', marker='o', s=30)
        # Scatter for projected predator positions
        self.predator_proj_scat = None
        self.quiver = None  # For velocity vectors
        self.connections = None  # For visualizing neighbor connections
        self.bold_connections = None  # For visualizing lattice connections
        # For beta-node plotting
        self.beta_scat = None
        self.beta_circles = []
        
        # For food visualization
        self.food_scat = None
        
        # For animation
        self.anim = None
        self.C_history = []   # relative connectivity
        self.R_history = []   # cohesion radius
        self.E_history = []   # normalized deviation energy
        self.K_history = []   # normalized velocity mismatch

        self.step_taken_to_reach_goal = -1

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
            # Check for prey that are eaten by predators
            predators = [node for node in self.V if getattr(node, "is_predator", False)]
            prey = [node for node in self.V if not getattr(node, "is_predator", False) 
                   and not getattr(node, "is_beta", False) 
                   and not getattr(node, "is_gamma", False)
                   and not getattr(node, "is_eaten", False)]
            
            # Mark prey as eaten if they are within 1 unit of any predator
            for prey_node in prey:
                prey_pos = prey_node.getPosition()
                for pred_node in predators:
                    pred_pos = pred_node.getPosition()
                    distance = np.linalg.norm(prey_pos - pred_pos)
                    if distance <= 10.0:  # 1 unit distance threshold for being eaten
                        prey_node.is_eaten = True
                        break
            
            # Count alive prey and store in history
            alive_count = sum(1 for node in self.V if not getattr(node, "is_predator", False) 
                           and not getattr(node, "is_beta", False) 
                           and not getattr(node, "is_gamma", False)
                           and not getattr(node, "is_eaten", False))
            self.alive_prey_count.append(alive_count)
            
            # Check for food consumption
            self.check_food_consumption()
            
            # Track food remaining
            self.track_food_remaining()
            
            # First compute control for all nodes (vectorized)
            for node in self.V:
                if not getattr(node, "is_eaten", False):  # Skip eaten prey
                    node.computeControl(self.V, self.obstacles)
            
            # Then update all node states
            for node in self.V:
                if not getattr(node, "is_eaten", False):  # Skip eaten prey
                    node.updateState()

            self.compute_metrics()
    
    def plotAlivePrey(self):
        """Plot the number of alive prey over time"""
        if not self.alive_prey_count:
            print("No prey count history to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.alive_prey_count)
        plt.title("Number of Alive Prey Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Number of Alive Prey")
        plt.grid(True)
        plt.show()
    
    def run_simulation(self, total_steps=100):
        """Run the simulation for a fixed number of steps"""
        for i in range(total_steps):
            self.simulate(1)
            # Optionally record data
            print(f"Step {i} of {total_steps}")

            if len(self.gatherGammaLocations()[0]) != 0 and self.hasReachedGoal():
                self.step_taken_to_reach_goal = i
                break
        
    
    def gatherNodeLocations(self):
        """Collect state information from all the nodes"""
        positions = np.array([
            node.getPosition() 
            for node in self.V 
            if not getattr(node, "is_gamma", False) and not getattr(node, "is_food", False) and not getattr(node, "is_eaten", False) and not getattr(node, "is_beta", False) and not getattr(node, "is_predator", False)
        ])
        if len(positions) == 0:
            return [], []
        return positions[:, 0], positions[:, 1]
    
    def gatherPredatorLocations(self):
        """Collect state information from all the nodes"""
        positions = np.array([
            node.getPosition() 
            for node in self.V 
            if getattr(node, "is_predator", False)
        ])
        if len(positions) == 0:
            return [], []
        return positions[:, 0], positions[:, 1]
    
    def gatherGammaLocations(self):
        """Collect state information from all the nodes"""
        positions = np.array([
            node.getPosition() 
            for node in self.V 
            if getattr(node, "is_gamma", False)
        ])
        if len(positions) == 0:
            return [], []
        return positions[:, 0], positions[:, 1]

    def gatherVelocities(self):
        """Collect velocity vectors from all nodes"""
        # only include non‑gamma and non-eaten agents
        filtered = [node for node in self.V 
                    if not getattr(node, "is_gamma", False) 
                    and not getattr(node, "is_food", False) 
                    and not getattr(node, "is_eaten", False) 
                    and not getattr(node, "is_beta", False) 
                    and not getattr(node, "is_predator", False)]
        positions  = np.array([n.getPosition()       for n in filtered])
        velocities = np.array([n.getVelocityVector() for n in filtered])
        return positions, velocities
    
    def gatherPhase(self):
        """Collect phase information (orientation angles)"""
        return np.array([node.getOrientation() for node in self.V if not getattr(node, "is_eaten", False)])
    
    def gatherNeighborConnections(self):
        """Gather connections between neighboring nodes based on neighbor_radius"""
        connections = []
        
        # For each node, find its neighbors     
        for node_i in self.V:
            # skip gamma agents and eaten prey
            if getattr(node_i, "is_gamma", False) or getattr(node_i, "is_eaten", False):
                continue
            if getattr(node_i, "is_beta", False):
                continue
            pos_i = node_i.getPosition()
            neighbor_radius = node_i.r
             
            for node_j in self.V:
                # skip self, gamma agents, and eaten prey
                if node_j is node_i or getattr(node_j, "is_gamma", False) or getattr(node_j, "is_eaten", False):
                    continue
                if node_j is node_i or getattr(node_j, "is_beta", False):
                    continue
                pos_j = node_j.getPosition()
                 
                if node_j in node_i.neighbors:
                    connections.append([pos_i, pos_j])
        
        return connections
    
    def gatherLatticeConnections(self):
        """Gather connections between neighboring nodes based on neighbor_radius"""
        connections = []
        
        # For each node, find its neighbors     
        for node_i in self.V:
            # skip gamma agents and eaten prey
            if getattr(node_i, "is_gamma", False) or getattr(node_i, "is_eaten", False):
                continue
            if getattr(node_i, "is_beta", False):
                continue
            pos_i = node_i.getPosition()
            neighbor_radius = node_i.r
             
            for node_j in self.V:
                # skip self, gamma agents, and eaten prey
                if node_j is node_i or getattr(node_j, "is_gamma", False) or getattr(node_j, "is_eaten", False):
                    continue
                if node_j is node_i or getattr(node_j, "is_beta", False):
                    continue
                pos_j = node_j.getPosition()
                 
                if abs(np.linalg.norm(pos_i - pos_j) - node_i.d) <= 1:
                    connections.append([pos_i, pos_j])
        
        return connections
    
    def drawObstacles(self):
        """Draw each obstacle exactly once as a line or circle."""
        for obs in self.obstacles:
            if obs.kind == 'hyperplane':
                # horizontal line?
                # normal = [0,1] or [0,-1]
                if abs(obs.a[0]) < 1e-6:
                    y0 = obs.y[1]
                    ln = self.ax.axhline(y0,
                                         color='black',
                                         linestyle='--',
                                         linewidth=1)
                else:
                    x0 = obs.y[0]
                    ln = self.ax.axvline(x0,
                                         color='black',
                                         linestyle='--',
                                         linewidth=1)
                self.obs_patches.append(ln)

            elif obs.kind == 'sphere':
                circ = Circle(xy=obs.y,
                              radius=obs.R,
                              edgecolor='green',
                              facecolor='none',
                              linestyle=':',
                              linewidth=1.5)
                self.ax.add_patch(circ)
                self.obs_patches.append(circ)
    
    def setupAnimation(self):
        self.drawObstacles()
        """Initialize the animation"""
        self.anim = animation.FuncAnimation(
            self.fig, self.animate, 
            frames=100, interval=self.animatedt, 
            blit=False)
        plt.show()
    
    def animate(self, i):
        """Animation step"""
        self.simulate(1)
        if len(self.gatherGammaLocations()[0]) != 0 and self.hasReachedGoal():
            self.anim.event_source.stop()

        # — update α‑agents —
        x, y = self.gatherNodeLocations()
        self.scat.set_offsets(np.column_stack([x, y]))

        pred_x, pred_y = self.gatherPredatorLocations()
        self.predator_scat.set_offsets(np.column_stack([pred_x, pred_y]))

        gamma_x, gamma_y = self.gatherGammaLocations()
        self.gamma_scat.set_offsets(np.column_stack([gamma_x, gamma_y]))
        
        # — update food locations —
        food_x, food_y = self.gatherFoodLocations()
        if self.food_scat is None:
            self.food_scat = self.ax.scatter(
                food_x, food_y, 
                c='red', marker='*', s=120
            )
        else:
            self.food_scat.set_offsets(np.column_stack([food_x, food_y]))

        # — update projected predator locations (for debug) —
        if self.predator_proj_scat:
            self.predator_proj_scat.remove()
        # collect predator nodes
        pred_nodes = [n for n in self.V if getattr(n, "is_predator", False)]
        proj_positions = []
        for n in pred_nodes:
            pos = n.getPosition()
            vel = n.getVelocity()
            dt_proj = n.nominaldt * n.projection_horizon  # projection time horizon
            proj = pos + vel * dt_proj
            proj_positions.append(proj)
        if proj_positions:
            arr = np.array(proj_positions)
            self.predator_proj_scat = self.ax.scatter(
                arr[:,0], arr[:,1],
                c='magenta', marker='x', s=80, label='pred_proj'
            )

        # — update velocity quiver —
        if self.quiver:
            self.quiver.remove()
        pos, vel = self.gatherVelocities()
        if len(pos)>0:
            self.quiver = self.ax.quiver(
                pos[:,0], pos[:,1],
                vel[:,0], vel[:,1],
                scale=5000, color='r'
            )

        # — update β‑nodes and their radii —
        if self.beta_scat:
            self.beta_scat.remove()
        for c in self.beta_circles:
            c.remove()
        self.beta_circles.clear()

        beta_nodes = [n for n in self.V if getattr(n, 'is_beta', False)]
        if beta_nodes:
            bp = np.array([n.getPosition() for n in beta_nodes])
            self.beta_scat = self.ax.scatter(
                bp[:,0], bp[:,1],
                facecolors='none',
                edgecolors='red',
                linewidths=1.5,
                s=100
            )
            for n in beta_nodes:
                radius = getattr(n, 'd_beta', None)
                if radius is not None:
                    circ = Circle(
                        xy=n.getPosition(),
                        radius=radius,
                        edgecolor='red',
                        facecolor='none',
                        linestyle='--',
                        linewidth=1
                    )
                    self.ax.add_patch(circ)
                    self.beta_circles.append(circ)

        # — update neighbor lines —
        if self.connections:
            self.connections.remove()

        lines = self.gatherNeighborConnections()
        if lines:
            self.connections = LineCollection(
                lines,
                colors='lightgray',
                linewidths=0.5,
                alpha=0.3,
                zorder=0
            )
            self.ax.add_collection(self.connections)

        if self.bold_connections and self.bold_connections.axes:
            self.bold_connections.remove()

        bold_lines = self.gatherLatticeConnections()
        if bold_lines:
            self.bold_connections = LineCollection(
                bold_lines,
                colors='darkgray',
                linewidths=0.8,
                alpha=0.8,
                zorder=0
            )
            self.ax.add_collection(self.bold_connections)

        # record phase
        return (self.scat, self.quiver)

    def track_food_remaining(self):
        """Track the amount of remaining food"""
        food_remaining = 0
        
        # Count remaining bites for each food
        for node in self.V:
            if getattr(node, "is_food", False) and not getattr(node, "is_consumed", False):
                # Add remaining consumption opportunities
                food_remaining += node.max_consumption - node.consumption_count
        
        self.food_remaining_history.append(food_remaining)
    
    def check_food_consumption(self):
        """Check if alpha agents have consumed any food"""
        # Get all alpha nodes
        alpha_nodes = [node for node in self.V 
                      if not getattr(node, "is_predator", False) 
                      and not getattr(node, "is_beta", False) 
                      and not getattr(node, "is_gamma", False)
                      and not getattr(node, "is_food", False)]
        
        # Get all food nodes
        food_nodes = [node for node in self.V if getattr(node, "is_food", False) and not getattr(node, "is_consumed", False)]
        
        # For each food item
        for food in food_nodes:
            # Skip if already fully consumed
            if getattr(food, "is_consumed", False):
                continue
                
            # Check which alpha nodes are close to this food
            for node in alpha_nodes:
            
                # Calculate distance to food
                dist = np.linalg.norm(node.getPosition() - food.getPosition())
                
                # If close enough, mark as consumed by this agent
                if dist <= 10.0 and node.food_cooldowns <= 0:  # 10 units threshold for consumption
                    # Add to food's eaten by set
                    
                    # Increment consumption count
                    food.consumption_count += 1
                    
                    # Add this food to agent's cooldown
                    node.food_cooldowns = node.cooldown_period
                    
                    # Break after one agent consumes food in this update
                    # (to avoid multiple agents consuming simultaneously)
                    break
            
            # Check if food has been consumed maximum number of times
            if food.consumption_count >= food.max_consumption:
                food.is_consumed = True

    def getPreyAlive(self):
        """Get the amount of food remaining"""
        return self.alive_prey_count
    
    def getFoodRemaining(self):
        """Get the amount of food remaining"""
        return self.food_remaining_history

    def getFinalAverageSpeed(self):
        """Calculate the final average speed over the simulation run."""
        if self.steps_for_speed == 0:
            return 0.0
        return self.sum_avg_speed / self.steps_for_speed

    def plotFoodRemaining(self):
        """Plot the amount of food remaining over time"""
        if not self.food_remaining_history:
            print("No food history to plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.food_remaining_history)
        plt.title("Food Remaining Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Number of Food Bites Remaining")
        plt.grid(True)
        plt.show()

    def gatherFoodLocations(self):
        """Collect state information from all food nodes"""
        positions = np.array([
            node.getPosition() 
            for node in self.V 
            if getattr(node, "is_food", False) and not getattr(node, "is_consumed", False)
        ])
        if len(positions) == 0:
            return [], []
        return positions[:, 0], positions[:, 1]
    
    def hasReachedGoal(self):
        """Check if all prey have reached the goal"""
        x, _ = self.gatherNodeLocations()
        return np.all(x > 100) 
    
    def compute_metrics(self):
        # collect only the "α-agents" (exclude beta/gamma/predator/food/eaten)
        agents = [
            n for n in self.V
            if not (n.is_beta or n.is_gamma or n.is_predator or n.is_food or n.is_eaten)
        ]
        n = len(agents)
        if n < 2:
            # degenerate
            self.C_history.append(0.0)
            self.R_history.append(0.0)
            self.E_history.append(0.0)
            self.K_history.append(0.0)
            self.E_history.append(0.0)
            self.K_history.append(0.0)
            return

        # positions & velocities
        pos = np.vstack([n.getPosition() for n in agents])
        vel = np.vstack([n.getVelocity() for n in agents])

        # Calculate average speed
        speeds = np.linalg.norm(vel, axis=1)
        avg_speed = np.mean(speeds) if n > 0 else 0.0
        self.sum_avg_speed += avg_speed
        self.steps_for_speed += 1

        # 1) build adjacency based on sensing radius r
        #    (you can also incorporate your can_see test if you like)
        dists = np.linalg.norm(pos[:,None,:] - pos[None,:,:], axis=2)
        A = (dists < agents[0].d + 1).astype(float)
        np.fill_diagonal(A, 0.0)

        # 1a) Relative connectivity C(t)
        D = np.diag(A.sum(axis=1))
        L = D - A
        rankL = np.linalg.matrix_rank(L)
        C = rankL / (n - 1)
        self.C_history.append(C)

        # 2) Cohesion radius R(t)
        centroid = pos.mean(axis=0)
        R = np.max(np.linalg.norm(pos - centroid, axis=1))
        self.R_history.append(R)

        # 3) Normalized deviation energy Ē(q)
        #    E = ½ ∑_{i,j} A_{ij} ‖q_i – q_j‖²
        Z = dists - agents[0].d               # shape (n,n)
        # only keep those pairs that are actually "neighbors"
        E_raw = np.sum(A * (Z**2))            # ∑_{i,j∈N_i} ψ(z_ij)
        num_edges = A.sum()                   # |E(q)| as directed-edge count
        E_norm = E_raw / (num_edges + 1)      # 1/(|E|+1) ∑ ψ
        self.E_history.append(E_norm)

        # 4) Normalized velocity mismatch K̄(v)
        vbar = vel.mean(axis=0)
        K = 0.5 * np.sum(np.linalg.norm(vel - vbar, axis=1)**2)
        K_bar = K / n
        self.K_history.append(K_bar)