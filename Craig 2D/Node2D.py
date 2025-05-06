import numpy as np

class Node2D:
    def __init__(self, uid):
        # Basic information
        self.uid = uid    # node UID (an integer)
        
        # State variables: [x, y, theta, speed] where theta is the heading angle
        self.state = np.zeros(4)
        self.state[3] = 0.5  # Default speed = 1.0 (constant, not adjustable)
        
        # Time step parameters
        self.nominaldt = 0.01
        
        # Flocking parameters
        self.collision_threshold = 0.1
        self.neighbor_radius = 2.0   # Radius for considering neighbors for alignment and cohesion
        self.collisionscale = 1.0
        self.velocityscale = 1.0
        self.centeringscale = 1.0
        self.angle_gain = 1.0
        
        # Angular velocity and control inputs
        self.w0 = 0.0
        self.u_theta = 0.0
        # Speed is not adjustable, so no u_speed control input
        
        # Flags
        self.is_obstacle = False
        self.flag = False

    def __str__(self):
        return f"Node {self.uid} at position ({self.state[0]:.2f}, {self.state[1]:.2f}), speed: {self.state[3]:.2f}"

    def setState(self, s):
        """Update the state of the node"""
        # If the provided state doesn't include speed, add the default speed
        if len(s) == 3:
            s = list(s) + [1.0]  # Default speed = 1.0
        self.state = np.array(s, dtype=float)
    
    def getPosition(self):
        """Return the position as a numpy array"""
        return self.state[:2]
    
    def getOrientation(self):
        """Return the orientation angle"""
        return self.state[2]
    
    def getSpeed(self):
        """Return the current speed"""
        return self.state[3]
    
    def getVelocityVector(self):
        """Return the velocity vector based on orientation and speed"""
        theta = self.state[2]
        speed = self.state[3]
        return speed * np.array([
            np.cos(theta),
            np.sin(theta)
        ])
    
    def getUnitDirectionVector(self):
        """Return the unit direction vector based on orientation"""
        theta = self.state[2]
        return np.array([
            np.cos(theta),
            np.sin(theta)
        ])
    
    def computeControl(self, all_nodes):
        """
        Compute control inputs based on all nodes in the system.
        Uses vectorized operations for efficiency.
        Speed is not adjustable, so only orientation control is computed.
        """
        if self.is_obstacle:
            return

        # Get current node's position and velocity direction (unit vector)
        pos_i = self.getPosition()
        dir_i = self.getUnitDirectionVector()
        
        # Collect positions and velocities of all other nodes
        other_nodes = [node for node in all_nodes if node.uid != self.uid]
        
        if not other_nodes:
            self.u_theta = 0
            return
        
        # Vectorized operations for all neighbors
        positions = np.array([node.getPosition() for node in other_nodes])
        velocities = np.array([node.getVelocityVector() for node in other_nodes])
        is_robot = np.array([not node.is_obstacle for node in other_nodes])
        
        # Calculate displacement vectors from current node to all others
        displacement_vectors = positions - pos_i
        
        # Calculate distances
        distances = np.linalg.norm(displacement_vectors, axis=1)
        
        # Initialize control components
        avoidance = np.zeros(2)
        match = np.zeros(2)
        center = np.zeros(2)
        
        # 1. Collision Avoidance (vectorized) - for very close nodes
        close_indices = distances < self.collision_threshold
        if np.any(close_indices):
            close_displacements = displacement_vectors[close_indices]
            close_distances = distances[close_indices].reshape(-1, 1)
            # Avoid division by zero
            close_distances = np.maximum(close_distances, 1e-6)
            # Stronger repulsion for closer objects
            repulsion_vectors = -close_displacements / (close_distances**2)
            avoidance = np.sum(repulsion_vectors, axis=0)
        
        # 2. Velocity Matching (vectorized) - only with neighbors within radius
        neighbor_indices = np.logical_and(is_robot, distances < self.neighbor_radius)
        if np.any(neighbor_indices):
            neighbor_velocities = velocities[neighbor_indices]
            
            # Align with average direction of neighbors
            avg_velocity = np.mean(neighbor_velocities, axis=0)
            if np.linalg.norm(avg_velocity) > 1e-6:  # Ensure non-zero velocity
                match = avg_velocity / np.linalg.norm(avg_velocity) - dir_i
        
        # 3. Flock Centering (vectorized) - only with neighbors within radius
        if np.any(neighbor_indices):
            neighbor_positions = positions[neighbor_indices]
            flock_center = np.mean(neighbor_positions, axis=0)
            center_vector = flock_center - pos_i
            
            # Normalize center vector if not zero
            if np.linalg.norm(center_vector) > 1e-6:
                center = center_vector / np.linalg.norm(center_vector)
        
        # Combine influences for direction with adjusted weights
        V = (self.collisionscale * avoidance + 
             self.velocityscale * match + 
             self.centeringscale * center)
        
        # If V is near zero, maintain current direction with slight random movement
        if np.linalg.norm(V) < 1e-6:
            self.u_theta = 0.01 * np.random.randn()  # Small random steering
            return
        
        # Normalize V to get direction only
        V_normalized = V / np.linalg.norm(V)
        
        # Convert desired velocity vector to desired orientation angle
        desired_theta = np.arctan2(V_normalized[1], V_normalized[0])
        
        # Calculate angle differences
        theta_diff = desired_theta - self.state[2]
        
        # Wrap angle difference to [-pi, pi]
        theta_diff = (theta_diff + np.pi) % (2*np.pi) - np.pi
        
        # Control input: steer orientation toward desired angle
        self.u_theta = self.angle_gain * theta_diff
        
        # Add small random perturbation to prevent synchronization deadlock
        # if self.flag:
        #     if np.random.random() < 0.01:  # Reduced randomness frequency
        #         self.u_theta += 0.05 * np.random.randn()
    
    def updateState(self):
        """Update the state based on dynamics with wrapping of position if out of bounds"""
        if self.is_obstacle:
            return
        
        # Get velocity unit direction vector
        direction = self.getUnitDirectionVector()
        
        # Speed is constant (not updated)
        speed = self.state[3]
        
        # Update position using constant speed
        self.state[0] += direction[0] * speed * self.nominaldt
        self.state[1] += direction[1] * speed * self.nominaldt
        
        # Update orientation
        self.state[2] += (self.w0 + self.u_theta) * self.nominaldt
        
        # Wrap theta to [-pi, pi]
        self.state[2] = (self.state[2] + np.pi) % (2*np.pi) - np.pi
        
        # # Wrap X, Y coordinates if needed
        # bounds = 5.0
        # for i in range(2):
        #     if self.state[i] > bounds:
        #         self.state[i] -= 2 * bounds
        #     elif self.state[i] < -bounds:
        #         self.state[i] += 2 * bounds 