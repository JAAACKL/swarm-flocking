import numpy as np

class Node3D:
    def __init__(self, uid):
        # Basic information
        self.uid = uid    # node UID (an integer)
        
        # State variables: [x, y, z, theta, phi, speed] where theta is the azimuth angle and phi is the elevation angle
        self.state = np.zeros(6)
        self.state[5] = 1.0  # Default speed = 1.0
        
        # Time step parameters
        self.nominaldt = 0.05
        
        # Flocking parameters
        self.collision_threshold = 0.1
        self.collisionscale = 10.0
        self.velocityscale = 5.0
        self.centeringscale = 0.0
        self.angle_gain = 2.0
        self.speed_gain = 1.0  # Gain for speed control
        
        # Angular velocity and control inputs
        self.w0 = 0.0
        self.u_theta = 0.0
        self.u_phi = 0.0
        self.u_speed = 0.0    # Speed control input
        
        # Speed limits
        self.min_speed = 0.2  # Minimum speed
        self.max_speed = 3.0  # Maximum speed
        
        # Flags
        self.is_obstacle = False
        self.flag = False

    def __str__(self):
        return f"Node {self.uid} at position ({self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f}), speed: {self.state[5]:.2f}"

    def setState(self, s):
        """Update the state of the node"""
        # If the provided state doesn't include speed, add the default speed
        if len(s) == 5:
            s = list(s) + [1.0]  # Default speed = 1.0
        self.state = np.array(s, dtype=float)
    
    def getPosition(self):
        """Return the position as a numpy array"""
        return self.state[:3]
    
    def getOrientation(self):
        """Return the orientation angles as a numpy array [theta, phi]"""
        return self.state[3:5]
    
    def getSpeed(self):
        """Return the current speed"""
        return self.state[5]
    
    def getVelocityVector(self):
        """Return the velocity vector based on orientation and speed"""
        theta, phi = self.state[3:5]
        speed = self.state[5]
        return speed * np.array([
            np.cos(theta) * np.cos(phi),
            np.sin(theta) * np.cos(phi),
            np.sin(phi)
        ])
    
    def getUnitDirectionVector(self):
        """Return the unit direction vector based on orientation"""
        theta, phi = self.state[3:5]
        return np.array([
            np.cos(theta) * np.cos(phi),
            np.sin(theta) * np.cos(phi),
            np.sin(phi)
        ])
    
    def computeControl(self, all_nodes):
        """
        Compute control inputs based on all nodes in the system.
        Uses vectorized operations for efficiency.
        """
        if self.is_obstacle:
            return

        # Get current node's position and velocity direction (unit vector)
        pos_i = self.getPosition()
        dir_i = self.getUnitDirectionVector()
        vel_i = self.getVelocityVector()
        current_speed = self.getSpeed()
        
        # Collect positions and velocities of all other nodes
        other_nodes = [node for node in all_nodes if node.uid != self.uid]
        
        if not other_nodes:
            self.u_theta = 0
            self.u_phi = 0
            self.u_speed = 0
            return
        
        # Vectorized operations for all neighbors
        positions = np.array([node.getPosition() for node in other_nodes])
        velocities = np.array([node.getVelocityVector() for node in other_nodes])
        speeds = np.array([node.getSpeed() for node in other_nodes])
        is_robot = np.array([not node.is_obstacle for node in other_nodes])
        
        # Calculate displacement vectors from current node to all others
        displacement_vectors = positions - pos_i
        
        # Calculate distances
        distances = np.linalg.norm(displacement_vectors, axis=1)
        
        # 1. Collision Avoidance (vectorized)
        close_indices = distances < self.collision_threshold
        if np.any(close_indices):
            close_displacements = displacement_vectors[close_indices]
            close_distances = distances[close_indices].reshape(-1, 1)
            # Avoid division by zero
            close_distances = np.maximum(close_distances, 1e-6)
            repulsion_vectors = -close_displacements / (close_distances**2)
            avoidance = np.sum(repulsion_vectors, axis=0)
            
            # Reduce speed when close to obstacles
            self.u_speed = -self.speed_gain
        else:
            avoidance = np.zeros(3)
        
        # 2. Velocity Matching (vectorized)
        robot_indices = is_robot
        if np.any(robot_indices):
            robot_velocities = velocities[robot_indices]
            robot_speeds = speeds[robot_indices]
            match = np.mean(robot_velocities, axis=0) - vel_i
            
            # Speed matching - try to match average speed of other robots
            avg_speed = np.mean(robot_speeds)
            speed_diff = avg_speed - current_speed
            self.u_speed = self.speed_gain * speed_diff
        else:
            match = np.zeros(3)
            self.u_speed = 0
        
        # 3. Flock Centering (vectorized)
        if np.any(robot_indices):
            robot_positions = positions[robot_indices]
            center = np.mean(robot_positions, axis=0) - pos_i
        else:
            center = np.zeros(3)
        
        # Combine influences for direction
        V = (self.collisionscale * avoidance + 
             self.velocityscale * match + 
             self.centeringscale * center)
        
        # If V is near zero, maintain current direction and speed
        if np.linalg.norm(V) < 1e-6:
            self.u_theta = 0
            self.u_phi = 0
            # Let u_speed gradually return to default speed=1.0
            self.u_speed = self.speed_gain * (1.0 - current_speed)
            return
        
        # Convert desired velocity vector to desired orientation angles
        desired_theta = np.arctan2(V[1], V[0])
        desired_phi = np.arcsin(np.clip(V[2] / np.linalg.norm(V), -1.0, 1.0))
        
        # Calculate angle differences
        theta_diff = desired_theta - self.state[3]
        phi_diff = desired_phi - self.state[4]
        
        # Wrap angle differences to [-pi, pi]
        theta_diff = (theta_diff + np.pi) % (2*np.pi) - np.pi
        phi_diff = (phi_diff + np.pi) % (2*np.pi) - np.pi
        
        # Control inputs: steer orientation toward desired angles
        self.u_theta = self.angle_gain * theta_diff
        self.u_phi = self.angle_gain * phi_diff
        
        # Add small random perturbation to prevent synchronization deadlock
        if self.flag:
            if np.random.random() < 0.2:
                self.u_theta += 0.1 * np.random.randn()
                self.u_phi += 0.1 * np.random.randn()
                self.u_speed += 0.05 * np.random.randn()
    
    def updateState(self):
        """Update the state based on dynamics with wrapping of position if out of bounds"""
        if self.is_obstacle:
            return
        
        # Get velocity unit direction vector
        direction = self.getUnitDirectionVector()
        
        # Update speed
        self.state[5] += self.u_speed * self.nominaldt
        
        # Clamp speed to limits
        self.state[5] = np.clip(self.state[5], self.min_speed, self.max_speed)
        
        # Update position using current speed
        speed = self.state[5]
        self.state[0] += direction[0] * speed * self.nominaldt
        self.state[1] += direction[1] * speed * self.nominaldt
        self.state[2] += direction[2] * speed * self.nominaldt
        
        # Update orientation
        self.state[3] += (self.w0 + self.u_theta) * self.nominaldt
        self.state[4] += self.u_phi * self.nominaldt
        
        # Wrap phi to prevent gimbal lock
        self.state[4] = np.clip(self.state[4], -np.pi/2 + 0.01, np.pi/2 - 0.01)
        
        # Wrap X, Y, Z coordinates if needed
        bounds = 1.5
        for i in range(3):
            if self.state[i] > bounds:
                self.state[i] -= 2 * bounds
            elif self.state[i] < -bounds:
                self.state[i] += 2 * bounds 