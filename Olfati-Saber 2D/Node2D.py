import numpy as np
from Obstacle2D import beta_agent_from_obstacle
import time
class Node2D:
    def __init__(self, uid):
        # Basic information
        self.uid = uid    # node UID (an integer)
        self.is_beta = False
        self.is_gamma = False
        self.is_eaten = False  # Flag to track if prey has been eaten by a predator
        self.is_food = False   # Flag to indicate if this node is food
        self.has_obstruction = True
        self.is_predator       = False
        self.predator_strategy = 1      # 1=closest α, 2=mean in range, 3=mean+proj v
        self.predator_range    = 200.0  # only for strategy 2 & 3
        self.predator_gain     = 1500.0  # how strongly it accelerates toward its target
        self.projection_horizon = 50.0
        
        # State variables: [x, y, vx, vy] - position and velocity components
        self.state = np.zeros(4)
        
        # Time step parameters
        self.nominaldt = 0.01
        
        # Flocking parameters
        # Parameters
        self.eps = 0.1  # sigma-norm parameter
        self.r = 100.0
        self.d = 20.0  # Desired inter-agent distance
        self.a = 5.0
        self.b = 15.0
        self.c = abs(self.a - self.b) / (2 * np.sqrt(self.a * self.b))
        self.h = 0.2
        self.max_accel = 2000.0
        self.max_rotation_rate = 4.0 * np.pi  # Maximum rotation rate in radians per time step
        self.d_beta = 20.0
        self.neighbor_num = 6
        self.max_vel = 100.0
        self.min_vel = 100.0  # Minimum velocity

        self.neighbors = []

        # Predator repulsion parameters
        self.d_predator = 100

        # Behavioral weights
        self.c1_alpha = 50.0
        self.c2_alpha = 5.0
        self.c1_beta  = 10000.0
        self.c2_beta  = 100.0
        self.c1_gamma = 300.0
        self.c2_gamma = 0.0
        self.c1_predator = 500.0
        
        # Food attraction parameters
        self.food_attraction_gain = self.c1_gamma   # How strongly it accelerates toward food
        self.consumed_food_ids = set()      # IDs of food this agent has consumed
        
        # Food properties (if this node is food)
        self.eaten_by = set()  # Set of agent IDs that have eaten this food
        self.is_consumed = False  # Flag to indicate if this food is fully consumed
        self.max_consumption = 10  # Maximum times this food can be eaten
        self.consumption_count = 0  # How many times this food has been eaten
        
        # Food cooldown tracking (for agents)
        self.food_cooldowns = 0  # Dict mapping food ID to cooldown counter
        self.cooldown_period = 1000  # Updates until attraction to a food is restored
        
        self.is_flocking = True

        # Control inputs (x and y acceleration components)
        self.u = np.zeros(2)
        
        # Default speed for initialization
        self.default_speed = 1.0
        
        # Flags
        self.flag = False

    def can_see(self, other):
        """
        Return True if 'other' at p1 is visible from p0 given neighbors blocking.
        neighbors: indices of blocking neighbors
        positions: array of node positions
        """

        if not self.has_obstruction:
            return True

        p0 = self.getPosition()
        p1 = other.getPosition()
        v1 = p1 - p0
        d1 = np.linalg.norm(v1)
        if d1 == 0:
            return True
        elif d1 > self.r:
            return False

        theta1 = np.arctan2(v1[1], v1[0])

        for ni in self.neighbors:
            vn = ni.getPosition() - p0
            dn = np.linalg.norm(vn)
            # ignore neighbor if it's at the same spot or farther than the target
            if dn == 0 or dn >= d1:
                continue
            # compute blocking half-angle: π/3 if dn≤1, else (π/4)*(1/dn)
            half_angle = (np.pi/3) * self.d/dn
            thetan = np.arctan2(vn[1], vn[0])
            dtheta = (theta1 - thetan + np.pi) % (2*np.pi) - np.pi
            if abs(dtheta) <= half_angle:
                return False

        return True

    def __str__(self):
        vel = self.getVelocity()
        speed = np.linalg.norm(vel)
        return f"Node {self.uid} at position ({self.state[0]:.2f}, {self.state[1]:.2f}), speed: {speed:.2f}"

    def setState(self, s):
        """Update the state of the node"""
        if len(s) == 4:
            # Already has position and velocity components
            self.state = np.array(s, dtype=float)
        elif len(s) == 3:
            # Has [x, y, theta] format - convert to [x, y, vx, vy]
            x, y, theta = s
            speed = self.default_speed
            vx = speed * np.cos(theta)
            vy = speed * np.sin(theta)
            self.state = np.array([x, y, vx, vy], dtype=float)
        elif len(s) == 2:
            # Only position provided, set velocity to zero
            x, y = s
            self.state = np.array([x, y, 0, 0], dtype=float)
    
    def getPosition(self):
        """Return the position as a numpy array"""
        return self.state[:2]
    
    def getVelocity(self):
        """Return the velocity vector"""
        return self.state[2:4]
    
    def getOrientation(self):
        """Return the orientation angle (for compatibility)"""
        velocity = self.getVelocity()
        return np.arctan2(velocity[1], velocity[0])
    
    def getVelocityVector(self):
        """Return the velocity vector (same as getVelocity, for compatibility)"""
        return self.getVelocity()
    
    def sigma_grad(self, x):
        """σ‑norm of scalar distance x."""
        return x / (1 + self.eps * self.sigma_norm(x))

    def sigma_norm(self, z):
        """σ‑norm of scalar distance z."""
        return (np.sqrt(1 + self.eps * np.linalg.norm(z)**2) - 1) / self.eps

    def sigma1(self, x):
        return x / np.sqrt(1 + x**2)

    def sigma1_vec(self, z_vec):
        norm = np.linalg.norm(z_vec)
        return z_vec / np.sqrt(1 + norm**2)

    def bump(self, z):
        """
        Finite‑range window ρ(z):
        = 1           , if 0 ≤ z < h
        = 0.5*(1+cos(pi*(z-h)/(1-h))) , if h ≤ z ≤ 1
        = 0           , if z > 1
        """
        return np.where(
            z < self.h,
            1.0,
            np.where(
                z <= 1.0,
                0.5 * (1 + np.cos(np.pi * (z - self.h) / (1 - self.h))),
                0.0
            )
        )

    def phi(self, z):
        """Attractive/repulsive core φ as in Eq. (16), with shift c."""
        return 0.5 * ((self.a + self.b) * self.sigma1(z + self.c) + (self.a - self.b))

    def phi_alpha(self, z, r, d):
        """finite-range core phi_alpha(z) = bump(z/r) * phi(z - d)"""
        return self.bump(z / self.sigma_norm(r)) * self.phi(z - self.sigma_norm(d))

    def psi_alpha(self,z_array):
        """pairwise potential psi_alpha(z) = ∫_d^z phi_alpha(s) ds"""
        results = []
        for zi in np.atleast_1d(z_array):
            s = np.linspace(self.sigma_norm(self.d), zi, 500)
            results.append(np.trapz(self.phi_alpha(s), s, self.r, self.d))
        return np.array(results)

    def compute_predator_repulsion(self, all_nodes):
        """Compute repulsion from predator agents and their projected positions."""
        avoid_pred = np.zeros(2)
        xi = self.getPosition()
        dt_proj = self.nominaldt * self.projection_horizon # projection time horizon

        for pred in [n for n in all_nodes if getattr(n, "is_predator", False) and n.uid != self.uid]:
            # — avoid current position —
            pos_pred = pred.getPosition()
            delta    = pos_pred - xi
            dist     = np.linalg.norm(delta)
            if dist < self.d_predator:
                avoid_pred += (
                    self.phi_alpha(self.sigma_norm(delta), self.d_predator, self.d_predator)
                    * self.sigma_grad(delta)
                )

            # — avoid projected position —
            vel_pred = pred.getVelocity()
            for i in range(2):
                proj_pos = pos_pred + vel_pred * dt_proj * (i/2)
                delta_p  = proj_pos - xi
                dist_p   = np.linalg.norm(delta_p)
                if dist_p < self.d_predator:
                    avoid_pred += (
                        self.phi_alpha(self.sigma_norm(delta_p), self.d_predator, self.d_predator)
                        * self.sigma_grad(delta_p)
                    )

        return avoid_pred

    def compute_food_attraction(self, all_nodes):
        """Compute attraction to food that hasn't been consumed by this agent."""
        attraction = np.zeros(2)
        xi = self.getPosition()

        # Update cooldowns for all foods
        if self.food_cooldowns > 0:
            self.food_cooldowns -= 1
            return 0

        # Find food nodes that haven't been fully consumed
        food_nodes = [node for node in all_nodes 
                     if getattr(node, "is_food", False) 
                     and not getattr(node, "is_consumed", False)]

        # get closest food
        closest_food = None
        closest_dist = float('inf')
        for food in food_nodes:
                
            # Calculate vector to food
            food_pos = food.getPosition()
            delta = food_pos - xi
            dist = np.linalg.norm(delta)

            # If within attraction range, generate attraction force
            if dist < self.r and dist < closest_dist:
                closest_dist = dist
                closest_food = food

        if closest_food is not None:
            # Calculate vector to closest food
            food_pos = closest_food.getPosition()
            delta = food_pos - xi
            dist = np.linalg.norm(delta)

            # Apply attraction force
            attraction += (self.food_attraction_gain 
                           * self.phi_alpha(self.sigma_norm(delta), self.r, 0)
                           * self.sigma_grad(delta))
        return attraction

    def compute_obstacle_repulsion(self, obstacles):
        """Compute repulsion from static obstacles."""
        avoid_obs = np.zeros(2)
        obs_cohesion = np.zeros(2)
        xi = self.getPosition()
        vi = self.getVelocity()
        for obs in obstacles:
            q_hat, p_hat = beta_agent_from_obstacle(xi, vi, obs)
            delta = q_hat - xi
            dist = np.linalg.norm(delta)
            if dist < self.d_beta:
                avoid_obs += self.phi_alpha(self.sigma_norm(delta), self.d_beta, self.d_beta) * self.sigma_grad(delta)
                w = self.bump(self.sigma_norm(delta) / self.sigma_norm(self.d_beta))
                obs_cohesion += w * (p_hat - vi)
        return avoid_obs, obs_cohesion
    
    def compute_agent_repulsion(self, beta_agents):
        """Compute repulsion from static obstacles."""
        avoid_obs = np.zeros(2)
        obs_cohesion = np.zeros(2)
        xi = self.getPosition()
        vi = self.getVelocity()
        for n in beta_agents:
            q_hat, p_hat = n.getPosition(), n.getVelocity()
            if  np.linalg.norm(p_hat) > 100:
                continue
            delta = q_hat - xi
            dist = np.linalg.norm(delta)
            if dist < self.d_beta:
                avoid_obs += self.phi_alpha(self.sigma_norm(dist), self.d_beta, self.d_beta) * self.sigma_grad(delta)
                w = self.bump(self.sigma_norm(dist) / self.sigma_norm(self.d_beta))
                obs_cohesion += w * (p_hat - vi)
        return 10 * avoid_obs, 0
    
    def computeControlForPredator(self, all_nodes, obstacles):

        # if np.random.rand() > 0.05:
        #     return

        xi = self.getPosition()
        vi = self.getVelocity()

        alphas = [
            n for n in all_nodes
            if not n.is_beta and not n.is_gamma and not n.is_predator and not n.is_eaten
        ]
        if not alphas:
            self.u = np.zeros(2)
            return

        # 1) closest
        if self.predator_strategy == 1:
            dists = [np.linalg.norm(n.getPosition() - xi) for n in alphas]
            target = alphas[int(np.argmin(dists))].getPosition()

        # 2) average of those within predator_range
        elif self.predator_strategy == 2:
            in_range = [
                n.getPosition()
                for n in alphas
                if np.linalg.norm(n.getPosition() - xi) <= self.predator_range
            ]
            if in_range:
                target = np.mean(in_range, axis=0)
            else:
                target = alphas[0].getPosition()  # fallback

        # 3) same as (2) but project by current velocity
        else:
            in_range = [
                n.getPosition()
                for n in alphas
                if np.linalg.norm(n.getPosition() + vi * self.projection_horizon - xi) <= self.predator_range
            ]
            if in_range:
                avg_pos = np.mean(in_range, axis=0)
                target = avg_pos  # simple projection
            else:
                target = alphas[0].getPosition()

        # steer straight at `target`
        delta = target - xi
        norm  = np.linalg.norm(delta)
        if norm > 1e-6:
            dir_vec = (self.phi_alpha(self.sigma_norm(delta), self.predator_range, 0)
                        * self.sigma_grad(delta))
        else:
            dir_vec = np.zeros(2)

        predator_repulsion = self.c1_alpha * self.compute_predator_repulsion(all_nodes)
        
        # obstacle avoidance
        avoid_obs, obs_cohesion = self.compute_obstacle_repulsion(obstacles)
        u_obstacle = self.c1_beta * avoid_obs + self.c2_beta * obs_cohesion
            
        self.u = self.predator_gain * dir_vec + u_obstacle + predator_repulsion
        return
    
    def computeControl(self, all_nodes, obstacles):
        
        if getattr(self, "is_gamma", False):
            # self.u = np.array([np.sin(time.time() / 10), np.cos(time.time() / 10)]) * 10
            return
        if getattr(self, "is_beta", False):
            return
        if getattr(self, "is_eaten", False):
            self.u = np.zeros(2)
            return
        if getattr(self, "is_food", False):
            # Food doesn't move
            self.u = np.zeros(2)
            return

        xi = self.getPosition()
        vi = self.getVelocity()
        
        # ———————————— predator logic ————————————
        if getattr(self, "is_predator", False):
            # collect all α‑agents (excluding eaten prey)
            self.computeControlForPredator(all_nodes, obstacles)
            return

        # ————— collect special agents ——————————————
        gamma_nodes = [n for n in all_nodes if getattr(n, "is_gamma", False)]

        gamma_pos = np.mean([n.getPosition() for n in gamma_nodes], axis=0) if gamma_nodes else None
        gamma_vel = np.mean([n.getVelocity() for n in gamma_nodes], axis=0) if gamma_nodes else None


        # ————— initialize accumulators ——————————————
        cohesion = np.zeros(2)     # for α second term
        obs_cohesion = np.zeros(2) # for β second term

        # ————— compute pairwise forces —————————————————

        repulsion = np.zeros(2)   # from other robots

        # Find the closest two alpha agents (non-beta, non-gamma, non-eaten agents)
        alpha_agents = [n for n in all_nodes if not getattr(n, "is_beta", False) \
                        and not getattr(n, "is_gamma", False) \
                        and not getattr(n, "is_predator", False) \
                        and not getattr(n, "is_food", False) \
                        and n.uid != self.uid]
        
        if self.is_flocking:
            # Calculate distances to all alpha agents
            distances = []
            for agent in alpha_agents:
                dist = np.linalg.norm(agent.getPosition() - xi)
                distances.append((dist, agent))
            
            # Sort by distance and get the closest two
            distances.sort(key=lambda x: -x[0])
            self.neighbors = []
            while len(self.neighbors) < self.neighbor_num and len(distances) > 0:
                if self.can_see(distances[-1][1]):
                    self.neighbors.append(distances[-1][1])
                distances.pop()
            
            for agent in self.neighbors:
                delta = agent.getPosition() - xi
                dist = np.linalg.norm(delta)
                if dist < self.r:
                    repulsion += self.phi_alpha(self.sigma_norm(delta), self.r, self.d) * self.sigma_grad(delta)
                    cohesion  += self.bump(self.sigma_norm(dist)/self.sigma_norm(self.r)) * (agent.getVelocity() - vi)

        else:
            repulsion, cohesion = self.compute_agent_repulsion(alpha_agents)
        
        # ————— compute beta forces ————————————————————

        # ————— compose α and β contributions —————————————————
        u_alpha = self.c1_alpha * repulsion + self.c2_alpha * cohesion

        # obstacle avoidance
        avoid_obs, obs_cohesion = self.compute_obstacle_repulsion(obstacles)
        u_obstacle = self.c1_beta * avoid_obs + self.c2_beta * obs_cohesion

        # predator avoidance
        avoid_pred = self.compute_predator_repulsion(all_nodes)
        u_predator = self.c1_predator * avoid_pred

        # combine all contributions
        self.u = u_alpha + u_obstacle + u_predator

        # ————— food attraction ————————————————————
        u_food = np.zeros(2)
        u_food = self.compute_food_attraction(all_nodes)
        self.u += u_food

        # ————— navigation (γ‑agent) ————————————————————
        u_gamma = np.zeros(2)
        if gamma_pos is not None:
            # u^gamma = -c1_gamma * σ1_vec(xi - gamma_pos) - c2_gamma * (vi - gamma_vel)
            u_gamma = -self.c1_gamma * self.sigma1_vec(xi - gamma_pos) - self.c2_gamma * (vi - gamma_vel)

        # ————— combine ——————————————————
        self.u += u_gamma

        # if np.linalg.norm(vi) < 10:
        #     if np.random.rand() < 0.05:
        #         self.state[2:4] = np.random.randn(2) * 200

        return
    
    def updateState(self):
        """Update the state based on dynamics with wrapping of position if out of bounds"""

        if getattr(self, "is_beta", False):
            return
        if getattr(self, "is_gamma", False):
            # self.state[2:4] = [np.sin(time.time() / 2) * 300, np.cos(time.time() / 2) * 300]
            # self.state[0:2] = [np.sin(time.time() / 2) * 300, np.cos(time.time() / 2) * 300]
            return
        if getattr(self, "is_food", False):
            # Food doesn't move
            self.state[2:4] = [0, 0]
            return

        # Get current orientation and speed
        current_orientation = self.getOrientation()
        current_speed = np.linalg.norm(self.getVelocity())
        
        # Compute direction of acceleration
        accel_direction = np.arctan2(self.u[1], self.u[0]) if np.linalg.norm(self.u) > 0 else current_orientation
        
        # Project acceleration along current velocity to compute speed change
        vel_unit = self.getVelocity() / (current_speed if current_speed > 0 else 1.0)
        speed_accel = np.dot(self.u, vel_unit)
        
        # Compute rate of orientation change based on control input magnitude
        accel_mag = np.linalg.norm(self.u)
        rotation_rate = min(accel_mag / self.max_accel, 1.0) * self.max_rotation_rate
        
        # Compute new orientation by gradually turning toward the acceleration direction
        angle_diff = (accel_direction - current_orientation + np.pi) % (2 * np.pi) - np.pi
        new_orientation = current_orientation + np.sign(angle_diff) * min(abs(angle_diff), rotation_rate * self.nominaldt)
        
        # Update speed based on acceleration
        new_speed = current_speed + speed_accel * self.nominaldt
        
        # Clamp speed between min_vel and max_vel
        new_speed = max(self.min_vel, min(new_speed, self.max_vel))
        
        # Update velocity using new speed and orientation
        self.state[2] = new_speed * np.cos(new_orientation)
        self.state[3] = new_speed * np.sin(new_orientation)
        
        # Update position based on velocity
        self.state[0:2] += self.state[2:4] * self.nominaldt