from threading import Thread
from queue import Empty
import numpy as np
import time

class Node(Thread):
  def __init__(self, uid):
    """ Constructor """
    Thread.__init__(self)
    
    # basic information about network connectivity
    self.uid = uid    # node UID (an integer)
    self.out_nbr = [] # list of outgoing edges (see Edge class)
    self.in_nbr = []  # list of incoming edges (see Edge class)
    
    self.state = [0,0,0]   # state vars of interest ([rx, ry, theta])
    self.done = False      # termination flag
    
    self.nominaldt = 0.001 # desired time step
    self.dt = 0            # time step
    
    # Gains / placeholders for flocking control
    self.gain = -1
    self.w0 = 0
    self.u = 0
    self.flag = False

    # New attribute: special type of node called obstacle
    self.is_obstacle = False

    # Placeholders for flocking parameters
    self.neighbor_range = 1.0
    self.collision_threshold = 0.5  # distance to trigger collision avoidance
    self.collisionscale = 10.0       # weight for collision avoidance
    self.velocityscale = 1.0        # weight for velocity matching
    self.centeringscale = 1.0       # weight for flock centering
    self.angle_gain = 1.0           # gain for orientation control

  def __str__(self):
    """ Printing """
    return "Node %d has %d in_nbr and %d out_nbr" % (self.uid, len(self.in_nbr), len(self.out_nbr))

  ################################################
  #
  # Modify the graph
  #
  ################################################

  def addOutgoing(self, e):
    """ Add an edge for outgoing messages """
    self.out_nbr.append(e)
    
  def addIncoming(self, e):
    """ Add an edge for incoming messages """
    self.in_nbr.append(e)
    
  ################################################
  #
  # Set states externally
  #
  ################################################

  def setState(self, s):
    """ update the state of the node """
    self.state = s

  def terminate(self):
    """ stop sim """
    self.done = True
 
  ################################################
  #
  # MAIN LOOP
  #
  ################################################

  def run(self):
    """ Main loop: for obstacles, send messages (remain static) periodically; otherwise, update normally """
    if self.is_obstacle:
      # Obstacle nodes do not move, but they still send their state for collision avoidance
      while not self.done:
        self.send()
        time.sleep(self.nominaldt)
      return

    while (not self.done):
      start = time.time()
      self.send()
      self.transition()
      self.systemdynamics()
      end = time.time()
      time.sleep(max(self.nominaldt - (end - start), 0))

  def send(self):
    """ Send each neighbor our (x, y, theta, is_obstacle). """
    for out_edge in self.out_nbr:
      out_edge.put((self.state[0], self.state[1], self.state[2], self.is_obstacle))

  def transition(self):
    """ Implement flocking control: collision avoidance, velocity matching, flock centering 
        Now considers messages with four elements: (x, y, theta, is_obstacle) """
    neighbor_positions = []
    neighbor_vels = []
    is_robot = []

    pos_i = np.array([self.state[0], self.state[1]])

    # Gather neighbor data
    for in_edge in self.in_nbr:
      try:
        data = in_edge.get(True, 0.01)  # data is (x, y, theta, is_obstacle)
        neighbor_pos = np.array([data[0], data[1]])

        # Only consider nearby nodes
        if np.linalg.norm(neighbor_pos - pos_i) > self.neighbor_range:
          continue

        neighbor_positions.append(neighbor_pos)
        neighbor_vels.append(np.array([np.cos(data[2]), np.sin(data[2])]))
        is_robot.append(not data[3])
        
        # Optionally, data[3] (is_obstacle) can be used to adjust weights
      except Empty:
        # If no data, skip
        pass

    vel_i = np.array([np.cos(self.state[2]), np.sin(self.state[2])])

    if len(neighbor_positions) == 0:
      # No neighbors -> no control
      self.u = 0
      return

    # 1. Collision Avoidance
    avoidance = np.array([0.0, 0.0])
    for p_j in neighbor_positions:
      diff = pos_i - p_j
      dist = np.linalg.norm(diff)
      if dist < self.collision_threshold:
        # Repel from neighbor
        avoidance += diff / (dist * dist + 1e-6)

    # 2. Velocity Matching
    non_obstacle_vels = [v for v, obs in zip(neighbor_vels, is_robot) if obs]
    if len(non_obstacle_vels) > 0:
        avg_vel = np.mean(non_obstacle_vels, axis=0)
        match = avg_vel - vel_i
    else:
        match = np.array([0.0, 0.0])

    # 3. Flock Centering
    non_obstacle_pos = [v for v, obs in zip(neighbor_positions, is_robot) if obs]
    if len(non_obstacle_pos) > 0:
        avg_pos = np.mean(non_obstacle_pos, axis=0)
        center = avg_pos - pos_i
    else:
        center = np.array([0.0, 0.0])

    # Combine influences
    V = (self.collisionscale * avoidance 
         + self.velocityscale * match 
         + self.centeringscale * center)

    # If V is near zero, do nothing
    if np.linalg.norm(V) < 1e-6:
      self.u = 0
      return

    desired_theta = np.arctan2(V[1], V[0])
    angle_diff = desired_theta - self.state[2]
    # Wrap angle_diff to [-pi, pi]
    angle_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi

    # Control input: steer orientation toward desired_theta
    self.u = self.angle_gain * angle_diff

    # Example: if self.flag is set, override
    if self.flag:
      self.u = 1 if np.random.randint(0, 10) < 2 else 0

  def systemdynamics(self):
    """ Update the state based on dynamics with wrapping of position if out of bounds. 
        For obstacle nodes, do not update state (remain static)."""
    if self.is_obstacle:
      return

    new_state = self.state.copy()

    # Move forward in the direction of theta
    new_state[0] += np.cos(self.state[2]) * self.nominaldt
    new_state[1] += np.sin(self.state[2]) * self.nominaldt
    new_state[2] += (self.w0 + self.u) * self.nominaldt

    # Wrap X coordinate
    if new_state[0] > 1.5:
      new_state[0] -= 3.0
    elif new_state[0] < -1.5:
      new_state[0] += 3.0

    # Wrap Y coordinate
    if new_state[1] > 1.5:
      new_state[1] -= 3.0
    elif new_state[1] < -1.5:
      new_state[1] += 3.0

    self.state = new_state