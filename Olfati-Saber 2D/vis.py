import matplotlib.pyplot as plt
import numpy as np
import math

def can_see(p0, p1, neighbors, positions):
    """
    Return True if 'other' at p1 is visible from p0 given neighbors blocking.
    neighbors: indices of blocking neighbors
    positions: array of node positions
    """
    v1 = p1 - p0
    d1 = np.linalg.norm(v1)
    if d1 == 0:
        return True

    theta1 = math.atan2(v1[1], v1[0])

    for ni in neighbors:
        vn = positions[ni] - p0
        dn = np.linalg.norm(vn)
        # ignore neighbor if it's at the same spot or farther than the target
        if dn == 0 or dn >= d1:
            continue
        # compute blocking half-angle: π/4 if dn≤1, else (π/4)*(1/dn)
        half_angle = (math.pi/4) * (1 if dn <= 0.2 else 0.2/dn)
        thetan = math.atan2(vn[1], vn[0])
        dtheta = (theta1 - thetan + math.pi) % (2*math.pi) - math.pi
        if abs(dtheta) <= half_angle:
            return False

    return True

def visualize(nodes, adjacency):
    """
    nodes: Nx2 array of positions
    adjacency: list of lists; adjacency[i] = neighbors of i
    """
    N = len(nodes)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    # Plot nodes
    ax.scatter(nodes[:,0], nodes[:,1], color='black', s=10000)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)

    # Draw blocking fans for each neighbor
    for i in range(N):
        p0 = nodes[i]
        for ni in adjacency[i]:
            visible = can_see(p0, nodes[ni], adjacency[i], nodes)
            if not visible:
                continue

            vn = nodes[ni] - p0
            dn = np.linalg.norm(vn)
            theta_n = math.atan2(vn[1], vn[0])
            half_angle = (math.pi/4) * (1 if dn <= 0.21 else 0.21/dn)
            fan_radius = 2.5
            # Create the fan arc
            angles = np.linspace(theta_n - half_angle, theta_n + half_angle, 50)
            fan_x = p0[0] + np.cos(angles) * fan_radius
            fan_y = p0[1] + np.sin(angles) * fan_radius
            ax.fill(np.concatenate(([p0[0]], fan_x)),
                    np.concatenate(([p0[1]], fan_y)),
                    alpha=0.2)

    # Draw visibility lines
    for i in range(N):
        p0 = nodes[i]
        for j in range(N):
            if i == j:
                continue
            visible = can_see(p0, nodes[j], adjacency[i], nodes)
            color = 'green' if visible else 'red'
            ax.plot([p0[0], nodes[j][0]], [p0[1], nodes[j][1]], color=color, linewidth=0.8)

    plt.show()

# Example usage:
nodes = np.array([[0.7, 0.4], [1, 0], [0, 1]])  # 10 random nodes in a 10×10 area
# For demo: each node's 2 nearest neighbors are potential blockers
adjacency = []
for i in range(len(nodes)):
    dists = np.linalg.norm(nodes - nodes[i], axis=1)
    nearest = np.argsort(dists)[1:3]  # skip self, pick next two
    adjacency.append(list(nearest))

visualize(nodes, adjacency)