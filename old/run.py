import numpy as np
from Node import *
from Edge import *
from Graph import *


def generateRandomGraph(N):
    G = Graph()

    for inode in range(N):
        # randomly generate node states
        n = Node(inode)
        n.setState(np.multiply(np.random.rand(3), np.array([2,2,np.pi])) - np.array([1,1,0]))
        G.addNode(n)

        # add all-to-all edges
        for iedge in range(inode):
            G.addEdge(iedge, inode, 0)
            G.addEdge(inode, iedge, 0)

    return G

### MAIN
if __name__ == '__main__':

    # generate a random graph with 10 nodes
    G = generateRandomGraph(10)

    # print out the graph descriptor
    print(G)
    for inode in G.V:
        print(inode)

    print("========== Starting now ==========")
    print("Close the figure to stop the simulation")

    G.run()  # Start threads in nodes
    # periodicGatherPhase(G, interval=0.1)  # Start periodic phase gathering
    G.setupAnimation()  # Set up plotting, halt after 1 s

    print("Sending stop signal.....")
    G.stop()  # Send stop signal
    print("========== Terminated ==========")
