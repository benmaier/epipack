import numpy as np
import networkx as nx
import netwulf as nw

# load edges from txt file and construct Graph object
edges = np.loadtxt('facebook_combined.txt')
G = nx.Graph()
G.add_edges_from(edges)

# visualize and save visualization
network, config = nw.visualize(G)
nw.save("FB.json",network,config)
