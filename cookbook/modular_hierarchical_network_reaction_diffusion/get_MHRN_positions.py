import numpy as np
import networkx as nx
import netwulf as nw
import cMHRN

# load edges from txt file and construct Graph object
N, edges = cMHRN.fast_mhrn(8,3,7,0.18,True)
G = nx.Graph()
G.add_edges_from(edges)

# visualize and save visualization
network, config = nw.visualize(G)
nw.save("MHRN.json",network,config)
