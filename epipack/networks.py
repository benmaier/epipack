"""
Some network models and layout functions.
"""

import numpy as np

def _edge(u,v,w):
    u, v = sorted([u,v])
    return u, v, w

def get_2D_lattice_links(N_side,periodic=False,diagonal_links=False):
    """
    Return the links of a square 2D lattice

    Parameters
    ==========
    N_side : int
        Number of nodes per side of the square
    periodic : bool, default = False
        Whether or not to apply periodic boundary conditions
    diagonal_links : bool, default = False
        Whether or not to connect to diagonal neighbors, too.
    """

    links = []
    for i in range(N_side):
        for j in range(N_side):
            u = i*N_side + j
            if periodic or j+1 < N_side:
                v = i*N_side + ( (j+1) % N_side )
                links.append(_edge(u,v,1.0))
            if periodic or i+1 < N_side:
                v = ((i+1) % N_side)*N_side + j 
                links.append(_edge(u,v,1.0))

            if diagonal_links:
                if periodic or (j+1 < N_side and i+1 <N_side):
                    v = ((i+1) % N_side)*N_side + ( (j+1) % N_side )
                    links.append(_edge(u,v,1.0))
                if periodic or (j-1 >= 0 and i+1 < N_side):
                    v = ((i+1) % N_side)*N_side + ( (j-1) % N_side )
                    links.append(_edge(u,v,1.0))

    if N_side == 2:
        links = list(set(links))

    return links

def get_grid_layout(N_nodes,edge_weight_tuples=[],windowwidth=400,linkwidth=1):
    """
    Returns a stylized network dictionary that puts 
    nodes in a grid layout.

    Parameters
    ==========
    N_nodes : int
        The number of nodes in the network
    edge_weight_tuples : list, default = []
        A list of tuples. Each tuple describes an edge
        with the first entry being the source node index,
        the second entry being the target node indes
        and the third entry being the weight, e.g.

        .. code:: python

            [ (0, 1, 1.0) ]

    windowwidth : float, default = 400
        The width of the network visualization
    linkwidth : float, default = 1.0 
        All links get the same width.

    Returns
    =======
    network : dict
        A stylized network dictionary in netwulf-format.
    """

    w = h = windowwidth
    N = N_nodes
    N_side = int(np.ceil(np.sqrt(N)))
    dx = w / N_side
    radius = dx/2

    network = {}
    stylized_network = {
        'xlim': [0, w],
        'ylim': [0, h],
        'linkAlpha': 0.5,
        'nodeStrokeWidth': 0.0001,
    }

    nodes = [ {
                'id': i*N_side + j,
                'x_canvas': (i+0.5)*dx,
                'y_canvas': (j+0.5)*dx,
                'radius': radius
              } for i in range(N_side) for j in range(N_side) ]
    nodes = nodes[:N_nodes]
    links = [ {'source': u, 'target': v, 'width': linkwidth} for u, v, w in edge_weight_tuples ]

    stylized_network['nodes'] = nodes
    stylized_network['links'] = links

    return stylized_network

def get_random_layout(N_nodes,
                      edge_weight_tuples=[],
                      windowwidth=400,
                      linkwidth=1,
                      node_scale_by_degree=0.5,
                      circular=True,
                      ):
    """
    Returns a stylized network dictionary that puts 
    nodes in a random layout.

    Parameters
    ==========
    N_nodes : int
        The number of nodes in the network
    edge_weight_tuples : list, default = []
        A list of tuples. Each tuple describes an edge
        with the first entry being the source node index,
        the second entry being the target node index
        and the third entry being the weight, e.g.

        .. code:: python

            [ (0, 1, 1.0) ]

    windowwidth : float, default = 400
        The width of the network visualization
    linkwidth : float, default = 1.0 
        All links get the same width.
    node_scale_by_degree : float, default = 0.5
        Scale the node radius by ``degree**node_scale_by_degree``.
        Per default, the node disk area will be
        proportional to the degree. If you want 
        all nodes to be equally sized, set
        ``node_scale_by_degree = 0``.
    circular : bool, default = True
        Use a circular or square layout

    Returns
    =======
    network : dict
        A stylized network dictionary in netwulf-format.
    """

    w = h = windowwidth
    N = N_nodes
    N_side = int(np.ceil(np.sqrt(N)))
    dx = w / N_side
    radius = dx/3

    network = {}
    stylized_network = {
        'xlim': [0, w],
        'ylim': [0, h],
        'linkAlpha': 0.5,
        'nodeStrokeWidth': 0.0001,
    }

    degree = np.zeros(N,)
    for u, v, _w in edge_weight_tuples:
        degree[u] += 1
        degree[v] += 1

    median_degree = np.median(degree)
    if median_degree == 0:
        median_degree = 1
    radius_scale = (degree/median_degree)**node_scale_by_degree
    radius_scale[radius_scale==0] = 1.0
    radius = radius_scale * radius

    if not circular:
        pos = (np.random.rand(N, 2) * w).tolist()
    else:
        R = w/2
        r = R*np.sqrt(np.random.random(N,))
        t = 2*np.pi*np.random.random(N,)
        pos = np.zeros((N, 2))
        pos[:,0] = r*np.cos(t) + w/2
        pos[:,1] = r*np.sin(t) + w/2


    nodes = [ {
                'id': i,
                'x_canvas': pos[0],
                'y_canvas': pos[1],
                'radius': radius[i],
              } for i, pos in enumerate(pos)]
    nodes = nodes[:N_nodes]
    links = [ {'source': u, 'target': v, 'width': linkwidth} for u, v, w in edge_weight_tuples ]

    stylized_network['nodes'] = nodes
    stylized_network['links'] = links


    return stylized_network

