"""
Some network models and layout functions.
"""

import numpy as np
from scipy.spatial import KDTree

def _edge(u,v,w):
    u, v = sorted([u,v])
    return u, v, w

def _dist(i,j,N):
    return min(np.abs(i-j), N-np.abs(i-j))

def get_RGG_links_and_network(N,k,
                          windowwidth=400,
                          linkwidth=1,
                          node_scale_by_degree=0.5,
                          node_radius_scale = 1/3,
                          pos = None,
                      ):
    """
    Return the links and the stylized network
    of a non-periodic random geometric graph
    on a square.

    Parameters
    ==========
    N : int
        Number of nodes
    k : float
        mean degree
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
    node_radius_scale : float, default = 1/3
        Factor by which the default node size is scaled.
    pos : numpy.ndarray, default = None
        If ``None``, node positions will be drawn uniform at random.
        If not ``None``, should be position array of shape ``N x 2``.

    Returns
    =======
    edge_weight_tuples : list of tuple
        list of tuples that are structured like ``(source, target, weight)``
    network : dict
        stylized network that can be passed to the visualization
    """


    if pos is None:
        pos = np.random.rand(N,2)
        ndx = np.argsort(pos[:,0])
        pos = pos[ndx,:]

    tree = KDTree(pos)

    V = N-1
    R = np.sqrt(k/V/np.pi)

    pairs = tree.query_pairs(R)
    edge_weight_tuples = []
    for u, v in pairs:
        edge_weight_tuples.append(_edge(int(u),int(v),1.0))

    w = h = windowwidth
    N_side = int(np.ceil(np.sqrt(N)))
    dx = w / N_side
    radius = dx * node_radius_scale

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

    pos *= w

    nodes = [ {
                'id': i,
                'x_canvas': _pos[0],
                'y_canvas': _pos[1],
                'radius': radius[i],
              } for i, _pos in enumerate(pos)]

    nodes = nodes[:N]
    links = [ {'source': u, 'target': v, 'width': linkwidth} for u, v, w in edge_weight_tuples ]

    stylized_network['nodes'] = nodes
    stylized_network['links'] = links

    return edge_weight_tuples, stylized_network

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

def get_small_world_layout(
                      N_nodes,
                      edge_weight_tuples=[],
                      windowwidth=400,
                      linkwidth=1,
                      node_scale_by_degree=2,
                      nbounce=20,
                      Rbounce=0.1,
                      R=0.8,
                      ):
    """
    Returns a stylized network dictionary that puts
    nodes in a small-world inspired circular layout.
    The ring of nodes will be drawn bouncy
    to better display connections between nearby regions.
    Nodes that connect far-away regions of a network
    will be displayed more centrally.
    Distance is defined as lattice distance by integer
    node id. Hence, nodes must be integers in [0,N).

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
    node_scale_by_degree : float, default = 2
        Scale the node radius by ``degree**node_scale_by_degree``.
        If you want
        all nodes to be equally sized, set
        ``node_scale_by_degree = 0``.
    nbounce : int, default = 20
        How wobbly the outer shell should be.
    Rbounce : int, default = 0.1
        How thick the outer shell should be
        (in units of half window width)
    R : float, default = 0.8
        How large the radius of the whole layout should be
        (in units of half window width)

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

    Rbounce /= R
    R = R * w/2
    Rbounce *= R

    network = {}
    stylized_network = {
        'xlim': [0, w],
        'ylim': [0, h],
        'linkAlpha': 0.5,
        'nodeStrokeWidth': 0.0001,
    }

    degree = np.zeros(N,)
    node_distances = [[] for n in range(N)]
    for u, v, _w in edge_weight_tuples:
        node_distances[u].append(_dist(u,v,N))
        node_distances[v].append(_dist(u,v,N))
        degree[u] += 1
        degree[v] += 1

    median_degree = np.median(degree)
    if median_degree == 0:
        median_degree = 1
    radius_scale = (degree/median_degree)**node_scale_by_degree
    radius_scale[radius_scale==0] = 1.0
    radius = radius_scale * radius

    node_distance_maxs = [ np.max(dists+[0]) for dists in node_distances ]
    distance_measure = node_distance_maxs

    dphi = 2*np.pi/N
    maxmean = 3*N/4

    pos = np.zeros((N, 2))

    for n in range(N):
        phi = n*dphi
        rbase = R+Rbounce*np.sin(nbounce*phi)
        r = rbase*(1-distance_measure[n]/maxmean)
        x = r*np.cos(phi) + w/2
        y = r*np.sin(phi) + w/2
        pos[n,:] = x, y

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

def get_layout_from_positions(
                      N_nodes,
                      pos,
                      edge_weight_tuples=[],
                      windowwidth=400,
                      linkwidth=1,
                      node_scale_by_degree=0.5,
                      radius_scale=0.333,
                      scale_position=False,
                      ):
    """
    Returns a stylized network dictionary that puts
    nodes in a random layout.

    Parameters
    ==========
    N_nodes : int
        The number of nodes in the network
    pos : numpy.ndarray
        of shape ``(N_nodes, 2)``, carrying positions
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
    radius_scale : float, default = 1/3
        number to scale the node radius with
    scale_position : bool, default = False
        whether or not to rescale the position
        array such that the layout fits in the window.

    Returns
    =======
    network : dict
        A stylized network dictionary in netwulf-format.
    """

    w = h = windowwidth
    N = N_nodes
    N_side = int(np.ceil(np.sqrt(N)))
    dx = w / N_side
    radius = dx * radius_scale

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

    if scale_position:
        xmin, xmax = np.min(pos[:,0]), np.max(pos[:,0])
        ymin, ymax = np.min(pos[:,1]), np.max(pos[:,1])
        pos[:,0] = (pos[:,0] - xmin) / (xmax-xmin) * w
        pos[:,1] = (pos[:,1] - ymin) / (ymax-ymin) * h


    nodes = [ {
                'id': i,
                'x_canvas': float(pos[0]),
                'y_canvas': float(pos[1]),
                'radius': radius[i],
              } for i, pos in enumerate(pos)]
    nodes = nodes[:N_nodes]
    links = [ {'source': u, 'target': v, 'width': linkwidth} for u, v, w in edge_weight_tuples ]

    stylized_network['nodes'] = nodes
    stylized_network['links'] = links


    return stylized_network
