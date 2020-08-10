"""
Some network models.
"""

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


