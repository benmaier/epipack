"""
Some network models.
"""

def get_2D_lattice_links(N_side,periodic=False,diagonal_links=False):

    links = []
    for i in range(N_side):
        for j in range(N_side):
            u = i*N_side + j
            if periodic or j+1 < N_side:
                v = i*N_side + ( (j+1) % N_side )
                links.append((u,v,1.0))
            if periodic or i+1 < N_side:
                v = ((i+1) % N_side)*N_side + j 
                links.append((u,v,1.0))

            if diagonal_links:
                if periodic or (j+1 < N_side and i+1 <N_side):
                    v = ((i+1) % N_side)*N_side + ( (j+1) % N_side )
                    links.append((u,v,1.0))
                if periodic or (j-1 >= 0 and i+1 < N_side):
                    v = ((i+1) % N_side)*N_side + ( (j-1) % N_side )
                    links.append((u,v,1.0))

    return links


