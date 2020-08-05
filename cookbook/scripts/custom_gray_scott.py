import numpy as np
import scipy.sparse as sprs

from epipack import get_2D_lattice_links
from epipack.vis import visualize_reaction_diffusion, get_grid_layout

def get_initial_configuration(N, random_influence=0.2):
    """
    Initialize a concentration configuration. N is the side length
    of the (N x N)-sized grid.
    `random_influence` describes how much noise is added.
    """

    # We start with a configuration where on every grid cell
    # there's a lot of chemical A, so the concentration is high
    A = (1-random_influence) * np.ones((N,N)) + random_influence * np.random.random((N,N))

    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random((N,N))

    # Now let's add a disturbance in the center
    N2 = N//2
    radius = r = int(N/10.0)

    A[N2-r:N2+r, N2-r:N2+r] = 0.50
    B[N2-r:N2+r, N2-r:N2+r] = 0.25

    return A, B

class GrayScott():

    def __init__(self,u,v,k,f,Du,Dv,periodic=True,diagonal_links=False,animate='u'):

        assert(u.shape == v.shape)
        assert(len(u.shape)==2)
        assert(u.shape[0] == u.shape[1])

        self.N = u.shape[0]

        self.u = u.flatten()
        self.v = v.flatten()
        self.k = k
        self.f = f
        self.Du = Du
        self.Dv = Dv
        self.animate = animate

        links = get_2D_lattice_links(self.N,periodic=periodic,diagonal_links=diagonal_links)

        row = []
        col = []
        data = []

        for u,v,w in links:
            row.extend([v,u,u,v])
            col.extend([u,v,u,v])
            data.extend([w, w, -w, -w])

        self.L = sprs.coo_matrix((data,(row,col)),shape=(self.N**2, self.N**2),dtype=float).tocsr()

        if animate == 'u':
            self.y0 = self.u
        else:
            self.y0 = self.v

    def integrate_and_return_by_index(self,t,*args,**kwargs):

        u = self.u
        v = self.v
        k, f, Du, Dv = self.k, self.f, self.Du, self.Dv

        old_t = t[0]

        for t in t[1:]:
            uv2 = u*v**2
            dt = t - old_t
            du = -uv2 + f*(1-u) + Du*self.L.dot(u) 
            dv =  uv2 - (f+k)*v + Dv*self.L.dot(v)

            u += du*dt
            v += dv*dt
            old_t = t

        if self.animate == 'u':
            return u
        else:
            return v


if __name__=="__main__":

    NSide = 200
    N = NSide**2
    u, v = get_initial_configuration(NSide,random_influence=0.2)
    Du = 0.16
    Dv = 0.08
    f = 0.060
    k = 0.062

    #DA, DB, f, k = 0.14, 0.06, 0.035, 0.065 # bacteria

    dt = 1
    updates_per_frame = 40
    sampling_dt = updates_per_frame * dt

    GS = GrayScott(u,v,k,f,Du,Dv,periodic=True,diagonal_links=False)

    network = get_grid_layout(N,windowwidth=NSide)

    visualize_reaction_diffusion(GS,
                                 network,
                                 sampling_dt,
                                 node_compartments=np.arange(N),
                                 n_integrator_midpoints=updates_per_frame-1,
                                 #value_extent=[0.35,0.48],
                                 config={'draw_nodes_as_rectangles':True}
                                 )
    #from bfmplot import pl
    #pl.hist(GS.u)
    #pl.show()
