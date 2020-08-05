import numpy as np
import scipy.sparse as sprs

from epipack import get_2D_lattice_links
from epipack.vis import visualize_reaction_diffusion, get_grid_layout

# ============ define relevant functions =============

# an efficient function to compute a mean over neighboring cells
def apply_laplacian(mat):
    """This function applies a discretized Laplacian
    in periodic boundary conditions to a matrix
    For more information see 
    https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
    """

    # the cell appears 4 times in the formula to compute
    # the total difference
    neigh_mat = -4*mat.copy()

    # Each direct neighbor on the lattice is counted in
    # the discrete difference formula
    neighbors = [ 
                    ( 1.0,  (-1, 0) ),
                    ( 1.0,  ( 0,-1) ),
                    ( 1.0,  ( 0, 1) ),
                    ( 1.0,  ( 1, 0) ),
                ]

    # shift matrix according to demanded neighbors
    # and add to this cell with corresponding weight
    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0,1))

    return neigh_mat

# Define the update formula for chemicals A and B
def update(A, B, DA, DB, f, k, delta_t):
    """Apply the Gray-Scott update formula"""

    # compute the diffusion part of the update
    diff_A = DA * apply_laplacian(A)
    diff_B = DB * apply_laplacian(B)
    
    # Apply chemical reaction
    reaction = A*B**2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1-A)
    diff_B -= (k+f) * B

    A += diff_A * delta_t
    B += diff_B * delta_t

    return A, B

def get_initial_configuration(N, random_influence = 0.2):
    """get the initial chemical concentrations"""

    # get initial homogeneous concentrations
    A = (1-random_influence) * np.ones((N,N))
    B = np.zeros((N,N))

    # put some noise on there
    A += random_influence * np.random.random((N,N))
    B += random_influence * np.random.random((N,N))

    # get center and radius for initial disturbance
    N2, r = N//2, 50

    # apply initial disturbance
    A[N2-r:N2+r, N2-r:N2+r] = 0.50
    B[N2-r:N2+r, N2-r:N2+r] = 0.25

    return A, B

class GrayScott():

    def __init__(self,u,v,k,f,Du,Dv,periodic=True,diagonal_links=False,animate='u'):

        assert(u.shape == v.shape)
        assert(len(u.shape)==2)
        assert(u.shape[0] == u.shape[1])

        self.N = u.shape[0]

        self.u = u
        self.v = v
        self.k = k
        self.f = f
        self.Du = Du
        self.Dv = Dv
        self.animate = animate

        if animate == 'u':
            self.y0 = self.u.flatten()
        else:
            self.y0 = self.v.flatten()

    def integrate_and_return_by_index(self,t,*args,**kwargs):


        k, f, Du, Dv = self.k, self.f, self.Du, self.Dv

        old_t = t[0]

        for _t in t[1:]:
            dt = _t - old_t
            self.u, self.v = update(self.u, self.v, Du, Dv, f, k, dt)
            old_t = _t

        if self.animate == 'u':
            return self.u.flatten()
        else:
            return self.v.flatten()


if __name__=="__main__":

    NSide = 200
    N = NSide**2
    Du = 0.16
    Dv = 0.08
    f = 0.060
    k = 0.062

    u, v = get_initial_configuration(NSide,random_influence=0.2)

    dt = 20

    GS = GrayScott(u,v,k,f,Du,Dv,periodic=True,diagonal_links=True)

    network = get_grid_layout(N)

    visualize_reaction_diffusion(GS,
                                 network,
                                 dt,
                                 node_compartments=np.arange(N),
                                 n_integrator_midpoints=20,
                                 value_extent=[0,1],
                                 config={'draw_nodes_as_rectangles':True}
                                 )
