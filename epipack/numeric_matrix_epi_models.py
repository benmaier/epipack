"""
Provides an API to define Matrix epidemiological models.
"""

import numpy as np
import scipy.sparse as sprs

import warnings

from epipack.integrators import (
            integrate_dopri5,
            integrate_euler,
            IntegrationMixin,
        )

from epipack.process_conversions import (
            processes_to_rates,
            transition_processes_to_rates,
            fission_processes_to_rates,
            fusion_processes_to_rates,
            transmission_processes_to_rates,
        )


class MatrixEpiModel(IntegrationMixin):
    """
    A general class to define standard
    mean-field compartmental
    epidemiological model.

    Parameters
    ----------

    compartments : :obj:`list` of :obj:`string`
        A list containing compartment strings.

    Attributes
    ----------

    compartments : :obj:`list` of :obj:`string`
        A list containing strings that describe each compartment,
        (e.g. "S", "I", etc.).
    N_comp : :obj:`int`
        Number of compartments (including population number)
    linear_rates : scipy.sparse.csr_matrix
        Sparse matrix containing
        transition rates of the linear processes.
    quadratic_rates : scipy.sparse.csr_matrix
        List of sparse matrices that contain
        transition rates of the quadratic processes
        for each compartment.
    affected_by_quadratic_process : :obj:`list` of :obj:`int`
        List of integer compartment IDs, collecting
        compartments that are affected
        by the quadratic processes


    Example
    -------

    .. code:: python

        >>> epi = MatrixEpiModel(["S","I","R"])
        >>> print(epi.compartments)
        [ "S", "I", "R" ]


    """

    def __init__(self,compartments,initial_population_size=1,correct_for_dynamical_population_size=False):
        """
        """

        self.t0 = None
        self.y0 = None
        self.affected_by_quadratic_process = []

        self.compartments = list(compartments)
        self.compartment_ids = { C: iC for iC, C in enumerate(compartments) }
        self.initial_population_size = initial_population_size
        self.correct_for_dynamical_population_size = correct_for_dynamical_population_size
        self.N_comp = len(self.compartments)
        self.birth_rates = np.zeros((self.N_comp,),dtype=np.float64)
        self.linear_rates = sprs.csr_matrix((self.N_comp, self.N_comp),dtype=np.float64)
        self.quadratic_rates = [ sprs.csr_matrix((self.N_comp, self.N_comp),dtype=np.float64)\
                                 for c in range(self.N_comp) ]

    def get_compartment_id(self,C):
        """Get the integer ID of a compartment ``C``"""
        return self.compartment_ids[C]

    def get_compartment(self,iC):
        """Get the compartment, given an integer ID ``iC``"""
        return self.compartments[iC]

    def set_processes(self,
                      process_list,
                      allow_nonzero_column_sums=False,
                      reset_rates=True,
                      ignore_rate_position_checks=False,
                      ):
        """
        Converts a list of reaction process tuples to rate tuples and sets the rates for this model.

        Parameters
        ----------
        process_list : :obj:`list` of :obj:`tuple`
            A list containing reaction processes in terms of tuples.

            .. code:: python

                [
                    # transition process
                    ( source_compartment, rate, target_compartment),

                    # transmission process
                    ( coupling_compartment_0, coupling_compartment_1, rate, target_compartment_0, target_ccompartment_1),

                    # fission process
                    ( source_compartment, rate, target_compartment_0, target_ccompartment_1),

                    # fusion process
                    ( source_compartment_0, source_compartment_1, rate, target_compartment),

                    # death process
                    ( source_compartment, rate, None),

                    # birth process
                    ( None, rate, target_compartment),
                ]
        allow_nonzero_column_sums : bool, default : False
            Traditionally, epidemiological models preserve the
            total population size. If that's not the case,
            switch off testing for this.
        reset_rates : bool, default : True
            If this is `True`, reset all rates to zero before setting the new ones.
        ignore_rate_position_checks : bool, default = False
            This function usually checks whether the rate of
            a reaction is positioned correctly. You can
            turn this behavior off for transition, birth, death, and
            transmission processes. (Useful if you want to define
            symbolic transmission processes that are compartment-dependent).
        """

        quadratic_rates, linear_rates = processes_to_rates(process_list,
                                                           self.compartments,
                                                           ignore_rate_position_checks,
                                                           )
        self.set_linear_rates(linear_rates,allow_nonzero_column_sums=allow_nonzero_column_sums)
        self.set_quadratic_rates(quadratic_rates,allow_nonzero_column_sums=allow_nonzero_column_sums)

        return self

    def set_linear_rates(self,rate_list,allow_nonzero_column_sums=False,reset_rates=True):
        """
        Define the linear transition rates between compartments.

        Parameters
        ==========

        rate_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transitions rates in the following format:

            .. code:: python

                [
                    ( acting_compartment, affected_compartment, rate ),
                    ...
                ]

        allow_nonzero_column_sums : :obj:`bool`, default : False
            Traditionally, epidemiological models preserve the
            total population size. If that's not the case,
            switch off testing for this.

        Example

        reset_rates : bool, default : True
            Whether to reset all linear rates to zero before
            converting those.
        """

        if reset_rates:
            row = []
            col = []
            data = []
            birth_rates = np.zeros((self.N_comp,),dtype=np.float64)
        else:
            linear_rates = self.linear_rates.tocoo()
            row = linear_rates.row.tolist()
            col = linear_rates.col.tolist()
            data = linear_rates.data.tolist()
            birth_rates = self.birth_rates.copy()

        for acting_compartment, affected_compartment, rate in rate_list:

            _t = self.get_compartment_id(affected_compartment)
            if acting_compartment is None:
                birth_rates[_t] += rate
            else:
                _s = self.get_compartment_id(acting_compartment)
                row.append(_t)
                col.append(_s)
                data.append(rate)

        linear_rates = sprs.coo_matrix((data, (row, col)),shape=(self.N_comp, self.N_comp),dtype=np.float64)
        linear_rates = linear_rates.tocsr()

        if not allow_nonzero_column_sums:
            test = linear_rates.sum(axis=0) + birth_rates
            test_sum = test.sum()
            if np.abs(test_sum) > 1e-15:
                warnings.warn("Rates do not sum to zero for each column:" + str(test_sum))

        self.linear_rates = linear_rates
        self.birth_rates = birth_rates

        return self


    def add_transition_processes(self,process_list):
        """
        Define the linear transition processes between compartments.

        Parameters
        ==========

        process_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transitions rates in the following format:

            .. code:: python

                [
                    ( source_compartment, rate, target_compartment ),
                    ...
                ]

        Example
        -------

        For an SEIR model.

        .. code:: python

            epi.add_transition_processes([
                ("E", symptomatic_rate, "I" ),
                ("I", recovery_rate, "R" ),
            ])

        """

        linear_rates = transition_processes_to_rates(process_list)

        return self.set_linear_rates(linear_rates,
                                     reset_rates=False,
                                     allow_nonzero_column_sums=True,
                                     )

    def add_fission_processes(self,process_list):
        """
        Define linear fission processes between compartments.

        Parameters
        ==========

        process_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains fission rates in the following format:

            .. code:: python

                [
                    ("source_compartment", rate, "target_compartment_0", "target_compartment_1" ),
                    ...
                ]

        Example
        -------

        For pure exponential growth of compartment `B`.

        .. code:: python

            epi.add_fission_processes([
                ("B", growth_rate, "B", "B" ),
            ])

        """
        linear_rates = fission_processes_to_rates(process_list)

        return self.set_linear_rates(linear_rates,
                                     reset_rates=False,
                                     allow_nonzero_column_sums=True,
                                     )

    def add_fusion_processes(self,process_list):
        """
        Define fusion processes between compartments.

        Parameters
        ==========

        process_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains fission rates in the following format:

            .. code:: python

                [
                    ("coupling_compartment_0", "coupling_compartment_1", rate, "target_compartment_0" ),
                    ...
                ]

        Example
        -------

        Fusion of reactants "A", and "B" to form "C".

        .. code:: python

            epi.add_fusion_processes([
                ("A", "B", reaction_rate, "C" ),
            ])

        """
        quadratic_rates = fusion_processes_to_rates(process_list)

        return self.set_quadratic_rates(quadratic_rates,
                                        reset_rates=False,
                                        allow_nonzero_column_sums=True
                                        )

    def add_transmission_processes(self,process_list):
        r"""
        A wrapper to define quadratic process rates through
        transmission reaction equations. Note that in stochastic
        network/agent simulations, the transmission
        rate is equal to a rate per link. For the mean-field ODEs,
        the rates provided to this function will just be equal
        to the prefactor of the respective quadratic terms.

        For instance, if you analyze an SIR system and simulate on a network of mean degree :math:`k_0`,
        a basic reproduction number :math:`R_0`, and a recovery rate :math:`\mu`,
        you would define the single link transmission process as

            .. code:: python

                ("I", "S", R_0/k_0 * mu, "I", "I")

        For the mean-field system here, the corresponding reaction equation would read

            .. code:: python

                ("I", "S", R_0 * mu, "I", "I")

        Parameters
        ----------
        process_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transitions rates in the following format:

            .. code:: python

                [
                    ("source_compartment",
                     "target_compartment_initial",
                     rate,
                     "source_compartment",
                     "target_compartment_final",
                     ),
                    ...
                ]

        Example
        -------

        For an SEIR model.

        .. code:: python

            epi.add_transmission_processes([
                ("I", "S", +1, "I", "E" ),
            ])

        """
        quadratic_rates = transmission_processes_to_rates(process_list)

        return self.set_quadratic_rates(quadratic_rates,
                                        reset_rates=False,
                                        allow_nonzero_column_sums=True,
                                        )

    def add_quadratic_rates(self,rate_list,reset_rates=True,allow_nonzero_column_sums=False):
        """
        Add quadratic rates without resetting the existing rate terms.
        See :func:`_tacoma.set_quadratic_rates` for docstring.
        """

        return self.set_quadratic_rates(rate_list,
                                        reset_rates=False,
                                        allow_nonzero_column_sums=allow_nonzero_column_sums,
                                        )

    def add_linear_rates(self,rate_list,reset_rates=True,allow_nonzero_column_sums=False):
        """
        Add linear rates without resetting the existing rate terms.
        See :func:`_tacoma.set_linear_rates` for docstring.
        """

        return self.set_linear_rates(rate_list,
                                     reset_rates=False,
                                     allow_nonzero_column_sums=allow_nonzero_column_sums,
                                     )

    def set_quadratic_rates(self,rate_list,reset_rates=True,allow_nonzero_column_sums=False):
        r"""
        Define the quadratic transition processes between compartments.

        Parameters
        ----------
        rate_list : :obj:`list` of :obj:`tuple`
            A list of tuples that contains transitions rates in the following format:

            .. code:: python

                [
                    (
                      "coupling_compartment_0",
                      "coupling_compartment_1",
                      "affected_compartment",
                      rate
                     ),
                    ...
                ]

        allow_nonzero_column_sums : :obj:`bool`, default : False
            Traditionally, epidemiological models preserve the
            total population size. If that's not the case,
            switch off testing for this.

        Example
        -------

        For an SEIR model.

        .. code:: python

            epi.set_quadratic_rates([
                ("S", "I", "S", -1 ),
                ("S", "I", "E", +1 )
            ])

        Read  as

        "Coupling of *S* and *I* leads to
        a reduction in "S" proportional to :math:`S\times I` and rate -1/time_unit.
        Furthermore, coupling of *S* and *I* leads to
        an increase in "E" proportional to :math:`S\times I` and rate +1/time_unit."
        """

        matrices = [ [[], [], []] for c in self.compartments]
        if reset_rates:
            all_affected = []
        else:
            for iC, _M in enumerate(self.quadratic_rates):
                if _M.count_nonzero() > 0:
                    M = _M.tocoo()
                    matrices[iC][0] = M.row.tolist()
                    matrices[iC][1] = M.col.tolist()
                    matrices[iC][2] = M.data.tolist()
                else:
                    matrices[iC] = [ [], [], [] ]
            all_affected = self.affected_by_quadratic_process if len(self.affected_by_quadratic_process)>0 else []

        for coupling0, coupling1, affected, rate in rate_list:

            c0, c1 = sorted([ self.get_compartment_id(c) for c in [coupling0, coupling1] ])
            a = self.get_compartment_id(affected)

            matrices[a][0].append(c0)
            matrices[a][1].append(c1)
            matrices[a][2].append(rate)
            all_affected.append(a)

        total_sum = 0
        for c in range(self.N_comp):
            row = matrices[c][0]
            col = matrices[c][1]
            data = matrices[c][2]
            if len(row) > 0:
                matrices[c] = sprs.coo_matrix((data,(row, col)),
                                               shape=(self.N_comp, self.N_comp),
                                               dtype=np.float64,
                                              ).tocsr()
                total_sum += matrices[c].sum(axis=0).sum()
            else:
                matrices[c] = sprs.csr_matrix((self.N_comp, self.N_comp),dtype=np.float64)

        if np.abs(total_sum) > 1e-14:
            if not allow_nonzero_column_sums:
                warnings.warn("Rates do not sum to zero. Sum = "+ str(total_sum))

        self.affected_by_quadratic_process = sorted(list(set(all_affected)))
        self.quadratic_rates = matrices

        return self

    def dydt(self,t,y):
        """
        Compute the current momenta of the epidemiological model.

        Parameters
        ----------
        t : :obj:`float`
            Current time
        y : numpy.ndarray
            The first entry is equal to the population size.
            The remaining entries are equal to the current fractions
            of the population of the respective compartments
        """

        ynew = self.linear_rates.dot(y) + self.birth_rates
        if self.correct_for_dynamical_population_size:
            population_size = y.sum()
        else:
            population_size = self.initial_population_size
        for c in self.affected_by_quadratic_process:
            ynew[c] += y.T.dot(self.quadratic_rates[c].dot(y)) / population_size

        return ynew

    def get_numerical_dydt(self):
        """
        Return a function that obtains t and y as an input and returns dydt of this system
        """
        return self.dydt

    def jacobian(self,y0=None):
        """
        Return Jacobian at point ``y0``. Will use ``self.y0`` if argument ``y0`` is ``None``.
        """
        return self.get_transmission_matrix(y0=y0) + self.get_transition_matrix()

    def get_jacobian_leading_eigenvalue(self,y0=None,returntype='complex'):
        """
        Return leading eigenvalue of Jacobian at point ``y0``.
        Will use ``self.y0`` if argument ``y0`` is ``None``.
        Use argument ``returntype='real'`` to only obtain the
        real part of the eigenvalue.
        """
        J = self.jacobian(y0=y0)
        return self._get_leading_eigenvalue(J,returntype)

    def get_transmission_matrix(self,y0=None):
        """
        Return transmission matrix at point ``y0``.
        Will use ``self.y0`` if argument ``y0`` is ``None``.
        """
        if y0 is None:
            y0 = self.y0

        if y0 is None:
            raise ValueError("No initial conditions have been given or set.")

        if self.correct_for_dynamical_population_size:
            raise NotImplementedError("transmission matrices for models with "
                                      "varying population size have not been implemented.")

        rows = []
        for M in self.quadratic_rates:
            row = ((M+M.T).dot(y0)) / self.initial_population_size
            row = row.reshape(1,self.N_comp)
            row = sprs.csr_matrix(row)
            rows.append(row)

        T = sprs.vstack(rows,format='csr')

        return T

    def get_transition_matrix(self):
        """
        Return transition matrix.
        """
        return self.linear_rates

    def get_next_generation_matrix(self,y0=None):
        """
        Return next generation matrix at point y0.
        Will use ``self.y0`` if argument ``y0`` is ``None``.
        """
        T = self.get_transmission_matrix(y0=y0)
        Sigma = self.get_transition_matrix()

        # delete all compartments that contribute to the singularity
        # of the transition matrix
        del_cols = []
        del_rows = []

        n = Sigma.shape[0]

        for i in range(n):
            if np.all(Sigma[i,:].data==0):
                del_rows.append(i)
        for j in range(n):
            if np.all(Sigma[:,j].data==0):
                del_cols.append(j)

        del_comp = set(del_cols) | set(del_rows)
        use_comp = set(range(n)) - del_comp

        use_comp = sorted(list(use_comp))

        # filter our compartments that do not make the matrix singular
        Sigma = (Sigma[use_comp,:])[:,use_comp]
        T = (T[use_comp,:])[:,use_comp]

        # convert Sigma to csc for more efficient inverse algo
        K = -T.dot(sprs.linalg.inv(Sigma.tocsc()))

        return K

    def get_next_generation_matrix_leading_eigenvalue(self,y0=None,returntype='real'):
        """
        Return the leading eigenvalue of the next generation matrix
        at point ``y0``. Will use ``self.y0`` if argument
        ``y0`` is ``None``. When ``y0`` is equal to the initial
        disease-free state, this function will return the
        basic reproduction number :math:`R_0`.

        The function will return only the real part by default.
        Use ``returntype='complex'`` to change this.
        """
        K = self.get_next_generation_matrix(y0=y0)
        return self._get_leading_eigenvalue(K,returntype)

    def _get_leading_eigenvalue(self,M,returntype='complex'):

        if M.shape == (1,1):
            _lambda = M[0,0]
        elif M.shape == (1,):
            _lambda = M[0]
        else:
            # I thought CSC format would be better for solver
            # but it's not, so I uncommented this
            # M_ = M.tocsc()
            M_ = M
            if M_.shape == (2,2):
                lambdas = np.linalg.eig(M_.toarray())[0]
            else:
                lambdas = sprs.linalg.eigs(M_,k=min(2,M_.shape[0]-2),which='LR')[0]
            lambdas = sorted(lambdas, key=lambda x: -np.real(x))
            _lambda = lambdas[0]
        if returntype == 'real':
            _lambda = np.real(_lambda)
        return _lambda


class MatrixSIModel(MatrixEpiModel):
    """
    An SI model derived from :class:`epipack.numeric_matrix_based_epi_models.MatrixEpiModel`.
    """

    def __init__(self, infection_rate, initial_population_size=1.0):

        MatrixEpiModel.__init__(self, list("SI"), initial_population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])


class MatrixSISModel(MatrixEpiModel):
    """
    An SIS model derived from :class:`epipack.numeric_matrix_based_epi_models.MatrixEpiModel`.

    Parameters
    ----------
    R0 : float
        The basic reproduction number. From this, the infection
        rate is computed as ``infection_rate = R0 * recovery_rate``
    recovery_rate : float
        Recovery rate
    population_size : float, default = 1.0
        Number of people in the population.
    """

    def __init__(self, R0, recovery_rate, initial_population_size=1.0):

        infection_rate = R0 * recovery_rate

        MatrixEpiModel.__init__(self, list("SI"), initial_population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])
        self.add_transition_processes([
                ("I", recovery_rate, "S" ),
            ])

class MatrixSIRModel(MatrixEpiModel):
    """
    An SIR model derived from :class:`epipack.numeric_matrix_based_epi_models.MatrixEpiModel`.
    """

    def __init__(self, R0, recovery_rate, initial_population_size=1.0):

        infection_rate = R0 * recovery_rate

        MatrixEpiModel.__init__(self, list("SIR"), initial_population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])
        self.add_transition_processes([
                ("I", recovery_rate, "R"),
            ])

class MatrixSIRSModel(MatrixEpiModel):
    """
    An SIRS model derived from :class:`epipack.numeric_matrix_based_epi_models.MatrixEpiModel`.
    """

    def __init__(self, R0, recovery_rate, waning_immunity_rate, initial_population_size=1.0):

        infection_rate = R0 * recovery_rate

        MatrixEpiModel.__init__(self, list("SIR"), initial_population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "I", +infection_rate),
            ])
        self.add_transition_processes([
                ("I", recovery_rate, "R"),
                ("R", waning_immunity_rate, "S"),
            ])

class MatrixSEIRModel(MatrixEpiModel):
    """
    An SEIR model derived from :class:`epipack.numeric_matrix_based_epi_models.MatrixEpiModel`.
    """

    def __init__(self, R0, recovery_rate, symptomatic_rate, initial_population_size=1.0):

        infection_rate = R0 * recovery_rate

        MatrixEpiModel.__init__(self, list("SEIR"), initial_population_size)

        self.set_quadratic_rates([
                ("S", "I", "S", -infection_rate),
                ("S", "I", "E", +infection_rate),
            ])
        self.add_transition_processes([
                ("E", symptomatic_rate, "I"),
                ("I", recovery_rate, "R"),
            ])


if __name__=="__main__":    # pragma: no cover
    epi = MatrixEpiModel(list("SEIR"))
    print(epi.compartments)
    print()
    epi.add_transition_processes([
            ("E", 1.0, "I"),
            ("I", 1.0, "R"),
            ])
    print(epi.linear_rates)
    epi.set_quadratic_rates([
            ("S", "I", "S", -1.0),
            ("S", "I", "E", +1.0),
            ])
    print()
    for iM, M in enumerate(epi.quadratic_rates):
        print(epi.get_compartment(iM), M)

    import matplotlib.pyplot as pl

    N = 100
    epi = MatrixSISModel(R0=2,recovery_rate=1,initial_population_size=N)
    print(epi.linear_rates)
    epi.set_initial_conditions({'S':0.99*N,'I':0.01*N})
    tt = np.linspace(0,10,100)
    result = epi.integrate(tt,['S','I'])

    pl.plot(tt, result['S'],label='S')
    pl.plot(tt, result['I'],label='I')
    pl.legend()

    pl.show()
