# -*- coding: utf-8 -*-
"""
Build, simulate, and analyze epidemiological models.
"""

from .metadata import *

from .numeric_matrix_epi_models import (
        MatrixEpiModel,
        MatrixSIModel,
        MatrixSISModel,
        MatrixSIRModel,
        MatrixSIRSModel,
        MatrixSEIRModel,
    )

from .numeric_epi_models import (
        EpiModel,
        SIModel,
        SISModel,
        SIRModel,
        SIRSModel,
        SEIRModel,
    )

from .stochastic_epi_models import (
        StochasticEpiModel,
        StochasticSIModel,
        StochasticSIRModel,
        StochasticSISModel,
    )

from .symbolic_epi_models import (
        SymbolicEpiModel,
        SymbolicSIModel,
        SymbolicSISModel,
        SymbolicSIRModel,
        SymbolicSIRSModel,
        get_temporal_interpolation,
    )

from .symbolic_matrix_epi_models import (
        SymbolicMatrixEpiModel,
        SymbolicMatrixSISModel,
        SymbolicMatrixSIRModel,
        SymbolicMatrixSIRSModel,
        SymbolicMatrixSIModel,
    )

from .network_models import (
        get_2D_lattice_links,
    )
