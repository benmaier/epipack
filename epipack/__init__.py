# -*- coding: utf-8 -*-
"""
Build, simulate, and analyze epidemiological models.
"""

from .metadata import *

from .numeric_matrix_based_epi_models import (
        NumericMatrixBasedEpiModel,
        NumericMatrixBasedSIModel,
        NumericMatrixBasedSISModel,
        NumericMatrixBasedSIRModel,
        NumericMatrixBasedSIRSModel,
        NumericMatrixBasedSEIRModel,
        )

MatrixEpiModel = NumericMatrixBasedEpiModel
MatrixBasedSIModel = NumericMatrixBasedSIModel,
MatrixBasedSISModel = NumericMatrixBasedSISModel,
MatrixBasedSIRModel = NumericMatrixBasedSIRModel,
MatrixBasedSIRSModel = NumericMatrixBasedSIRSModel,
MatrixBasedSEIRModel = NumericMatrixBasedSEIRModel,

from .numeric_epi_models import (
        NumericEpiModel,
        NumericSIModel,
        NumericSISModel,
        NumericSIRModel,
        NumericSIRSModel,
        NumericSEIRModel,
        )

EpiModel = NumericEpiModel
SIModel = NumericSIModel
SISModel = NumericSISModel
SIRModel = NumericSIRModel
SIRSModel = NumericSIRSModel
SEIRModel = NumericSEIRModel

from .stochastic_epi_models import (
        StochasticEpiModel,
        StochasticSIRModel,
        StochasticSISModel,
        )

from .symbolic_epi_models import (
        SymbolicMatrixBasedEpiModel,
        SymbolicEpiModel,
        SymbolicSIModel,
        SymbolicSISModel,
        SymbolicSIRModel,
        SymbolicSIRSModel,
        )

from .network_models import (
        get_2D_lattice_links,
        )
