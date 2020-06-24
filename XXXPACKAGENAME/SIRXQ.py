import numpy as np
from metapop import EpiModel




def mean_field_SIRX_tracing(kw):

    R0 = kw['R0']
    Q0 = kw['Q0']
    waning_quarantine_rate = omega = kw['waning_quarantine_rate']
    recovery_rate = rho = kw['recovery_rate']
    quarantine_rate = kappa = rho * Q0 / (1-Q0)
    infection_rate = eta = R0 * (rho)
    app_participation_ratio = theta = kw['app_participation_ratio']
    k0 = kw['k0']


    model = EpiModel(list("SIXRQ"))
    model.set_linear_processes([
           ( "Q", "S", waning_quarantine_rate), 
           ( "I", "R", recovery_rate),
           ( "I", "X", kappa),
        ])

    model.set_quadratic_processes([
            ( "S", "I", "S", -eta),
            ( "S", "I", "I", +eta),
            ( "I", "I", "X", +theta**2*kappa*k0),
            ( "I", "I", "I", -theta**2*kappa*k0),
            ( "S", "I", "Q", +theta**2*kappa*k0),
            ( "S", "I", "S", -theta**2*kappa*k0),
        ],allow_nonzero_column_sums=True)


    I0 = kw["I0"]
    model.set_initial_conditions({"S":1-I0,"I":I0})

    t = kw["t"]

    result = model.get_result_dict(t)

    return t, result
