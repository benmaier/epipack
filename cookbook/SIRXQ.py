import numpy as np
from epipack.DeterministicEpiModels import EpiModel




def old_mean_field_SIRX_tracing(kw):

    R0 = kw['R0']
    Q0 = kw['Q0']
    waning_quarantine_rate = omega = kw['waning_quarantine_rate']
    recovery_rate = rho = kw['recovery_rate']
    quarantine_rate = kappa = rho * Q0 / (1-Q0)
    infection_rate = eta = R0 * (rho)
    app_participation_ratio = theta = kw['app_participation_ratio']
    k0 = kw['k0']


    model = EpiModel(list("SIXRQ"))
    model.add_transition_processes([
           ( "Q", waning_quarantine_rate,"S"), 
           ( "I", recovery_rate,"R"),
           ( "I", kappa,"X"),
        ])

    model.set_quadratic_rates([
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

    result = model.integrate(t)

    return t, result

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
    model.add_transition_processes([
           ( "Q", waning_quarantine_rate,"S"), 
           ( "I", recovery_rate,"R"),
           ( "I", kappa,"X"),
        ])

    model.add_transmission_processes([
        ( "S", "I", eta, "I", "I"),
        ( "I", "I", theta**2*kappa*k0, "I", "X"),
        ( "S", "I", theta**2*kappa*k0, "I", "Q"),
        ])

    #for C, M in zip(model.compartments, model.quadratic_rates):
    #    print(C, M)

    I0 = kw["I0"]
    model.set_initial_conditions({"S":1-I0,"I":I0})

    t = kw["t"]

    result = model.integrate(t)

    return t, result
