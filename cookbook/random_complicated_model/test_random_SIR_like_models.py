import numpy as np
import epipack as epk

np.random.seed(2334)

Nage = 10
CI = np.random.rand(Nage,Nage)
CA = np.random.rand(Nage,Nage)
Ns = np.random.rand(Nage)
rate_E = np.random.rand(Nage)
rate_A = np.random.rand(Nage)
rate_I = np.random.rand(Nage)

CI = np.ones((Nage, Nage))
CA = np.ones((Nage, Nage))
#CI = np.eye(Nage)
#CA = np.eye(Nage)
Ns = np.ones(Nage)/Nage
rate_E = np.ones(Nage)
rate_A = np.ones(Nage)
rate_I = np.ones(Nage)

_i0 = 1e-9

def model_SIR(t, R0, i00=_i0):
    s0 = 1-_i0
    recovery_rate = 1/3
    model = epk.SIRModel(R0*recovery_rate, recovery_rate=recovery_rate)\
               .set_initial_conditions({'S':s0, 'I': 1-s0})
    res = model.integrate(t)
    return res['R']

def get_relative_R_timeseries(res):
    N = Ns.sum()
    tot = 0
    for i in range(Nage):
        tot += res[f'R{i}']
    return tot / N

def get_model_raw(R0=1,infection_rate_norm=1,i00=_i0):
    norm = infection_rate_norm
    s00 = 1-i00

    S = [f'S{i}' for i in range(Nage)]
    E = [f'E{i}' for i in range(Nage)]
    A = [f'A{i}' for i in range(Nage)]
    I = [f'I{i}' for i in range(Nage)]
    R = [f'R{i}' for i in range(Nage)]

    compartments = S + E + A + I + R

    model = epk.MatrixEpiModel(compartments)

    for i in range(Nage):
        for j in range(Nage):
            model.add_transmission_processes([
                    (f'I{j}', f'S{i}', R0*CI[i,j]/Ns[j]/norm, f'I{j}', f'E{i}'),
                    (f'A{j}', f'S{i}', R0*CA[i,j]/Ns[j]/norm, f'A{j}', f'E{i}'),
                ])
    for i in range(Nage):
        model.add_transition_processes([
                (f'E{i}', rate_E[i], f'A{i}'),
                (f'A{i}', rate_A[i], f'I{i}'),
                (f'I{i}', rate_I[i], f'R{i}'),
            ])

    initial_conditions = {}
    for i in range(Nage):
        initial_conditions.update({ f'S{i}': Ns[i]*s00, f'I{i}': Ns[i]*i00 })

    model.set_initial_conditions(initial_conditions)

    return model


def get_model(R0,i00=_i0):
    model = get_model_raw(i00=0) # setup in disease-free state to compute R0
    model_R0_initial = model.get_next_generation_matrix_leading_eigenvalue()
    print(f"{model_R0_initial=}")
    norm = model_R0_initial
    model = get_model_raw(R0=R0,infection_rate_norm=norm,i00=i00)
    model_R0_final = model.get_next_generation_matrix_leading_eigenvalue()
    print(f"{model_R0_final=}")
    return model


if __name__ == "__main__":
    import matplotlib.pyplot as pl
    R0 = 1.2
    model = get_model(R0)
    t = np.linspace(0,1000,1001)
    res = model.integrate(t)
    R = get_relative_R_timeseries(res)
    pl.plot(t, R)

    R = model_SIR(t,R0)
    pl.plot(t, R)
    pl.ylim(0,1)
    pl.show()
