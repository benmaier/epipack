import epipack as epk

N = 1000
S, I, R = list("SIR")
model = epk.EpiModel([S,I,R],
                     initial_population_size=N)

alpha = 1/2
beta = 1/4
gamma = 1
I0 = 100

model.set_processes([
        (S, I, alpha, I, I),
        (I, beta, R),
        (None, gamma, S),
    ])
