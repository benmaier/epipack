from epipack import StochasticEpiModel, DeterministicEpiModel
import networkx as nx

R0 = 2.5
Q0 = 0.4
a = 0.20
#QT = 0.8
QT = 1
k0 = 100

alpha = 1/5
rho = 1/8
eta = R0 * rho / k0
kappa = Q0/(1-Q0) * rho
chi = 1/2
xi = (1-Q0)/Q0 * chi
omega = 1/14

S, E, I, R, X, T_I, T_E, Q = "S E I R X T_I T_E Q".split(" ")

N = 100000

print("generating network")
edges = [(e[0],e[1],1.0) for e in (nx.fast_gnp_random_graph(N,k0/(N-1))).edges() ] 
model = StochasticEpiModel([S, E, I, R, X, T_I, T_E, Q], N, edges)
#model = StochasticEpiModel([S, E, I, R, X, T_I, T_E, Q], N, well_mixed_mean_contact_number=k0)

model.set_node_transition_processes([
        (E, alpha, I),
        (I, rho, R),
        (I, kappa, T_I),
        (T_I, chi, X),
        #(T_E, xi, R),
        #(T_E, chi, X),
    ])

model.set_link_transmission_processes([
        (I, S, R0*rho/k0, I, E),
    ])

model.set_conditional_link_transmission_processes({
        (T_I, "->", X) : {
                ( X, I, QT, X, T_I), 
                ( X, E, QT, X, T_E), 
                ( X, S, QT, X, Q), 
            },
        (T_E, "->", X) : {
                ( X, I, QT, X, T_I), 
                ( X, E, QT, X, T_E), 
                ( X, S, QT, X, Q), 
            }
    })

print(model.conditional_link_transmission_events)

model.set_random_initial_conditions({
            S: N - int(0.01*N),
            I: int(0.005*N),
            E: int(0.005*N),
        })

print("simulating")
t, result = model.simulate(800)

from bfmplot import pl
for c, res in result.items():
    pl.plot(t, res,label=c)


model = DeterministicEpiModel([S, E, I, R, X, T_I, T_E, Q])

model.add_transition_processes([
        (E, alpha, I),
        (I, rho, R),
        (I, kappa, T_I),
        (T_I, chi, X),
        (Q, omega, S),
    ])

model.add_transmission_processes([
        (I, S, R0*rho, I, E),
        (T_I, I, chi*QT*k0 , T_I, T_I),
        (T_I, E, chi*QT*k0 , T_I, T_E),
    ])

model.set_initial_conditions({
            S: 1 - 0.01,
            I: 0.005,
            E: 0.005,
        })

ODE_result = model.integrate(t)


for c, res in ODE_result.items():
    pl.plot(t, res*N,label=c,lw=2,alpha=0.3)


#pl.plot(t,label=c)
pl.legend()
pl.ylim([0,N])
pl.show()
    
