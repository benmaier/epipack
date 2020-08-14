from epipack import StochasticEpiModel, EpiModel, EpiModel
import networkx as nx

R0 = 2.5
Q0 = 0.4
a = 0.20
#QT = 0.8
QT = 0.3
k0 = 100

alpha = 1/5
rho = 1/8
eta = R0 * rho / k0
kappa = Q0/(1-Q0) * rho
chi = 1/2
xi = (1-Q0)/Q0 * chi
omega = 1/14

S, E, I, R, X, T_I, T_E, Q = "S E I R X T_I T_E Q".split(" ")

N = 20000

print("generating network")
edges = [(e[0],e[1],1.0) for e in (nx.fast_gnp_random_graph(N,k0/(N-1))).edges() ] 
agentmodel = StochasticEpiModel([S, E, I, R, X, T_I, T_E, Q], N, edges)
#model = StochasticEpiModel([S, E, I, R, X, T_I, T_E, Q], N, well_mixed_mean_contact_number=k0)

agentmodel.set_node_transition_processes([
        (E, alpha, I),
        (I, rho, R),
        (I, kappa, T_I),
        (T_I, chi, X),
        (T_E, xi, R),
        (T_E, chi, X),
        #(Q, omega, S),
    ])

agentmodel.set_link_transmission_processes([
        (I, S, R0*rho/k0, I, E),
    ])

agentmodel.set_conditional_link_transmission_processes({
        (T_I, "->", X) : {
                ( X, I, QT, X, T_I), 
                ( X, E, QT, X, T_E), 
                #( X, S, QT, X, Q), 
            },
        (T_E, "->", X) : {
                ( X, I, QT, X, T_I), 
                ( X, E, QT, X, T_E), 
                #( X, S, QT, X, Q), 
            }
    })

print(agentmodel.conditional_link_transmission_events)

agentmodel.set_random_initial_conditions({
            S: N - int(0.01*N),
            I: int(0.005*N),
            E: int(0.005*N),
        })

print("simulating")
t, result = agentmodel.simulate(800)

from bfmplot import pl
for c, res in result.items():
    pl.plot(t, res,label=c)


model = EpiModel([S, E, I, R, X, T_I, T_E, Q],N)

model.add_transition_processes([
        (E, alpha, I),
        (I, rho, R),
        (I, kappa, T_I),
        (T_I, chi, X),
        (T_E, xi, R),
        (T_E, chi, X),
        #(Q, omega, S),
    ])

model.add_transmission_processes([
        (I, S, R0*rho, I, E),
        (T_I, I, chi*QT*k0 , T_I, T_I),
        (T_I, E, chi*QT*k0 , T_I, T_E),
        #(T_I, S, chi*QT*k0 , T_I, Q),
        (T_E, I, chi*QT*k0 , T_E, T_I),        
        (T_E, E, chi*QT*k0 , T_E, T_E),
        #(T_E, S, chi*QT*k0 , T_E, Q),
    ])

model.set_initial_conditions({
            S: int(N*(1 - 0.01)),
            I: int(N*(0.005)),
            E: int(N*(0.005)),
        })

print(model.y0, model.t0)
ODE_result = model.integrate(t)


for c, res in ODE_result.items():
    pl.plot(t, res,label=c,lw=2,alpha=0.3)

print(model.y0, model.t0)

tt, MF_result = model.simulate(t[-1])

for c, res in MF_result.items():
    pl.plot(tt, res,'--',label=c,lw=2,alpha=0.3)

#pl.plot(t,label=c)
pl.legend()
pl.ylim([0,N])
pl.show()
    
