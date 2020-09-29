import tacoma as tc
import epipack as epk

# load DTU data as edge_changes
dtu = tc.load_json_taco('~/.tacoma/dtu_1_weeks.taco')

# convert to edge_lists
dtu = tc.convert(dtu)

k = tc.time_average(*tc.mean_degree(dtu),tmax=dtu.tmax)

R0 = 3
recovery_rate = 1/(24*3600)
infection_rate = R0 / k * recovery_rate








tmax = 7*24*3600
SIS = tc.SIS(dtu.N, dtu.tmax, infection_rate, recovery_rate)
