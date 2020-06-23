import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
#from lmfit import minimize, Parameters


# set equation of motion for SIRX dynamics
class Metapopulation_model:
    def dxdt(self,t,y,R_linear,SI_parameters,Mobility_input,Fraction_population_in_mobile_compartments ,Population_rescale_system,Population_change_matrix,data_penetration):
        # t is time
        # y is array of variables (dimension n x 1). y[0+c] is S for compartment c, y[1+c] is I for compartment c.
        # R is linear rate matrix (dimension n x p), has 1 on entry i,j if variable i is linearly dependent on variable j
        # SI_parameters is an array of parameters that gets multiplied on the SI-terms. 

        # Calculate basic numbers to input in arrays below
        tot_variables = np.shape(R_linear)[0]
        N_populations = int(len(SI_parameters))
        variables_per_population = int(tot_variables/N_populations)


        # Define variables
        dy = np.zeros(tot_variables)

        # Insert nonlinear terms
        for population in range (N_populations) :
            dy[population*variables_per_population + 1] = -SI_parameters[population]*y[population*variables_per_population+1]*y[population*variables_per_population+2]
            dy[population*variables_per_population + 2] = +SI_parameters[population]*y[population*variables_per_population+1]*y[population*variables_per_population+2]

        # Add linear terms
        dy += R_linear.dot(y)


        # Add mobility
        tiny_number_to_ensure_no_zerodivision = 10**(-10)
        fraction_mobile_reciprocal = np.reciprocal(np.matmul(Fraction_population_in_mobile_compartments,y)+tiny_number_to_ensure_no_zerodivision)

        dy += 1/(data_penetration)*np.reciprocal(np.matmul(Population_change_matrix,y))*(np.matmul(Mobility_input,np.multiply(y,fraction_mobile_reciprocal))-np.matmul(np.transpose(np.multiply(Mobility_input,np.multiply(y,fraction_mobile_reciprocal))),np.ones(len(y))))        
        # Take population change into account
        dy -= 1/(data_penetration)*np.reciprocal(np.matmul(Population_change_matrix,y))*(np.matmul(Population_rescale_system,np.ones(len(y)))-np.matmul(np.transpose(Population_rescale_system),np.ones(len(y))))*y
        return dy


    def Solve_metapopulation_model(self,t,y0,R_linear,SI_parameters,Mobility_input,Fraction_population_in_mobile_compartments,Population_rescale_system,Population_change_matrix,data_penetration):
        
        N_populations = len(SI_parameters)
        tot_variables = np.shape(R_linear)[0]
        variables_per_population = int(tot_variables/N_populations)

        for population in range (N_populations) :
            if (np.around(sum(y0[variables_per_population*population+1:variables_per_population*(population+1)]),decimals=10) != 1) :
                print("ERROR: Sum of initial conditions must sum to 1 for every population")
                exit()
        

        t0 = t[0]

        t = t[1:]

        r = ode(self.dxdt)

        # Runge-Kutta with step size control
        r.set_integrator('dopri5')

        # set initial values
        r.set_initial_value(y0,t0)

        # set transmission rate and recovery rate
        r.set_f_params(R_linear,SI_parameters,Mobility_input,Fraction_population_in_mobile_compartments ,Population_rescale_system,Population_change_matrix,data_penetration)

        result = np.zeros((tot_variables,len(t)+1))
        result[:,0] = y0

        # loop through all demanded time points
        for it, t_ in enumerate(t):

            # get result of ODE integration
            y = r.integrate(t_)

            # write result to result vector
            result[:,it+1] = y

        return result



def make_mobility_matrix(mobility_between_populations,variables_affected_by_mobility) :
    populations = len(mobility_between_populations)
    variables = len(variables_affected_by_mobility)    
    
    Res = np.zeros((populations*variables,populations*variables))

    for i in range (populations) :
        Res[i*variables:i*variables+variables,i*variables:i*variables+variables] = mobility_between_populations[i,i]*np.eye(variables) 
        for j in range (i+1,populations) :
            Res[i*variables:i*variables+variables,j*variables:(j+1)*variables] = mobility_between_populations[i,j]*variables_affected_by_mobility
            Res[j*variables:(j+1)*variables,i*variables:i*variables+variables] = mobility_between_populations[j,i]*variables_affected_by_mobility


    return Res
def make_fraction_population_in_mobile_compartments_matrix(Mobility_block,N_populations) :
    M = np.zeros((len(Mobility_block)*N_populations,len(Mobility_block)*N_populations))
    Mobility_arr = np.diag(Mobility_block)
    population_number = 0
    number_of_variables = len(Mobility_arr)
    for i in range (len(M)) :
        M[i,population_number*number_of_variables:(population_number+1)*number_of_variables] = Mobility_arr
        if ((i+1)//number_of_variables == (i+1)/number_of_variables) :
            population_number += 1
    return M

def make_population_change_matrix(variables_per_population,N_populations) :
    Matrix = np.zeros((variables_per_population*N_populations,variables_per_population*N_populations))
    for population in range (N_populations) :
        Matrix[population*variables_per_population:(population+1)*variables_per_population,population*variables_per_population] = ([-1]+[1]*(variables_per_population-1))
    return Matrix
def import_parameters_from_file(SI_parameters,parameters,filename) :
    f = open(filename,'r')
    f.readline()
    parameters = np.array([])
    SI_parameters = np.array([])

    for line in f :
        line.strip()
        columns = line.split('\t')
        SI_parameters = np.concatenate((SI_parameters,np.array([columns[0]],float)))
        parameters = np.concatenate((parameters,np.array(columns[1:],float)))
    f.close()
    return SI_parameters,parameters

def import_initialconditions_from_file(y0,filename,filename_populationsize) :
    # First import populations sizes
    population_sizes = []
    f = open(filename_populationsize,'r')
    f.readline()
    for line in f :
        line.strip()
        columns = line.split('\n')
        population_sizes.append(float(columns[0]))
    f.close()

    f = open(filename,'r')
    f.readline()
    line_num = -1
    for line in f :
        line_num +=1
        line.strip()
        columns = line.split('\t')
        y0 = np.concatenate((y0,np.array([population_sizes[line_num]]+columns[:],float)))
    f.close()
    return y0


def import_mobility_from_file(mobility_array,filename) :
    f = open(filename,'r')
    f.readline()
    for line in f :
        line.strip()
        columns = line.split('\t')
        mobility_array[int(columns[0]),int(columns[1])] = float(columns[2])
    f.close()
    return mobility_array

def give_new_initialconditions(y0,solution,variables_per_population) :
    
    last_variable_y0 = 1.
    for variable in range (len(solution)) :
        if ((variable+1)//variables_per_population ==(variable+1)/variables_per_population ) :
            y0.append(last_variable_y0)
            last_variable_y0 = 1.


        elif (variable//variables_per_population != variable/variables_per_population) :    
            y0.append(solution[variable][-1])
            last_variable_y0 -=solution[variable][-1]

        else : 
            y0.append(solution[variable][-1])
    return y0#np.around(y0,decimals=10)

# -----------------
# Define everything 
# -----------------

# Opportunity to run code with predefined parameters. No need to define files.
run_toy_example = True

# Define ODE model
# (linear rates:)
def add_metapopulation(parameters) :
    
    N = 1
    S = 1
    I = 1
    #E = 1
    R = 1
    
    R_linear = np.array([
        # SEIR system:
        #[0*S, 0*E, 0*I,0*R],
        #[0*S,-parameters[0]*E,0*I,0*R],
        #[0*S,parameters[0]*E,-parameters[1]*I,0*R],
        #[0*S,0*E,parameters[1]*I,0*R]

        # SIR system:
        [0*N,0*S, 0*I,0*R],
        [0*N,0*S,0*I,0*R],        
        [0*N,0*S,-parameters[0]*I,0*R],
        [0*N,0*S,parameters[0]*I,0*R]
        
    ])

    return R_linear

# Which compartments are not affected by mobility? (e.g. quarantined people)
Variables_not_mobile = [0,2] # Mobility does not count as mobile
  



if (run_toy_example == False) :

    # Define where model inputs for different time intervals are saved

    # data folder
    data_folder = 'data/example_data/'
    # parameters
    parameter_filename = data_folder+'parameters_day%i.csv'
    # initial conditions
    initialconditions_filename = data_folder+'initialconditions.csv'
    # mobility network
    mobility_filename = data_folder+'mobility_day%i.csv'
    # populations size
    populationsize_filename = data_folder+'population_day0.csv'

    # Define how many time intervals (the code assumes that number of interval can be used to identify the correct input files above)
    time_intervals = np.arange(0,11,1)
    # How long does each time interval last
    length_of_timeintervals = np.ones(len(time_intervals))

    # Define mobility data penetration rate
    data_penetration = 1.


    # -------
    # Now the code takes care of the rest.
    # -------

    # Define arrays containing an array of inputs for each time interval
    SI_parameters_all = [[]]*len(time_intervals)
    parameters_all = [[]]*len(time_intervals)
    Mobility_data_all = [[]]*len(time_intervals)
    #population_size_all = [[]]*len(time_intervals)

    # And define an
    for time_interval in time_intervals :


        # Import parameters
        SI_parameters_all[time_interval],parameters_all[time_interval]=import_parameters_from_file(SI_parameters_all[time_interval],parameters_all[time_interval],parameter_filename%time_interval)
        # Import Mobility data
        Mobility_data_all[time_interval] = import_mobility_from_file(np.zeros((len(SI_parameters_all[time_interval]),len(SI_parameters_all[time_interval]))),mobility_filename%time_interval)

    # Define time and initial conditions
    y0 = import_initialconditions_from_file(np.array([],float),initialconditions_filename,populationsize_filename)
    #t = np.arange(0,10,0.01)

    # Define populations
    N_populations = len(SI_parameters_all[0])





# Below : Run code with toy parameters.
else  :

    # Define parameters
    parameters_all = [np.array([.2,.4,.6])]
    SI_parameters_all = [np.array([.7,.7,.7])]

    # Define time and initial conditions
    y0 = [80.,.00,0.0,1.0,90.,.99,0.01,0.0,100.,1.00,0.00,0.0]
    #t = np.arange(0,10,0.01)
    time_intervals = [0]
    length_of_timeintervals = [10]

    # Define Mobility data
    Mobility_data_all = [np.array([[5,1,0],[0,6,0],[0,0,7]])]
    data_penetration = 1.

# -----------------
# Now solve
# -----------------


model = Metapopulation_model()


interval_number = -1
current_time = 0
for time_interval in time_intervals :
    interval_number +=1
    t = np.arange(current_time,current_time+length_of_timeintervals[interval_number],0.01)
    current_time+=length_of_timeintervals[interval_number]


    SI_parameters = SI_parameters_all[time_interval]
    parameters = parameters_all[time_interval]
    Mobility_data = Mobility_data_all[time_interval]

    N_populations = len(SI_parameters)
    parameters_per_population = int(len(parameters)/N_populations)

    # Make matrix of linear rates
    R_linear = add_metapopulation(parameters[:parameters_per_population])
    for population in range (1,N_populations) :
        parameter_block = add_metapopulation(parameters[parameters_per_population*population:parameters_per_population*(population+1)])

        R_linear = np.block([[R_linear,np.zeros((np.shape(R_linear)[0],np.shape(parameter_block)[1]))],
                            [np.zeros((np.shape(parameter_block)[0],np.shape(R_linear)[1])),parameter_block]])

    variables_per_population = int(len(R_linear)/N_populations)

    Mobility_block = np.eye(variables_per_population)
    # Mobility block [i,i] = 1 if people in compartment i can move. else = 0
    for i in Variables_not_mobile :
        Mobility_block[i,i] = 0 

    Mobility_input_system = make_mobility_matrix(Mobility_data,Mobility_block)
    Fraction_population_in_mobile_compartments = make_fraction_population_in_mobile_compartments_matrix(Mobility_block,N_populations)
    Population_rescale_system = make_mobility_matrix(Mobility_data,np.eye(len(Mobility_block)))

    # Make population change matrix for input..
    Population_change_matrix = make_population_change_matrix(variables_per_population,N_populations)



    # Solve system
    solution = model.Solve_metapopulation_model(t,y0,R_linear,SI_parameters,Mobility_input_system,Fraction_population_in_mobile_compartments ,Population_rescale_system,Population_change_matrix,data_penetration)
    
    # Save results for this integration in a concatenated array that can be plotted when we are done
    if (current_time == length_of_timeintervals[0]) :
        time_concatenated = np.array([],float)
        solution_concatenated = [np.array([],float)]*(variables_per_population*N_populations)
    time_concatenated = np.concatenate((time_concatenated,t))
    solution_concatenated=np.concatenate((solution_concatenated,solution),axis=1)
    
    # Initial conditions for next simulation (must ensure that they sum to 1 in each population)
    y0 = give_new_initialconditions([],solution,variables_per_population)
    



# Plot
colors = ['C0','C1','C2','C3']
markers = ['-','--','-.','dotted']
plt.figure(1)
for pop in range (N_populations) :
    for i in range (1,variables_per_population):
        plt.plot(time_concatenated,solution_concatenated[i+pop*variables_per_population],linestyle=markers[i-1],color=colors[pop],alpha=0.8)

plt.xlabel('time')
plt.ylabel('fraction')      
plt.legend(handles=[Line2D([0],[0],linestyle=markers[i],color='k', label='Compartment %i'%i) for i in range (variables_per_population)])

plt.figure(2)
for i in range (len(y0)) :
    if (i//variables_per_population == i/variables_per_population) :
        plt.plot(time_concatenated,solution_concatenated[i],linestyle=markers[i-(i//variables_per_population)*variables_per_population],color=colors[i//variables_per_population],alpha=0.8)

plt.xlabel('time')
plt.ylabel('N(t)')      
#plt.legend(handles=

plt.show()
