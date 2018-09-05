import Dirichlet_Uncertainty_set
import Gaussian_Uncertainty_Set
from craam import crobust
import Utils
import Plot
import numpy as np
import tqdm
import pickle
import datetime

### Run Bayesian Experiments
if __name__ == "__main__":
    # number of assumes states in the MDP
    num_next_states = 5
    # number of sampling steps
    num_iterations = 10 #30
    # the desired confidence level
    confidence_level = 0.95
    # number of runs
    runs = 30 #30
    # step size in the number of samples
    sample_step = 10
    
    value_function = np.random.randint(10, size=num_next_states)
    #define reward for the simple mdp with 1 state, 1 action, num_next_states number of next states with uniform transition probability
    reward = np.arange(num_next_states, dtype=np.double)
    
    bayes_results = []
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)
    
    for pos, i in enumerate(tqdm.tqdm(sample_steps)):
        bayes_results.append(evaluate_bayesian_uncertainty(i, num_next_states, reward, confidence_level, runs, reward))

with open('dumped_results/Bayes_result_state_'+str(num_next_states)+"_iteration_"+str(num_iterations)\
        +"_conf_"+str(confidence_level)+"_runs_"+str(runs)+"_step_"+str(sample_step),'wb') as fp:
    
    pickle.dump([bayes_results, sample_steps], fp)

###
import pickle

f = open('dumped_results/Bayes_result_state_5_iteration_10_conf_0.95_runs_200_step_10', 'rb')   # 'r' for reading; can be omitted
br = pickle.load(f)         # load file content as mydict
bayes_results, sample_steps = br[0], br[1]
f.close()
#with open ('outfile', 'rb') as fp:
#    itemlist = pickle.load(fp)

### Plot Bayesian Results
if __name__ == "__main__":
    compare_methods = [Methods.BAYES.value, Methods.CENTROID.value, Methods.HOEFF.value, Methods.HOEFFTIGHT.value,  Methods.INCR_ADD_V.value]
    plot_returns(bayes_results, sample_steps, compare_methods, "Dirichlet_return_single_state.pdf",1)
    plot_thresholds(bayes_results, sample_steps, compare_methods, "Dirichlet_threshold_single_state.pdf",1)
    plot_violations(bayes_results, sample_steps, compare_methods, "Dirichlet_violations_single_state.pdf")

###Load & Plot
if __name__ == "__main__":
    import pickle
    
    f = open('dumped_results/Bayes_result_state_5_iteration_10_conf_0.95_runs_200_step_10', 'rb')   # 'r' for reading; can be omitted
    bayes_results, sample_steps = pickle.load(f)         # load file content as mydict
    f.close()                
    
    num_iterations = 10
    sample_step = 10
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)
    
    compare_methods = [Methods.BAYES.value, Methods.CENTROID.value, Methods.HOEFF.value, Methods.HOEFFTIGHT.value,  Methods.INCR_ADD_V.value]
    plot_returns(bayes_results, sample_steps, compare_methods, "Dirichlet_return_single_state.pdf",runs)
    plot_thresholds(bayes_results, sample_steps, compare_methods, "Dirichlet_threshold_single_state.pdf",runs)
    plot_violations(bayes_results, sample_steps, compare_methods, "Dirichlet_violations_single_state.pdf")


### Run Gaussian Experiments
if __name__ == "__main__":
    now = datetime.datetime.now()
    # number of sampling steps
    num_iterations = 10
    # the desired confidence level
    confidence_level = 0.95
    # number of runs
    runs = 50
    min_demand, max_demand = 0, 30
    demand_mean_prior_mean, demand_mean_prior_std, true_demand_std = 13, 3, 4
    
    sample_step = 50
    
    gauss_results = []
    value_function = np.arange(max_demand-min_demand+1) #np.random.randint(10, size=max_demand-min_demand+1)
    #np.random.uniform(low=0, high=10, size=(max_demand-min_demand+1))
    print("value_function",value_function)
    
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)

    #for pos, i in enumerate(tqdm.tqdm(sample_steps)):
        #gauss_results.append(evaluate_gaussian_uncertainty(i, confidence_level, runs, value_function, value_function, [value_function], min_demand, max_demand))

    for pos, i in enumerate(tqdm.tqdm(sample_steps)):
        gauss_results.append(evaluate_gaussian_uncertainty(i, confidence_level, runs, value_function, min_demand, max_demand,\
        demand_mean_prior_mean, demand_mean_prior_std, true_demand_std))

### Only test KnownV
if __name__ == "__main__":
    now = datetime.datetime.now()
    # number of sampling steps
    num_iterations = 10
    # the desired confidence level
    confidence_level = 0.95
    # number of runs
    runs = 50
    min_demand, max_demand = 0, 30
    demand_mean_prior_mean, demand_mean_prior_std, true_demand_std = 15, 3, 4

    sample_step = 20

    gauss_results_knownV = []
    value_function = np.random.randint(10, size=max_demand-min_demand+1) #np.arange(max_demand-min_demand+1)
    #np.random.uniform(low=0, high=10, size=(max_demand-min_demand+1))
    print("value_function",value_function)
    
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)

    #for pos, i in enumerate(tqdm.tqdm(sample_steps)):
    gauss_results_knownV.append(evaluate_gaussian_knownV(50, confidence_level, runs, value_function, min_demand, max_demand,\
        demand_mean_prior_mean, demand_mean_prior_std, true_demand_std))

###
#for i in range(len(gauss_results_knownV)):
#    print(np.mean(gauss_results_knownV[i][0][3])*len(gauss_results_knownV[i][0][3]))

import matplotlib.pyplot as plt
plt.plot(gauss_results_knownV, color='black', label="Prior mean")
plt.legend(loc='best', fancybox=True, framealpha=0.3)
plt.show()

###
with open('dumped_results/gauss_results'+"_iteration_"+str(num_iterations)\
        +"_conf_"+str(confidence_level)+"_runs_"+str(runs)+"_step_"+str(sample_step)+str(now),'wb') as fp:
    
    pickle.dump([gauss_results, sample_steps], fp)

### Plot Gaussian Results
    
if __name__ == "__main__":
    compare_methods = [Methods.BAYES, Methods.CENTROID, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.INCR_ADD_V]
    plot_returns_ext(gauss_results, sample_steps, compare_methods, "gaussian_return_single_state_exp.pdf", runs)
    #plot_thresholds(gauss_results, sample_steps, compare_methods,"gaussian_threshold_single_state_exp.pdf", runs)
    plot_violations_ext(gauss_results, sample_steps, compare_methods ,"gaussian_violation_single_state_exp.pdf")
    
###Load & Plot
if __name__ == "__main__":
    compare_methods = [Methods.BAYES.value, Methods.CENTROID.value, Methods.HOEFF.value, Methods.HOEFFTIGHT.value, Methods.INCR_ADD_V.value]
    plot_returns(gauss_results, sample_steps, compare_methods, "gaussian_return_single_state.pdf", runs)
    plot_thresholds(gauss_results, sample_steps, compare_methods,"gaussian_threshold_single_state.pdf", runs)
    plot_violations(gauss_results, sample_steps, compare_methods ,"gaussian_violation_single_state.pdf")

"""
    import pickle
    
    f = open('dumped_results/gauss_results_20_0.95_200_10_05.09.18', 'rb')   # 'r' for reading; can be omitted
    gauss_results = pickle.load(f)         # load file content as mydict
    f.close()                
    
    num_iterations = 20
    sample_step = 20
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)
"""

### Invasive Species Simulation
if __name__ == "__main__":
    initial_population, carrying_capacity, mean_growth_rate, std_growth_rate, std_observation, \
    beta_1, beta_2, n_hat, threshold_control, prob_control, seed = 30, 1000, 1.02, 0.02, 10, 0.001, -0.0000021, 300, 0, 0.5, 3
    
    species_simulator = crobust.SimulatorSpecies(initial_population, carrying_capacity, mean_growth_rate, std_growth_rate, \
                                                    std_observation, beta_1, beta_2, n_hat, threshold_control, prob_control, seed)
    samples = species_simulator.simulate_species(horizon, runs)
    
    states_from = samples.get_states_from()
    actions = samples.get_actions()
    states_to = samples.get_states_to()
    
    for i in range(len(states_from)):
        print(states_from[i], actions[i], states_to[i])

### Customized plotting methods
# The method takes the data as the first parameter & name of the methods to compare & the figure name
def plot_returns_ext(results_dir, sample_steps, compare_methods, figure_name="Return_compare.pdf",runs=1):
    indices = {}
    methods = [r[0] for r in results_dir[0]]
    for method_index, method_name in enumerate(methods):
        indices[method_name.value] = method_index

    method_names = [Methods.CENTROID, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.BAYES, Methods.INCR_ADD_V]
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')
    
    for method_index, method_name in enumerate(method_names):
        
        method_index = indices[method_name.value]
        if method_name not in compare_methods:
            continue
        print("method index", method_index, "method name", method_name)
        method_label = method_name.value
        if method_name.value == Methods.INCR_ADD_V.value:
            method_label = "RSVF"
        elif method_name.value == Methods.BAYES.value:
            method_label = "BCI"
        elif method_name.value == Methods.HOEFFTIGHT.value:
            method_label = "Hoeffding Monotone"
        elif method_name.value == Methods.CENTROID.value:
            method_label = "Mean Transition"

        mean = np.array([r[method_index][1] for r in results_dir])

        sigma = np.array([r[method_index][5] for r in results_dir]) / np.sqrt(sample_steps)
        
        print("mean",mean, "sigma", sigma)
        
        plt.plot(sample_steps, mean, linestyle=lineStyles[method_index%num_styles], marker=markers[method_index%num_markers], alpha=0.7, label = method_label, color=LI_COLORS[method_index%num_colors])
        plt.fill_between(sample_steps, mean - STD_95 * sigma, mean + STD_95 * sigma, alpha=0.2, color=LI_COLORS[method_index%num_colors])

    plt.xlabel('Number of samples')
    plt.ylabel('Calculated return error: '+r'$\mathbb{E}[\xi]$')
    #plt.title('Expected error in return with 95% confidence interval')
    plt.legend(loc='best', fancybox=True, framealpha=0.3)
    plt.yscale('log') #, nonposy='clip'
    plt.grid()
    plt.savefig("fig/"+figure_name)
    plt.show()


# The method takes the data as the first parameter & name of the methods to compare & the figure name
def plot_violations_ext(results_dir, sample_steps, compare_methods, figure_name="Violations_compare.pdf"):
    indices = {}
    methods = [r[0] for r in results_dir[0]]
    for method_index, method_name in enumerate(methods):
        indices[method_name.value] = method_index
        
    method_names = [Methods.CENTROID, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.BAYES, Methods.INCR_ADD_V]
    plt.figure(num=1, figsize=(fig_height, fig_width), dpi=80, facecolor='w', edgecolor='k')

    for method_index, method_name in enumerate(method_names):
        method_index = indices[method_name.value]
        if method_name not in compare_methods:
            continue
        #method_label = "Robustify with Sensible Value Functions (RSVF)" if method_name.value == Methods.INCR_ADD_V.value else method_name.value
        method_label = method_name.value
        if method_name.value == Methods.INCR_ADD_V.value:
            method_label = "RSVF"
        elif method_name.value == Methods.BAYES.value:
            method_label = "BCI"
        elif method_name.value == Methods.HOEFFTIGHT.value:
            method_label = "Hoeffding Monotone"
        elif method_name.value == Methods.CENTROID.value:
            method_label = "Mean Transition"
        plt.plot(sample_steps, [r[method_index][3] for r in results_dir], linestyle=lineStyles[method_index%num_styles], marker=markers[method_index%num_markers], alpha=0.7, label = method_label, color=LI_COLORS[method_index%num_colors])
    plt.xlabel('Number of samples')
    plt.ylabel('Fraction violated')
    #plt.title('L1 threshold values')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    #plt.yscale('log')
    plt.grid()
    plt.savefig("fig/"+figure_name)
    plt.show()





















    
    