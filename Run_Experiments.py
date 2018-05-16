import Dirichlet_Uncertainty_set
import Gaussian_Uncertainty_Set
from craam import crobust
import Utils
import Plot
import numpy as np
import tqdm
import pickle

### Run Bayesian Experiments
if __name__ == "__main__":
    # number of assumes states in the MDP
    num_next_states = 5
    # number of sampling steps
    num_iterations = 10 #30
    # the desired confidence level
    confidence_level = 0.95
    # number of runs
    runs = 200 #30
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

#with open ('outfile', 'rb') as fp:
#    itemlist = pickle.load(fp)

### Plot Bayesian Results
if __name__ == "__main__":
    compare_methods = [Methods.BAYES.value, Methods.CENTROID.value, Methods.HOEFF.value, Methods.HOEFFTIGHT.value,  Methods.INCR_ADD_V.value]
    plot_returns(bayes_results, sample_steps, compare_methods, "Dirichlet_return_single_state.pdf",runs)
    plot_thresholds(bayes_results, sample_steps, compare_methods, "Dirichlet_threshold_single_state.pdf",runs)
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
    # number of sampling steps
    num_iterations = 10
    # the desired confidence level
    confidence_level = 0.95
    # number of runs
    runs = 100
    min_demand, max_demand = 0, 150
    demand_mean_prior_mean, demand_mean_prior_std, true_demand_std = 80, 15, 13

    sample_step = 10

    gauss_results = []
    value_function = np.arange(max_demand-min_demand+1)
    #np.random.uniform(low=0, high=10, size=(max_demand-min_demand+1))
    print("value_function",value_function)
    
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)

    #for pos, i in enumerate(tqdm.tqdm(sample_steps)):
        #gauss_results.append(evaluate_gaussian_uncertainty(i, confidence_level, runs, value_function, value_function, [value_function], min_demand, max_demand))
        
    for pos, i in enumerate(tqdm.tqdm(sample_steps)):
        gauss_results.append(evaluate_gaussian_uncertainty(i, confidence_level, runs, value_function, min_demand, max_demand))

with open('dumped_results/gauss_results'+"_iteration_"+str(num_iterations)\
        +"_conf_"+str(confidence_level)+"_runs_"+str(runs)+"_step_"+str(sample_step),'wb') as fp:
    
    pickle.dump([gauss_results, sample_steps], fp)

### Plot Gaussian Results
if __name__ == "__main__":
    compare_methods = [Methods.BAYES, Methods.CENTROID, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.INCR_ADD_V]
    plot_returns(gauss_results, sample_steps, compare_methods, "gaussian_return_single_state.pdf", runs)
    plot_thresholds(gauss_results, sample_steps, compare_methods,"gaussian_threshold_single_state.pdf", runs)
    plot_violations(gauss_results, sample_steps, compare_methods ,"gaussian_violation_single_state.pdf")

###Load & Plot
if __name__ == "__main__":
    import pickle
    
    f = open('dumped_results/gauss_results_20_0.95_200_10_05.09.18', 'rb')   # 'r' for reading; can be omitted
    gauss_results = pickle.load(f)         # load file content as mydict
    f.close()                
    
    num_iterations = 10
    sample_step = 10
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)
    
    compare_methods = [Methods.BAYES.value, Methods.CENTROID.value, Methods.HOEFF.value, Methods.HOEFFTIGHT.value, Methods.INCR_ADD_V.value]
    plot_returns(gauss_results[:10], sample_steps, compare_methods, "gaussian_return_single_state.pdf", runs)
    plot_thresholds(gauss_results[:10], sample_steps, compare_methods,"gaussian_threshold_single_state.pdf", runs)
    plot_violations(gauss_results[:10], sample_steps, compare_methods ,"gaussian_violation_single_state.pdf")

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























    
    