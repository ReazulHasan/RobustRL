import Bayesian_Uncertainty_Set
import Gaussian_Uncertainty_Set
from craam import crobust
import Utils
import Plot
import numpy as np
import tqdm


### Run Bayesian Experiments
if __name__ == "__main__":
    # number of assumes states in the MDP
    num_next_states = 5
    # number of sampling steps
    num_iterations = 5
    # the desired confidence level
    confidence_level = 0.90
    # number of runs
    runs = 5
    # step size in the number of samples
    sample_step = 5
    
    value_function = np.random.randint(10, size=num_next_states)
    #define reward for the simple mdp with 1 state, 1 action, num_next_states number of next states with uniform transition probability
    reward = np.arange(num_next_states, dtype=np.double)
    
    bayes_results = []
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)
    
    for pos, i in enumerate(tqdm.tqdm(sample_steps)):
        bayes_results.append(evaluate_bayesian_uncertainty(i, num_next_states, reward, confidence_level, runs, value_function))

###Save results
import pickle

with open('dumped_results/Bayes_result_'+str(num_next_states)+"_"+str(num_iterations)\
        +"_"+str(confidence_level)+"_"+str(runs)+"_"+str(sample_step),'wb') as fp:
    
    pickle.dump(bayes_results, fp)

#with open ('outfile', 'rb') as fp:
#    itemlist = pickle.load(fp)

### Plot Bayesian Results
if __name__ == "__main__":
    plot_returns(bayes_results, sample_steps, [Methods.BAYES, Methods.HOEFF, Methods.HOEFFTIGHT], "Bayes_return_BHHT.pdf",runs)
    #plot_returns(bayes_results, sample_steps, [Methods.BAYES, Methods.EM, Methods.KNOWNV], "Bayes_return_BEK.pdf",runs)
    plot_thresholds(bayes_results, sample_steps, [Methods.BAYES, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.EM, Methods.KNOWNV], "Bayes_threshold_comparison.pdf",runs)
    plot_violations(bayes_results, sample_steps, [Methods.BAYES, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.EM, Methods.KNOWNV], "Bayes_violations_comparison.pdf")


### Run Gaussian Experiments
if __name__ == "__main__":
    # number of sampling steps
    num_iterations = 5
    # the desired confidence level
    confidence_level = 0.90
    # number of runs
    runs = 5
    min_demand, max_demand = 0, 100
    demand_mean_prior_mean, demand_mean_prior_std, true_demand_std = 50, 15, 25

    sample_step = 2

    gauss_results = []
    value_function = np.random.uniform(low=0, high=10, size=(max_demand-min_demand+1))
    print("value_function",value_function)
    
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)
    
    #for pos, i in enumerate(tqdm.tqdm(sample_steps)):
        #gauss_results.append(evaluate_gaussian_uncertainty(i, confidence_level, runs, value_function, value_function, [value_function], min_demand, max_demand))
        
    for pos, i in enumerate(tqdm.tqdm(sample_steps)):
        gauss_results.append(evaluate_gaussian_uncertainty(i, confidence_level, runs, value_function, min_demand, max_demand))
 
    
### Plot Gaussian Results
if __name__ == "__main__":
    compare_methods = [Methods.BAYES, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.INCR_REPLACE_V, Methods.INCR_ADD_V]
    plot_returns(gauss_results, sample_steps, compare_methods, "gaussian_return_single_state.pdf", runs)
    #plot_thresholds(gauss_results, sample_steps, [Methods.HOEFF, Methods.HOEFFTIGHT],"gaussian_threhold_hoeff_vs_tight.pdf", runs)
    #plot_violations(gauss_results, sample_steps, [Methods.HOEFF, Methods.HOEFFTIGHT, Methods.KNOWNV],"gaussian_violations_hoeff_vs_tight.pdf")


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























    
    