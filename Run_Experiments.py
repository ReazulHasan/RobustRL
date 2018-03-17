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
    num_next_states = 10
    # number of sampling steps
    num_iterations = 30
    # the desired confidence level
    confidence_level = 0.90
    # number of runs
    runs = 20
    # step size in the number of samples
    sample_step = 20
    
    value_function = np.random.randint(10, size=num_next_states)
    #define reward for the simple mdp with 1 state, 1 action, num_next_states number of next states with uniform transition probability
    reward = np.arange(num_next_states, dtype=np.double)
    
    bayes_results = []
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)
    
    for pos, i in enumerate(tqdm.tqdm(sample_steps)):
        bayes_results.append(evaluate_bayesian_uncertainty(i, num_next_states, reward, confidence_level, runs, value_function))


### Plot Bayesian Results
if __name__ == "__main__":
    #plot_returns(bayes_results, sample_steps, [Methods.BAYES, Methods.EM, Methods.KNOWNV], "Bayes_return_BEK.pdf")
    #plot_thresholds(bayes_results, sample_steps, [Methods.BAYES, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.EM, Methods.KNOWNV], "Bayes_threshold_comparison.pdf")
    plot_violations(bayes_results, sample_steps, [Methods.BAYES, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.EM, Methods.KNOWNV], "bayesian_violations_hoeff_vs_tight.pdf")


### Run Gaussian Experiments
if __name__ == "__main__":
    # number of sampling steps
    num_iterations = 3
    # the desired confidence level
    confidence_level = 0.90
    # number of runs
    runs = 3
    min_demand, max_demand = 0, 100
    demand_mean_prior_mean, demand_mean_prior_std, true_demand_std = 50, 15, 25

    sample_step = 3

    gauss_results = []
    value_function = np.random.randint(10, size=(max_demand-min_demand+1))
    #print("value_function",value_function)
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)
    
    for pos, i in enumerate(tqdm.tqdm(sample_steps)):
        gauss_results.append(evaluate_gaussian_uncertainty(i, confidence_level, runs, value_function, min_demand, max_demand))
    
### Plot Gaussian Results
if __name__ == "__main__":
    plot_returns(gauss_results, sample_steps, [Methods.BAYES, Methods.HOEFF, Methods.HOEFFTIGHT], "gaussian_return_bayes_hoeff_tight.pdf")
    plot_thresholds(gauss_results, sample_steps, [Methods.HOEFF, Methods.HOEFFTIGHT],"gaussian_threhold_hoeff_vs_tight.pdf")
    plot_violations(gauss_results, sample_steps, [Methods.HOEFF, Methods.HOEFFTIGHT, Methods.KNOWNV],"gaussian_violations_hoeff_vs_tight.pdf")
    
###Inventory Simulation
if __name__ == "__main__":
    initial, max_inventory, purchase_cost, sale_price = 0, 50, 2.0, 3.0,
    prior_mean, prior_std, demand_std, rand_seed =  10.0, 5.0, 6.0, 3
    horizon, runs = 10, 5
    inventory_simulator = crobust.SimulatorInventory(initial, prior_mean, prior_std, demand_std, purchase_cost, sale_price, max_inventory, rand_seed)
    samples = inventory_simulator.simulate_inventory(horizon, runs)
    
    states_from = samples.get_states_from()
    actions = samples.get_actions()
    states_to = samples.get_states_to()
    
    demands = []
    for i in range(len(states_from)):
        demands.append(states_from[i] + actions[i] - states_to[i])   
    
    smdp = crobust.SampledMDP()
    smdp.add_samples(samples)
    mdp = smdp.get_mdp(0.9)
    
    state_count = mdp.state_count()
    action_count = 0
    for i in range(state_count):
        action_count += mdp.action_count(i)
    
    overall_confidence = 0.9
    
    #(1-overall_confidence) is the total violation allowed. This total violation is distributed among all the state action pairs
    # according to the Union bound.
    sa_confidence = 1 - ((1 - overall_confidence) / action_count)

    min_demand, max_demand = 0, max(demands)
    demand_mean_prior_mean = np.mean(demands)
    demand_mean_prior_std = np.std(demands)
    true_demand_std = demand_std
    
    num_iterations = 3
    # number of runs
    runs = 3
    sample_step = 3
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)
    
    #initial, randomly assigned value function
    value_function = np.random.randint(10, size=(max_demand-min_demand+1))
    print(max_demand)
    #In thresholds, the first dimension is methods (e.g Bayesian, EM etc.)
    #for each method, there are 3 lists containing state-action-threshold packed into a list
    thresholds = [ [[] for _ in range(3)] for _ in range(Methods.NUM_METHODS.value) ]
    calc_return = [[] for _ in range(Methods.NUM_METHODS.value)]

###
if __name__ == "__main__":
    for pos, num_samples in enumerate(tqdm.tqdm(sample_steps)):
        for s in range(mdp.state_count()):
            actions = mdp.action_count(s)
            for a in range(actions):
                gu = evaluate_gaussian_uncertainty(num_samples, sa_confidence, runs, value_function, min_demand, max_demand,\
                                    demand_mean_prior_mean, demand_mean_prior_std, true_demand_std)
                for m in range(Methods.NUM_METHODS.value):
                    thresholds[m][0].append(s)
                    thresholds[m][1].append(a)
                    thresholds[m][2].append(gu[m][2])

        for m in range(Methods.NUM_METHODS.value):
            sol = mdp.rsolve_vi("robust_l1".encode(),thresholds[m])
            calc_return[m].append(sol.valuefunction[0])
            print(LI_METHODS[m].value,sol.valuefunction)

### Plot inventory simulation
generic_plot(sample_steps, calc_return, "Number of samples", "Return on the initial state", "lower right", "Number of samples vs. return", "MDP_return")

### Improve over when the value function is known
if __name__ == "__main__":
    #initially assign random value function to each state
    value_function = np.random.randint(10, size=(max_demand-min_demand+2))
    print(len(value_function))
    tuple_size = 3 #s-a-th

    num_samples = 10
    knownV_threshold = np.zeros((tuple_size, action_count))
    knownV_nominal_points = {}

    X = []
    Y = []

    #this loop iterates incrementally with the latest value function 
    #to further improve upon
    for i in tqdm.tqdm(range(5)):
        print(knownV_threshold)
        position=0
        for s in range(mdp.state_count()):
            actions = mdp.action_count(s)
            for a in range(actions):

                #Computes the return, threshold, nominal point etc. for 
                #current state & action
                guk = evaluate_gaussian_knownV(num_samples, sa_confidence, runs, value_function, min_demand, max_demand, demand_mean_prior_mean, demand_mean_prior_std, true_demand_std)

                if (s,a) not in knownV_nominal_points:
                    knownV_nominal_points[(s,a)] = []

                if len(knownV_nominal_points[(s,a)])>=2:
                    #find the nominal point of all the nominal points
                    nominalp_of_nominal = find_nominal_point(np.asarray\
                                            (knownV_nominal_points[(s,a)]))
                    uset, threshold = get_uset(np.asarray(knownV_nominal_points[(s,a)]), np.asarray(nominalp_of_nominal), len(knownV_nominal_points[(s,a)]))

                    #Compute the distance between the current nominal point & 
                    #the nominal point of all the previous nominal points
                    dist = np.linalg.norm(guk[0][7] - nominalp_of_nominal, ord = 1)

                    #if the new nominal point lies inside the previously
                    #constructed l1-ball, a reasonable estimation is found 
                    #& continue without updating the threshold for this 
                    #state-action
                    if dist<threshold:
                        #print("dist<threshold",s,a,knownV_threshold[2,position])
                        position+=1
                        continue

                #Stack state-action-threshold to pass to the mdp solver for 
                #robust solution
                knownV_threshold[0,position] = s
                knownV_threshold[1,position] = a
                knownV_threshold[2,position] = guk[0][2]

                #print("KnownV_nomianl_point",guk[0][7])
                knownV_nominal_points[(s,a)].append(guk[0][7])
                position+=1

        sol = mdp.rsolve_vi("robust_l1".encode(),knownV_threshold)
        vf = sol.valuefunction
        value_function = []

        #As S,s policy is used as random policy to generate samples, 
        #the possible inventory levels are, o & (max_inventory - max_demand) 
        #to max_inventory, inclusive. So filter out the value functions
        #for possible states.
        value_function.append(vf[0])
        for v in range(max_demand+1,0,-1):
            value_function.append(vf[-v])
        X.append(i)
        Y.append(vf[0])
        print(i,"value_function",vf,value_function)
    print(X,Y)
    print("knownV_nominal_points",knownV_nominal_points[(0,0)][0])
    simple_generic_plot(X, Y, "Iteration over Value Function", "Return on the initial state", "lower right", "Number of interations vs. return", "MDP_return_1")

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























    
    