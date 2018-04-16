import Bayesian_Uncertainty_Set
import Gaussian_Uncertainty_Set
from craam import crobust
import Utils
import Plot
import numpy as np
import tqdm
import time

horizon, num_runs = 20, 20
min_population, carrying_capacity = 0, 20
initial_population = int(carrying_capacity/3)#np.random.randint(min_population, carrying_capacity)
mean_growth_rate, std_growth_rate, std_observation = 1.1, 0.4, 2
beta_1, beta_2, n_hat = 0.001, -0.0000021, int(carrying_capacity*2/3)
threshold_control, prob_control, seed = 0, 0.5, 5
discount_factor = 0.9
num_samples, num_actions = 20, 2
population = np.arange(min_population, carrying_capacity + 1, dtype=np.double)

### Construct uncertainty set for each state-action

def get_Bootstrapped_transition_kernel(current_population, horizon, num_samples, seed):
    """
    Use bootstrapping to produce a transition matrix for all possible actions from current state.
    Sample multiple transitions from a specific state-action to compute a transition probability over
    next states. 
    
    @current_population The level of current population, which really is the state
    @horizon Sampling horizon for the bootstrapping
    @num_samples Number of bootstrapped samples
    @seed Seed for the random number
    
    @return transition_points Transition points for all the actions
    """
    
    transitions_points = [[] for _ in range(num_actions)]
    #rewards = np.zeros((num_actions, carrying_capacity-min_population+1))
    
    for i in range(num_samples):
        transitions = np.zeros((num_actions, carrying_capacity-min_population+1))

        species_simulator = crobust.SimulatorSpecies(current_population, carrying_capacity,\
                    mean_growth_rate, std_growth_rate, std_observation, beta_1, beta_2, n_hat,\
                    threshold_control, prob_control, i*seed)
        samples = species_simulator.simulate_species(horizon, num_runs)
        
        count = np.zeros((num_actions))
        for a,s,r in zip(samples.get_actions(),samples.get_states_to(),samples.get_rewards()):
            transitions[a,s] += 1
            count[a] +=1
            #rewards[a,s] = r
        for a in range(num_actions):
            transitions[a] /= count[a] if count[a]>0 else 1
            transitions_points[a].append( (transitions[a]) )
    #print("len(transitions_points[0])",len(transitions_points[0][0]))
    return transitions_points#, rewards

### Bayesian approach to construct uncertainty set

def get_Bayesian_transition_kernel(current_population, num_samples):
    """
    Use Bayesian approach to produce a transition matrix for all possible actions from current state.
    Use posterior distribution over the next population from the prior growth rate distribution.
    
    @current_population The level of current population, which really is the state
    @num_samples Number of samples from true distribution
    
    @return transition_points Transition points for all the actions
    """
    bayes_samples = 300
    transitions_points = {}#[[] for _ in range(num_actions)]
    if current_population==0:
        current_population=1
    for action in range(num_actions):
        growth_rate_mean_prior_mean = max(0.0, mean_growth_rate - action * \
        current_population * beta_1 - action*max(current_population-n_hat,0)**2 * beta_2 )
        
        growth_rate_mean_prior_std = std_growth_rate
        
        true_growth_rate_mean = np.random.normal(growth_rate_mean_prior_mean,\
                                                        growth_rate_mean_prior_std)
        #let's take the same prior & true std for simplicity
        true_growth_rate_std = growth_rate_mean_prior_std
        
        #Here, growth rate is a normally distributed random variable & the product with
        #current_population defines the next states. We discretize the distribution over the 
        #next states.
        #Multiplying a random variable by a constant value, multiplies the expected value or mean 
        #by that constant. current_population is the constant here, growth rate is the normally
        #distributed random variable
        growth_rate_mean_prior_mean = growth_rate_mean_prior_mean * current_population
        true_population_mean = true_growth_rate_mean * current_population
        
        #Multiplying a random variable by a constant increases the variance by the square of the
        #constant. Hence, increases the std by that constant.
        growth_rate_mean_prior_std = growth_rate_mean_prior_std * current_population
        true_population_std = true_growth_rate_std * current_population
        
        true_distribution = discretize_gaussian(min_population, carrying_capacity,\
                                        true_population_mean, true_population_std)
        
        mult = np.random.multinomial(num_samples, true_distribution)
        
        estmean_population_mean, estmean_population_std = normal_aposteriori(population, mult, \
                            true_population_std, growth_rate_mean_prior_mean, growth_rate_mean_prior_std)
        
        dir_points = np.array([discretize_gaussian(min_population, carrying_capacity,\
                    np.random.normal(estmean_population_mean, estmean_population_std),\
                    true_population_std) for k in range(bayes_samples)])
        
        transitions_points[action] = dir_points
        #print(dir_points)
    return transitions_points

#start_time = time.time()
#for i in range(250):
    #get_Bayesian_transition_kernel(10, 50)
#print(get_Bayesian_transition_kernel(10, 50))
#print("--- %s seconds ---" % (time.time() - start_time))

def calc_reward(next_state, trp_to_next_state, action):
    """
    Compute the reward for the next state & action.
    
    @next_state The next state in the transition
    @trp_to_next_state Transition probability for the next state
    @action The action taken
    
    @return reward Computed reward
    """
    return next_state*trp_to_next_state*(-1) + action * (-4000)
    
### Construct a boostrapped MDP, solve it, find an arbitrary policy
if __name__ == "__main__":
    seed = np.random.randint(num_runs)
    #initial_population = 30
    '''
    species_simulator = crobust.SimulatorSpecies(initial_population, carrying_capacity,\
                mean_growth_rate, std_growth_rate, std_observation, beta_1, beta_2, n_hat,\
                threshold_control, prob_control, seed)
    samples = species_simulator.simulate_species(horizon, num_runs)
    
    states_from = samples.get_states_from()
    actions = samples.get_actions()
    states_to = samples.get_states_to()
    rewards = samples.get_rewards()
    
    smdp = crobust.SampledMDP()
    smdp.add_samples(samples)
    mdp = smdp.get_mdp(discount_factor)
    '''
    mdp = crobust.MDP(0, discount_factor)
    for s in population:
        transitions_points = get_Bootstrapped_transition_kernel(s, horizon, 1, s)
        for a in range(num_actions):
            trp = transitions_points[a][0]
            
            for next_st in population:
                reward = calc_reward(next_st, trp[int(next_st)], a)
                mdp.add_transition(s, a, next_st, trp[int(next_st)], reward)
    
    orig_sol = mdp.solve_vi()
    orig_policy = orig_sol.policy
    
    #print(len(orig_policy))
    random_policy = np.random.randint(2, size=(carrying_capacity-min_population+1))
    arbitrary_valuefunction = mdp.solve_vi(policy=random_policy).valuefunction#rewards_vec(random_policy)
    
    print(arbitrary_valuefunction)
    initial = np.ones(carrying_capacity-min_population+1)/(carrying_capacity-min_population+1)
    #print(initial)
    #print(np.dot(initial,orig_sol.valuefunction))

###
def evaluate_uncertainty_set(current_population, num_samples, num_simulation, value_function, confidence_level):
    """
    Run evaluation of the uncertainty set to compute the nominal point & threshold with different
    methods (e.g. Bayes Simple, Hoeffding etc.).
    
    @current_population Current population level
    @num_samples Number of samples to estimate the true distribution
    @num_simulation Number of simulation
    @value_function The initially known value function
    @confidence_level The required confidence level
    
    @return nominal points & threshold for different methods
    """
    horizon = 1 #only take samples of the next states from current state
    num_next_states = carrying_capacity-min_population+1
    num_v = len(value_function)
    
    bayes_th = np.zeros((num_actions, num_simulation))
    hoeff_th = np.zeros((num_actions, num_simulation))
    tight_hoeff_th = np.zeros((num_actions, num_simulation))
    em_th = np.zeros((num_actions, num_simulation))
    incrementallyReplaceV_th = np.zeros((num_actions, num_simulation))
    incrementallyAddV_th = np.zeros((num_actions, num_simulation))
    
    bayes_nominalPoints = [[] for _ in range(num_actions)]
    em_nominalPoints = [[] for _ in range(num_actions)]
    knownV_nominalPoints = [[] for _ in range(num_actions)]
    incrementallyReplaceV_nominalPoints = [[] for _ in range(num_actions)]
    incrementallyAddV_nomianlPoints = [[] for _ in range(num_actions)]
    
    for i in range(num_simulation):
        
        #transitions_points = get_Bootstrapped_transition_reward(current_population, \
                                                #horizon, num_samples, i)
        transitions_points = get_Bayesian_transition_kernel(current_population, num_samples)
        
        for a in range(num_actions):
            dir_points = np.asarray(transitions_points[a])
            
            #print("dir_points.shape",dir_points[0].shape, dir_points[0])
            
            nominal_prob_bayes = np.mean(dir_points, axis=0)
            nominal_prob_bayes /= np.sum(nominal_prob_bayes)
            
            bayes_nominalPoints[a].append(nominal_prob_bayes)
            
            #get uncertainty set & threshold
            bayes_th[a,i] = compute_bayesian_threshold(dir_points,nominal_prob_bayes,\
                                confidence_level)        
            
            #calc threshold from hoeffding bound equation
            hoeff_th[a,i] = np.sqrt((2 / num_samples )*np.log((2**num_next_states-2) \
                                / (1-confidence_level) ))   
            
            # ** calculate the tight hoeffding bound
            tight_hoeff_th[a,i]= np.sqrt((2 / num_samples )*np.log((num_next_states-1) \
                                / (1 - confidence_level) ))   # TODO: should be -1 or -2?
            """
            em_nominal, emthreshold = calc_EM_rand(dir_points, confidence_level,\
                                            nominal_prob_bayes)
            em_nominal /= np.sum(em_nominal)
            em_th[a,i] = emthreshold
            em_nominalPoints[a].append(em_nominal)
            """
            ivf = construct_uset_known_value_function(dir_points, value_function,\
                                                            confidence_level)
            incrementallyReplaceV_th[a,i] = ivf[1]
            incrementallyReplaceV_nominalPoints[a].append(ivf[2])
            
            
            rvf = construct_uset_known_value_function(dir_points, value_function,\
                                                            confidence_level)
            incrementallyAddV_th[a,i] = rvf[1]
            incrementallyAddV_nomianlPoints[a].append(rvf[2])
    
    return [(Methods.BAYES, np.mean(bayes_th, axis=1), np.std(bayes_th, axis=1),\
                np.mean(bayes_nominalPoints, axis=1) ),\
            (Methods.HOEFF, np.mean(hoeff_th, axis=1), np.std(hoeff_th, axis=1),\
                np.mean(bayes_nominalPoints, axis=1) ),\
            (Methods.HOEFFTIGHT, np.mean(tight_hoeff_th, axis=1),\
                np.std(tight_hoeff_th, axis=1),\
                np.mean(bayes_nominalPoints, axis=1)),\
            (Methods.EM, np.mean(em_th, axis=1), np.std(em_th, axis=1),\
                np.mean(em_nominalPoints, axis=1) ),\
            (Methods.INCR_REPLACE_V, np.mean(incrementallyReplaceV_th, axis=1),\
                np.std(incrementallyReplaceV_th, axis=1),\
                np.mean(incrementallyReplaceV_nominalPoints,axis=1)),\
            (Methods.INCR_ADD_V, np.mean(incrementallyAddV_th, axis=1),\
                np.std(incrementallyAddV_th,axis=1),\
                np.mean(incrementallyAddV_nomianlPoints,axis=1))]

#print(evaluate_uncertainty_set(5, 5, 5, arbitrary_valuefunction, 0.9))

###
def incrementally_replace_V(valuefunction, num_samples, num_simulation,\
                                                        num_update, sa_confidence):
    """
    Method to incrementally improve the value function by replacing the old value function with 
    the new one.
    
    @value_function The initially known value function
    @num_samples Number of samples to estimate the true distribution
    @num_simulation Number of simulation
    @num_update Number of updates over the value functions
    @sa_confidence Required confidence for each state-action from 
    
    @return valuefunction The updated final value function
    """
    horizon = 1
    X = []
    Y = []
    
    list_transitions_points = {}
    for s in population:
        #transitions_points = get_Bootstrapped_transition_reward(s, horizon,\
                                        #num_samples, np.random.randint(len(population)))
        transitions_points = get_Bayesian_transition_kernel(s, num_samples)
        list_transitions_points[s] = transitions_points
    
    regret = 99999
    for i in range(num_update):
        threshold = [[] for _ in range(3)]
        rmdp = crobust.MDP(0, discount_factor)
        for s in population:
            #transitions_points = get_Bootstrapped_transition_reward(s, horizon, num_samples, i)
            transitions_points = list_transitions_points[s] #get_Bayesian_transition_kernel(s, num_samples)
            for a in range(num_actions):
                dir_points = np.asarray(transitions_points[a])
                res = construct_uset_known_value_function(dir_points, valuefunction,\
                                                            confidence)
                
                threshold[0].append(s)
                threshold[1].append(a)
                threshold[2].append(res[1])
                
                trp = res[2]
                
                for next_st in population:
                    reward = calc_reward(next_st, trp[int(next_st)], a)
                    rmdp.add_transition(s, a, next_st, trp[int(next_st)], reward)
                
        rsol = rmdp.rsolve_vi("robust_l1".encode(),threshold)
        sol = rmdp.solve_vi()
        cur_regret = np.dot(initial,sol.valuefunction) - np.dot(initial,rsol.valuefunction)
        if cur_regret>regret:
            break
        regret = cur_regret
        valuefunction = rsol.valuefunction
        X.append(i)
        Y.append(valuefunction[0])
    #print(X, Y)
    #simple_generic_plot(X, Y, "Number of samples", "Returned value to initial state")
    return regret #valuefunction
#incrementally_replace_V(arbitrary_valuefunction, 5, 5, 5, 0.9)

###
def incrementally_add_V(valuefunctions, num_samples, num_simulation,\
                                                        num_update, sa_confidence):
    """
    Method to incrementally improve value function by adding the new value function with 
    previous valuefunctions, finding the nominal point & threshold for this cluster of value functions
    with the required sa-confidence.
    
    @value_function The initially known value function
    @num_samples Number of samples to estimate the true distribution
    @num_simulation Number of simulation
    @num_update Number of updates over the value functions
    @sa_confidence Required confidence for each state-action from 
    
    @return valuefunction The updated final value function
    """
    horizon = 1
    X = []
    Y = []
    
    valuefunctions = [valuefunctions]
    th_list = []
    list_transitions_points = {}
    for s in population:
        #transitions_points = get_Bootstrapped_transition_reward(s, horizon,\
                                        #num_samples, np.random.randint(len(population)))
        transitions_points = get_Bayesian_transition_kernel(s, num_samples)
        list_transitions_points[s] = transitions_points
    
    #Store the nominal points for each state-action pairs
    nomianl_points = {}
    
    #Store the latest nominal of nominal point & threshold
    nominal_threshold = {}
    regret = 0
    print(num_update)
    for i in range(num_update):
        #print("valuefunctions",i,": ",valuefunctions)
        #keep track whether the current iteration keeps the mdp unchanged
        is_mdp_unchanged = True
        threshold = [[] for _ in range(3)]
        rmdp = crobust.MDP(0, discount_factor)
        for s in population:
            transitions_points = list_transitions_points[s]
            #get_transition_reward(s, horizon, num_samples, i)
            for a in range(num_actions):
                dir_points = np.asarray(transitions_points[a])

                #nomianl_points = []

                #for valuefunction in valuefunctions:
                res = construct_uset_known_value_function(dir_points, valuefunctions[-1],\
                                                            confidence)
                
                if (s,a) not in nomianl_points:
                    nomianl_points[(s,a)] = []
                
                trp, th = None, 0
                #If there's a previously constructed L1 ball. Check whether the new nominal point
                #needs to be considered.
                if (s,a) in nominal_threshold:
                    old_trp = nominal_threshold[(s,a)][0]
                    old_th = nominal_threshold[(s,a)][1]
                    
                    #Compute the L1 distance between the newly computed nominal point & the previous 
                    #nominal of nominal points
                    new_th = np.linalg.norm(res[2] - old_trp, ord = 1)
                    
                    #The new point is inside the previous L1 ball. Don't consider it & proceed with
                    #the previous trp & threshold
                    if  (new_th - old_th) < 0.0001:
                        trp, th = old_trp, old_th
                
                #Consider the new nominal point to construct a new uncertainty set. This block will
                #execute if there's no previous nominal_threshold entry or the new nominal point
                #resides outside
                if trp is None:
                    #print(i,"trp is None")
                    is_mdp_unchanged = False
                    nomianl_points[(s,a)].append(res[2])
                    
                    start_time = time.time()
                    #Find the center of the L1 ball for the nominal points with different 
                    #value functions
                    trp, th = find_nominal_point(np.asarray(nomianl_points[(s,a)]))
                    nominal_threshold[(s,a)] = (trp, th)
                    #print("find_nominal_point --- %s seconds ---" % (time.time() - start_time))
                #if s==5 and i==num_update-1:
                    #print("incrementally_add_V, transitions_points",trp, "Threshold", th)
                
                threshold[0].append(s)
                threshold[1].append(a)
                threshold[2].append(th)
                
                for next_st in population:
                    reward = calc_reward(next_st, trp[int(next_st)], a)
                    rmdp.add_transition(s, a, next_st, trp[int(next_st)], reward)
        
        rsol = rmdp.rsolve_vi("robust_l1".encode(),threshold)
        
        #If the whole MDP is unchanged, meaning the new value function didn't change the uncertanty
        #set for any state-action, no need to iterate more!
        if is_mdp_unchanged or i==num_update:
            #print("MDP remains unchanged after number of iteration:",i)
            sol = rmdp.solve_vi()
            regret = np.dot(initial,sol.valuefunction) - np.dot(initial,rsol.valuefunction)
            break

        valuefunction = rsol.valuefunction
        valuefunctions.append(valuefunction)
        X.append(i)
        Y.append(valuefunction[0])
    #print("valuefunctions",valuefunctions)
    #print("Incrementally add V", "threshold: ", threshold)
    #print(X, Y)
    #simple_generic_plot(X, Y, "Number of samples", "Returned value to initial state")
    return regret #valuefunctions[-1]

#incrementally_add_V(arbitrary_valuefunction, 30, 10, 10, 0.9)

### run experiments
if __name__ == "__main__":
    # number of sampling steps
    num_iterations = 10
    # number of runs
    num_simulation = 10
    sample_step = 5
    confidence_level = 0.9
    
    #max number of iterations to improve value functions
    num_update = 10
    
    #(1-overall_confidence) is the total violation allowed. This total violation is distributed among all the state action pairs
    # according to the Union bound.
    sa_confidence = 1 - ( (1 - confidence_level) / (num_actions * (carrying_capacity-min_population+1)) )
    
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)
    
    #In thresholds, the first dimension is methods (e.g Bayesian, EM etc.)
    #for each method, there are 3 lists containing state-action-threshold packed into a list
    thresholds = [ [[] for _ in range(3)] for _ in range(Methods.NUM_METHODS.value) ]
    calc_return = [[] for _ in range(Methods.NUM_METHODS.value)]
    
    for pos, num_samples in enumerate(tqdm.tqdm(sample_steps)):        
        rmdps = []
        for m in range(Methods.NUM_METHODS.value):
            rmdps.append(crobust.MDP(0, discount_factor))
        
        for s in population:          
            params = evaluate_uncertainty_set(s, num_samples, num_simulation, arbitrary_valuefunction, sa_confidence)
            
            for m in range(Methods.NUM_METHODS.value):
                if LI_METHODS[m] is Methods.EM:
                    continue
                trp = params[m][3]
                threshold = params[m][1]
                
                #if s==5 and m==0:
                    #print("transition probability",trp,"threshold",threshold)
                
                for a in range(num_actions):
                    for next_st in population:
                        reward = calc_reward(next_st, trp[a][int(next_st)], a)
                        rmdps[m].add_transition(s, a, next_st, trp[a][int(next_st)], reward)
                    thresholds[m][0].append(s)
                    thresholds[m][1].append(a)
                    thresholds[m][2].append(threshold[a])

        for m in range(Methods.NUM_METHODS.value):
            if LI_METHODS[m] is Methods.EM:
                continue
            #if LI_METHODS[m] == Methods.BAYES:
                #print(Methods.BAYES.value," ", rmdps[m].to_json(), "thresholds: ", thresholds[m])
            rsol = rmdps[m].rsolve_vi("robust_l1".encode(),np.asarray(thresholds[m]))
            sol = rmdps[m].solve_vi()
            #print("Method",LI_METHODS[m].value,"value function",sol.valuefunction,"policy",sol.policy)
            if LI_METHODS[m] is Methods.INCR_REPLACE_V:
                regret = incrementally_replace_V(rsol.valuefunction, num_samples,\
                                                num_simulation, num_update, sa_confidence)
                calc_return[m].append(regret)
            elif LI_METHODS[m] is Methods.INCR_ADD_V:
                regret = incrementally_add_V(rsol.valuefunction, num_samples,\
                                                num_simulation, num_update, sa_confidence)
                calc_return[m].append(regret)
            else:
                calc_return[m].append( np.dot(initial,sol.valuefunction) -\
                                                np.dot(initial,rsol.valuefunction))

### Plot results
print(calc_return)
#generic_plot(sample_steps, calc_return, "Number of samples", "Total expected return \n (initial distribution x valuefunction)")

generic_plot(sample_steps, calc_return, "Number of samples", 'Calculated return error', legend_pos="upper right", figure_name="Generic_plot_Return_Error.pdf")



