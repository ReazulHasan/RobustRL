import Bayesian_Uncertainty_Set
import Gaussian_Uncertainty_Set
from craam import crobust
import Utils
import Plot
import numpy as np
import tqdm

horizon, num_runs = 50, 100
min_population, initial_population, carrying_capacity = 0, 5, 15
mean_growth_rate, std_growth_rate, std_observation = 1.02, 0.02, 2
beta_1, beta_2, n_hat = 0.001, -0.0000021, 9
threshold_control, prob_control, seed = 0, 0.5, 5
discount_factor = 0.9
num_samples, num_actions = 5, 2
bayes_samples = 5
population = np.arange(min_population, carrying_capacity + 1, dtype=np.double)

### Construct the original MDP, solve the MDP & find an arbitrary policy
if __name__ == "__main__":
    seed = np.random.randint(num_runs)
    #initial_population = 30
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
    
    orig_sol = mdp.solve_vi()
    orig_policy = orig_sol.policy
    
    random_policy = np.random.randint(2, size=(carrying_capacity-min_population+1))
    arbitrary_valuefunction = mdp.rewards_vec(random_policy)
    #print(arbitrary_valuefunction)

### Construct uncertainty set for each state-action

def get_transition_reward(current_population, horizon, num_samples, seed):
    transitions_points = [[] for _ in range(num_actions)]
    rewards = np.zeros((num_actions, carrying_capacity-min_population+1))
    
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
            rewards[a,s] = r
        for a in range(num_actions):
            transitions[a] /= count[a] if count[a]>0 else 1
            transitions_points[a].append( (transitions[a]) )

    return transitions_points, rewards
        
def evaluate_uncertainty_set(current_population, num_samples, num_simulation, value_function, confidence_level):
    horizon = 1
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
        
        transitions_points, rewards = get_transition_reward(current_population, \
                                                horizon, num_samples, i)
        
        for a in range(num_actions):
            dir_points = np.asarray(transitions_points[a])
            
            #print(len(dir_points))
            
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
            
            em_nominal, emthreshold = calc_EM_rand(dir_points, confidence_level,\
                                            nominal_prob_bayes)
            em_nominal /= np.sum(em_nominal)
            em_th[a,i] = emthreshold
            em_nominalPoints[a].append(em_nominal)
            
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
                np.mean(incrementallyAddV_nomianlPoints,axis=1))], rewards

#print(evaluate_uncertainty_set(5, 5, 5, arbitrary_valuefunction, 0.9))

###
def incrementally_replace_V(valuefunction, num_samples, num_simulation,\
                                                        num_update, sa_confidence):
    horizon = 1
    X = []
    Y = []
    for i in range(num_update):
        threshold = [[] for _ in range(3)]
        rmdp = crobust.MDP(0, discount_factor)
        for s in population:
            transitions_points, rewards = get_transition_reward(s, horizon,\
                                                        num_samples, i)
            for a in range(num_actions):
                dir_points = np.asarray(transitions_points[a])
                res = construct_uset_known_value_function(dir_points, valuefunction,\
                                                            confidence)
                
                threshold[0].append(s)
                threshold[1].append(a)
                threshold[2].append(res[1])
                
                trp = res[2]
                
                for next_st in population:
                    rmdp.add_transition(s, a, next_st, trp[int(next_st)],\
                                            rewards[a,int(next_st)])
                
        sol = rmdp.rsolve_vi("robust_l1".encode(),threshold)
        valuefunction = sol.valuefunction
        X.append(i)
        Y.append(valuefunction[0])
    print(X, Y)
    simple_generic_plot(X, Y, "Number of samples", "Returned value to initial state")

incrementally_replace_V(arbitrary_valuefunction, 5, 5, 5, 0.9)

###
def incrementally_add_V(valuefunctions, num_samples, num_simulation,\
                                                        num_update, sa_confidence):
    horizon = 1
    X = []
    Y = []
    
    valuefunctions = [valuefunctions]
    th_list = []
    list_transitions_points, list_rewards = {}, {}
    for s in population:
        transitions_points, rewards = get_transition_reward(s, horizon,\
                                        num_samples, np.random.randint(len(population)))
        list_transitions_points[s] = transitions_points
        list_rewards[s] = rewards
        
    for i in range(num_update):
        #print("valuefunctions",i,": ",valuefunctions)
        threshold = [[] for _ in range(3)]
        rmdp = crobust.MDP(0, discount_factor)
        for s in population:
            transitions_points, rewards = list_transitions_points[s], list_rewards[s]
            #get_transition_reward(s, horizon, num_samples, i)
            for a in range(num_actions):
                dir_points = np.asarray(transitions_points[a])
                
                nomianl_points = []
    
                for valuefunction in valuefunctions:
                    res = construct_uset_known_value_function(dir_points, valuefunction,\
                                                            confidence)
                    nomianl_points.append(res[2])
                
                #Find the center of the L1 ball for the nominal points with different 
                #value functions
                trp, th = find_nominal_point(np.asarray(nomianl_points))
                
                if s==0 and a==1:
                    th_list.append(th)
                
                threshold[0].append(s)
                threshold[1].append(a)
                threshold[2].append(th)
                
                for next_st in population:
                    rmdp.add_transition(s, a, next_st, trp[int(next_st)],\
                                            rewards[a,int(next_st)])
                
        sol = rmdp.rsolve_vi("robust_l1".encode(),threshold)
        valuefunction = sol.valuefunction
        valuefunctions.append(valuefunction)
        X.append(i)
        Y.append(valuefunction[0])
    #print(X, Y)
    print(th_list)
    simple_generic_plot(X, Y, "Number of samples", "Returned value to initial state")

incrementally_add_V(arbitrary_valuefunction, 30, 10, 10, 0.9)
### run experiments
if __name__ == "__main__":
    # number of sampling steps
    num_iterations = 5
    # number of runs
    num_simulation = 5
    sample_step = 2
    confidence_level = 0.9
    
    #max number of iterations to improve value functions
    num_update = 5
    
    #(1-overall_confidence) is the total violation allowed. This total violation is distributed among all the state action pairs
    # according to the Union bound.
    sa_confidence = 1 - ((1 - confidence_level) / action_count)
    
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
            params, rewards = evaluate_uncertainty_set(s, num_samples, num_simulation, arbitrary_valuefunction, sa_confidence)
            
            for m in range(Methods.NUM_METHODS.value):
                trp = params[m][3]
                threshold = params[m][1]
                
                for a in range(num_actions):
                    for next_st in population:
                        rmdps[m].add_transition(s, a, next_st, trp[a][int(next_st)],\
                                            rewards[a,int(next_st)])
                    thresholds[m][0].append(s)
                    thresholds[m][1].append(a)
                    thresholds[m][2].append(threshold[a])

        for m in range(Methods.NUM_METHODS.value):
            sol = rmdps[m].rsolve_vi("robust_l1".encode(),np.asarray(thresholds[m]))
            if LI_METHODS[m] is Methods.INCR_REPLACE_V:
                calc_return[m].append(incrementally_replace_V(sol.valuefunction,\
                                num_samples, num_simulation, num_iterations, sa_confidence))
            elif LI_METHODS[m] is Methods.INCR_ADD_V:
                calc_return[m].append(randomly_improve_V(sol.valuefunction,thresholds[m], dir_points))
            else:
                calc_return[m].append(sol.valuefunction[0])

### Plot results
print(calc_return)
generic_plot(sample_steps, calc_return, "Number of samples", "Returned value to initial state")


### Bayesian approach to construct uncertainty set

def evaluate_Bayesian_uncertainty_set(current_population, action):
    
    current_mean_growth_rate = max(0.0, mean_growth_rate -action*current_population*beta_1 - action*max(current_population-n_hat,0)**2 * beta_2 )
    
    current_growth_rate = np.random.normal(current_mean_growth_rate, std_growth_rate)
    
    next_population = max(0, min(carrying_capacity, current_growth_rate * current_population))
    
    #Here, growth rate is a normally distributed random variable & the product with
    #current_population defines the next states. We discretize the distribution over the 
    #next states.
    #Multiplying a random variable by a constant value, multiplies the expected value or mean 
    #by that constant. current_population is the constant here, growth rate is the normally
    #distributed random variable
    current_mean_transition = current_mean_growth_rate * current_population
    
    #Multiplying a random variable by a constant increases the variance by the square of the
    #constant. Hence, increases the std by that constant.
    current_std_transition = std_growth_rate * current_population
    
    true_distribution = discretize_gaussian(min_population, carrying_capacity+1, next_population, std_observation)
    
    mult = np.random.multinomial(num_samples, true_distribution)
    
    #print(current_mean_growth_rate, std_growth_rate)
    #print(current_mean_transition, current_std_transition)
    for ind,val in enumerate(mult):
        if val>0:
            print(ind,val)
    #print(mult)

    
#evaluate_uncertainty_set(10, 1)
#evaluate_uncertainty_set(10, 0)

#evaluate_uncertainty_set(100, 1)
evaluate_Bayesian_uncertainty_set(100, 0)