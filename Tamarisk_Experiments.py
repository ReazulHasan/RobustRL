"""
Overall approaches for Tamarisk experiments:
The 2.7 python codes are used with RL glue to generate samples. Codes residing at: /home/reazul/Summer-2017/ML+Summer/Summer-2017/TamariskInvasiveSpecies/invasive_species/src are used with the virtual env at: /home/reazul/Summer-2017/ML+Summer/Summer-2017/RL_Glue/python2_venv. After running the venv, navigate to the src directory & run the following commands to generate samples & write to tamarix_samples.txt

rl_glue &
python InvasiveAgent.py &
python InvasiveEnvironment.py &
python InvasiveExperiment.py &

Rewards for all state-action is precomputed. All the states & actions are generated & saved into files by init_states() & init_actions() methods of this file. this file are copied to /home/reazul/Summer-2017/ML+Summer/Summer-2017/TamariskInvasiveSpecies/invasive_species/modified_src & modified invasive_enviroenment.py file is run to produce the state-action-reward & write into tamarix_rewards_for_all_states_action.txt file. This file is then copied to this scripts directory to parse & use.

"""

import Dirichlet_Uncertainty_set
import Gaussian_Uncertainty_Set
from craam import crobust
import Utils
import Plot
import numpy as np
import tqdm
import time
import random
import math

nbrReaches=2
habitatSize=1
num_states = 3
num_actions = 4
# number of samples of the true distribution to take when estimating the Bayes samples
bayes_samples = 25

###Generate all possible states & action

#generate all states, assign unique index to each state
state_to_index = {}
index_to_state = {}
state_index = 0

file_name = "tamarix_all_states.txt"
file = open(file_name,'w')

def init_states(state):
    global state_index
    if len(state) >= nbrReaches*habitatSize:
        file.write(state+'\n')
        state_to_index[state] = state_index
        index_to_state[state_index] = state
        state_index += 1
        return
    for i in range(1,num_states+1):
        init_states(state+str(i))

#print(state_to_index)
init_states("")
file.close()

#generate all actions, assign unique index to each action
action_to_index = {}
index_to_action = {}
action_index = 0

file_name = "tamarix_all_actions.txt"
file = open(file_name,'w')

def init_actions(action):
    global action_index
    if len(action) >= nbrReaches:
        file.write(action+"\n")
        action_to_index[action] = action_index
        index_to_action[action_index] = action
        action_index += 1
        return
    for i in range(1,num_actions+1):
        init_actions(action+str(i))

init_actions("")
file.close()

state_action_reward = {}
with open("tamarix_rewards_for_all_states_action.txt", "r") as ins:
    for line in ins:
        ar = line.split(',')
        state = ar[0].replace(" ","")[1:-1]
        action = ar[1].replace(" ","")[1:-1]
        reward = float(ar[2])
        state_action_reward[(state,action)] = reward

#print(state_action_reward)

#print(action_to_index)

### construct & evaluate uncertainty with Multinomial + dirichlet distributed data points. Calculate L1 worstcase return

def evaluate_dirichlet_uncertainty(transition_mult, num_simulation, value_function,  confidence_level):
    """
    Compares Hoeffding and Bayesian methods for constructing uncertainty sets
    
    @returns list of results with tuples 
            (method_name, 
                error_fractions: error as a fraction of the true return, 
                L1 thresholds, 
                violations: fraction of instances in which the value is not a lower bound)
    """
    bayes_th = np.zeros(num_simulation)
    hoeff_th = np.zeros(num_simulation)
    tight_hoeff_th = np.zeros(num_simulation)
    em_th = np.zeros(num_simulation)
    incrementallyReplaceV_th = np.zeros(num_simulation)
    incrementallyAddV_th = np.zeros(num_simulation)

    bayes_nominalPoints = []
    hoeff_nominalPoints = []
    em_nominalPoints = []
    knownV_nominalPoints = []
    incrementallyReplaceV_nominalPoints = []
    incrementallyAddV_nomianlPoints = []
    
    num_next_states = num_total_states
    #num points is the number of points drawn from the true distribution to construct the 
    #multinomial distribution. 
    num_points = np.sum(transition_mult)

    prior = np.ones(num_next_states)

    for i in range(num_simulation):        
        # ** calculate simple bayesian threshold
        # sample transition points from the posterior Dirichlet distribution        
        dir_points = np.random.dirichlet(transition_mult, bayes_samples) # prior + 
        
        #print("true_distribution",true_distribution,"mult",mult)
        #print("dir_points",dir_points)

        # calc mean probability p_hat 
        # TODO: marek changed from: nominal_prob = np.mean(dir_points, axis=0)
        # TODO: that may not result in a valid probability distribution, take the mean of samples instead
        nominal_prob_bayes = np.mean(dir_points, axis=0)
        nominal_prob_bayes /= np.sum(nominal_prob_bayes)
        bayes_nominalPoints.append(nominal_prob_bayes)
        
        #print("evaluate_dirichlet_uncertainty 0 ",i)
        
        nominal_prob_freq = transition_mult / np.sum(transition_mult)
        hoeff_nominalPoints.append(nominal_prob_freq)
        
        #print("evaluate_dirichlet_uncertainty 1 ",i)
        
        # TODO: marek: delta is 1 - confidence
        # get uncertainty set & threshold
        bayes_th[i] = compute_bayesian_threshold(dir_points, nominal_prob_bayes, confidence_level)        
        
        #print("evaluate_dirichlet_uncertainty 2 ",i)
        
        # TODO: marek: delta is 1 - confidence
        # ** calculate threshold from hoeffding bound equation
        hoeff_th[i] = np.sqrt((2 / num_points )*np.log((2**num_next_states-2)/ (1-confidence_level) ))   
        
        # ** calculate the tight hoeffding bound
        tight_hoeff_th[i]= np.sqrt((2 / num_points )*np.log((num_next_states-1)/ (1-confidence_level) ))
        # TODO: marek needs to fige out whether this should be -1 or -2?
        
        #em_nominal, emthreshold = calc_EM_rand(dir_points, confidence_level, nominal_prob_bayes)
        #em_nominal /= np.sum(em_nominal)
        #em_nominalPoints.append(em_nominal)
        #print("nominal_prob_bayes",np.sum(nominal_prob_bayes), "em_nominal", np.sum(em_nominal))
        #em_th[i] = emthreshold
        
        #print("evaluate_dirichlet_uncertainty 3 ",i)
        
        ivf = construct_uset_known_value_function(dir_points, value_function, confidence_level)
        incrementallyReplaceV_th[i] = ivf[1]
        incrementallyReplaceV_nominalPoints.append(ivf[2])
        incrementallyAddV_th[i] = ivf[1]
        incrementallyAddV_nomianlPoints.append(ivf[2])
        
        #print("evaluate_dirichlet_uncertainty 4 ",i)
        
    return [(Methods.BAYES, np.mean(bayes_th), np.std(bayes_th), np.mean(bayes_nominalPoints, axis=0) ),\
            (Methods.CENTROID, 0, 0, np.mean(hoeff_nominalPoints, axis=0) ),\
            (Methods.HOEFF, np.mean(hoeff_th), np.std(hoeff_th), np.mean(hoeff_nominalPoints, axis=0) ),\
            (Methods.HOEFFTIGHT, np.mean(tight_hoeff_th), np.std(tight_hoeff_th),\
                np.mean(hoeff_nominalPoints, axis=0)),\
            (Methods.EM, np.mean(em_th), np.std(em_th), np.mean(em_nominalPoints, axis=0) ),\
            (Methods.INCR_REPLACE_V, np.mean(incrementallyReplaceV_th),\
                np.std(incrementallyReplaceV_th), np.mean(incrementallyReplaceV_nominalPoints,axis=0)),\
            (Methods.INCR_ADD_V, np.mean(incrementallyAddV_th),\
                np.std(incrementallyAddV_th), np.mean(incrementallyAddV_nomianlPoints,axis=0))]
            
###
def incrementally_add_V(valuefunctions, num_samples, num_simulation,\
                                                    num_update, sa_confidence, orig_sol):
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

    X = []
    Y = []
    
    valuefunctions = [valuefunctions]
    th_list = []
    list_transitions_points = {}
    
    for s in range(num_total_states):
        for a in range(num_total_actions): 
            transitions_points = np.random.dirichlet(transition_samples[s, a], bayes_samples)
            list_transitions_points[(s,a)] = transitions_points
    
    #Store the nominal points for each state-action pairs
    nomianl_points = {}
    
    #Store the latest nominal of nominal point & threshold
    nominal_threshold = {}
    under_estimate, real_regret = 0.0, 0.0
    
    for i in range(num_update):
        #print("valuefunctions",i,": ",valuefunctions)
        #keep track whether the current iteration keeps the mdp unchanged
        is_mdp_unchanged = True
        threshold = [[] for _ in range(3)]
        rmdp = crobust.MDP(0, discount_factor)
        for s in range(num_total_states):
            for a in range(num_total_actions): 
                dir_points = list_transitions_points[(s,a)]#np.asarray(transitions_points[a])

                res = construct_uset_known_value_function(dir_points, valuefunctions[-1],\
                                                            sa_confidence)
                
                if (s,a) not in nomianl_points:
                    nomianl_points[(s,a)] = []
                
                trp, th = None, 0
                #If there's a previously constructed L1 ball. Check whether the new nominal point
                #needs to be considered.
                if (s,a) in nominal_threshold:
                    old_trp, old_th = nominal_threshold[(s,a)][0], nominal_threshold[(s,a)][1]
                    
                    #Compute the L1 distance between the newly computed nominal point & the previous 
                    #nominal of nominal points
                    new_th = np.linalg.norm(res[2] - old_trp, ord = 1)
                    
                    #If the new point is inside the previous L1 ball, don't consider it & proceed with
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
                    
                    #Find the center of the L1 ball for the nominal points with different 
                    #value functions
                    trp, th = find_nominal_point(np.asarray(nomianl_points[(s,a)]))
                    nominal_threshold[(s,a)] = (trp, th)
                
                threshold[0].append(s)
                threshold[1].append(a)
                threshold[2].append(th)
                
                for next_st in range(num_total_states):
                    reward = state_action_reward[(index_to_state[s], index_to_action[a])]
                    rmdp.add_transition(s, a, next_st, trp[int(next_st)], reward)

        rsol = rmdp.rsolve_mpi(b"robust_l1",threshold)
        
        violation = 0

        #If the whole MDP is unchanged, meaning the new value function didn't change the uncertanty
        #set for any state-action, no need to iterate more!
        if is_mdp_unchanged or i==num_update-1:
            print("**** Add Values *****")
            print("MDP remains unchanged after number of iteration:",i)
            print("Policy",rsol.policy, "threshold", threshold)

            rpolicy = rsol.policy
            ret = est_true_mdp.solve_mpi(policy=rpolicy)
            under_estimate = np.dot(initial,ret.valuefunction) - np.dot(initial,rsol.valuefunction)
            
            #ropt_sol = rmdp.solve_mpi(policy=orig_sol.policy)
            real_regret = np.dot(initial,orig_sol.valuefunction) -\
                                                np.dot(initial,ret.valuefunction)
                                                
            violation = 1 if (np.dot(initial, ret.valuefunction) - np.dot(initial,\
                            rsol.valuefunction))<0 else 0                        
            break

        valuefunction = rsol.valuefunction
        valuefunctions.append(valuefunction)
        X.append(i)
        Y.append(valuefunction[0])

    return under_estimate, real_regret, violation

###
num_slots = nbrReaches*habitatSize
num_total_states = num_states**num_slots
num_total_actions = num_actions**nbrReaches
state_samples = np.zeros(num_states**num_slots)
transition_samples = np.ones( (num_total_states, num_total_actions, num_total_states) )
#reward_samples = np.zeros( (num_total_states, num_total_actions, num_total_states) )

#After the specified horizon, the simulator start again. The last state of the previous iteration 
#& the first state of the next iteration are not related.
sampled_horizon = 500

#Read Samples from Tamarisk simulator. get the multinomial distribution consdiering the samples
#from Tamarisk Simulator
count = 0
with open("tamarix_samples.txt", "r") as ins:
    for line in ins:
        if count%sampled_horizon==0:
            prev_state, prev_action, prev_reward = "", "", 0.0
        count += 1
        #print("line",line)
        s_a_r = line.split(',')
        cur_state = state_to_index[s_a_r[0].replace(" ","")[1:-1]]
        cur_action = action_to_index[s_a_r[1].replace(" ","")[1:-1]]
        cur_reward = float(s_a_r[2])
        
        state_samples[ cur_state ] += 1
        if prev_state is not "":
            transition_samples[ prev_state, prev_action, cur_state ] += 1.0
            #reward_samples[ prev_state, prev_action, cur_state ] = prev_reward
        
        prev_state, prev_action, prev_reward = cur_state, cur_action, cur_reward

for s in range(num_total_states):
    for a in range(num_total_actions):
        print(s,a,transition_samples[s,a])
            
#np.set_printoptions(threshold=np.nan)
#print("state_samples", state_samples, "transition_samples", transition_samples[prev_state, prev_action], "reward", reward_samples[prev_state, prev_action, cur_state])
  
### run experiments
if __name__ == "__main__":  
    discount_factor = 0.9  
    #/home/reazul/RobustRL/Code/RobustRL
    #Construct the estimated true MDP by taking a lot of samples.
    est_true_mdp = crobust.MDP(0, discount_factor)
    for cur_state in range(num_total_states):
        for a in range(num_total_actions):
            denom = np.sum(transition_samples[cur_state, a])
            trp = transition_samples[cur_state, a] / (denom if denom>0 else 1)
            for next_state in range(num_total_states):
                #if trp[next_state]>0:
                    #print(cur_state, a, next_state, trp[next_state], \
                    #            reward_samples[ cur_state, a, next_state ])
                est_true_mdp.add_transition(cur_state, a, next_state,\
                trp[next_state], state_action_reward[(index_to_state[cur_state], index_to_action[a])] ) #reward_samples[ cur_state, a, next_state ]
###
if __name__ == "__main__":
    orig_sol = est_true_mdp.solve_mpi()
    orig_policy = orig_sol.policy
    #print("orig_sol.valuefunction",orig_sol.valuefunction,"orig_sol.policy", orig_sol.policy, "len", len(orig_sol.policy))
    #est_true_mdp.
    #print(est_true_mdp.state_count())

    random_policy = np.random.randint(num_total_actions, size=(num_total_states))
    #np.random.randint(num_total_actions, size=(est_true_mdp.state_count()))
    arbitrary_valuefunction = est_true_mdp.solve_vi(policy=random_policy).valuefunction
    #print("random_policy",random_policy, "arbitrary_valuefunction", arbitrary_valuefunction)
    
###
if __name__ == "__main__":
    # number of sampling steps
    num_iterations = 5
    # number of runs
    num_simulation = 5
    sample_step = 3
    
    confidence_level = 0.9
    
    #max number of iterations to improve value functions
    num_update = 10
    
    initial = np.ones(num_total_states)/num_total_states
    
    #(1-overall_confidence) is the total violation allowed. This total violation is distributed among all the state action pairs
    # according to the Union bound.
    sa_confidence = 1 - ( (1 - confidence_level) / (num_total_actions * num_total_states) )
    
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)
    
    #In thresholds, the first dimension is methods (e.g Bayesian, EM etc.)
    #for each method, there are 3 lists containing state-action-threshold packed into a list
    thresholds = [ [[] for _ in range(3)] for _ in range(Methods.NUM_METHODS.value) ]
    under_estimation = [[] for _ in range(Methods.NUM_METHODS.value)] #estimated regret
    real_regret = [[] for _ in range(Methods.NUM_METHODS.value)] #optimal regret
    violations = [[] for _ in range(Methods.NUM_METHODS.value)]
    
    #sol = est_true_mdp.solve_mpi()
    #print("Start")
    for pos, num_samples in enumerate(tqdm.tqdm(sample_steps)):
        rmdps = []
        for m in range(Methods.NUM_METHODS.value):
            rmdps.append(crobust.MDP(0, discount_factor))
        
        for cur_state in range(num_total_states):
            for a in range(num_total_actions):   
                params = evaluate_dirichlet_uncertainty(transition_samples[cur_state, a], num_simulation, arbitrary_valuefunction,  confidence_level)
                #evaluate_uncertainty_set(s, num_samples, num_simulation, arbitrary_valuefunction, sa_confidence)
                #print(cur_state, a)
                for m in range(Methods.NUM_METHODS.value):
                    if LI_METHODS[m] is Methods.EM:
                        continue
                        
                    trp = params[m][3]
                    threshold = params[m][1]                
                    
                    #print("Method",LI_METHODS[m].value, trp)
                    
                    for next_st in range(num_total_states):
                        #reward = calc_reward(next_st, trp[a][int(next_st)], a)
                        rmdps[m].add_transition(cur_state, a, next_st, trp[int(next_st)],\
                                                            state_action_reward[(index_to_state[cur_state], index_to_action[a])])
                    thresholds[m][0].append(cur_state)
                    thresholds[m][1].append(a)
                    thresholds[m][2].append(threshold)
        #print(pos, "middle")
        for m in range(Methods.NUM_METHODS.value):
            if LI_METHODS[m] is Methods.EM:
                continue
            #if LI_METHODS[m] == Methods.BAYES:
                #print(Methods.BAYES.value," ", rmdps[m].to_json(), "thresholds: ", thresholds[m])
            rsol = rmdps[m].rsolve_mpi(b"robust_l1",np.asarray(thresholds[m]))
            """
            if LI_METHODS[m] is Methods.INCR_REPLACE_V:
                u_estimate, regret, violation = incrementally_replace_V(rsol.valuefunction,\
                                num_samples,num_simulation, num_update, sa_confidence, orig_sol)
                under_estimation[m].append(u_estimate)
                real_regret[m].append(regret)
                violations[m].append(violation)
            """
            if LI_METHODS[m] is Methods.INCR_ADD_V:
                u_estimate, regret, violation = incrementally_add_V(rsol.valuefunction, num_samples,\
                                        num_simulation, num_update, sa_confidence, orig_sol)
                under_estimation[m].append(u_estimate)
                real_regret[m].append(regret)
                violations[m].append(violation)
            else:
                under_estimation[m].append( np.dot(initial,orig_sol.valuefunction) -\
                                                np.dot(initial,rsol.valuefunction))
                ropt_sol = est_true_mdp.solve_mpi(policy=rsol.policy)
                real_regret[m].append( np.dot(initial,orig_sol.valuefunction) -\
                                                np.dot(initial,ropt_sol.valuefunction))
                violations[m].append( 1 if (np.dot(initial, ropt_sol.valuefunction) - \
                                        np.dot(initial, rsol.valuefunction)) < 0 else 0 )

### Plot results

generic_plot(sample_steps, under_estimation, "Number of samples", 'Calculated return error', legend_pos="upper right", figure_name="Generic_plot_Under_Estimation.pdf")

generic_plot(sample_steps, real_regret, "Number of samples", 'Calculated true regret', legend_pos="upper right", figure_name="Generic_plot_True_Regret.pdf")

generic_plot(sample_steps, violations, "Number of samples", 'Violations', legend_pos="upper right", figure_name="Generic_plot_violations.pdf")

