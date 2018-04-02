import Bayesian_Uncertainty_Set
import Gaussian_Uncertainty_Set
from craam import crobust
import Utils
import Plot
import numpy as np
import tqdm

initial, max_inventory, purchase_cost, sale_price = 0, 30, 2.0, 3.0,
prior_mean, prior_std, demand_std, rand_seed =  5.0, 2.0, 3, 3
num_iterations_for_vf = 10
horizon, runs = 20, 30
discount_factor = 0.9
num_samples = 30
tuple_size = 3 #s-a-th

###Inventory Simulation
if __name__ == "__main__":
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
    mdp = smdp.get_mdp(discount_factor)
    
    print("Original MDP: ", mdp.to_json())
    
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

### solve MDP
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

### Compute the reward for inventory problem
def compute_rewards(min_demand, max_demand, a, s, sale_price, purchase_cost):
    rewards = []
    for d in range(min_demand, max_demand+1):
        #Compute the next inventory level
        next_inventory = max(0, a + s - d)
        
        #Back calculate how many items were sold
        sold_amount = d-next_inventory + a
        
        #Compute the obtained revenue
        revenue = sold_amount * sale_price
        
        #Compute the expense
        expense = a * purchase_cost
        
        #Reward is equivalent to the profit & obtained from 
        #revenue & total expense
        reward = revenue - expense
        
        rewards.append((next_inventory,reward))
    return rewards
    
def construct_rmdp(s, a, value_functions, rmdp, is_multiple_v=False):
    rewards = compute_rewards(min_demand, max_demand, a, s,\
        sale_price, purchase_cost)

    #Computes the return, threshold, nominal point etc. for 
    #current state & action
    guk = evaluate_gaussian_knownV(num_samples, sa_confidence, runs, value_functions, min_demand, max_demand, demand_mean_prior_mean, demand_mean_prior_std, true_demand_std)
    
    trp = None
    th = 0
    if is_multiple_v:
        #Find the center of the L1 ball for the nominal points with different value functions
        trp = find_nominal_point(guk[2])
        
        #Find the maximum distance from center of the L1 ball to the nominal points
        #This is the size of the L1 ball, set it as threshold
        th = get_uset(guk[2], trp, len(guk[2]))[1]
    else:
        th = guk[0] # threshold is 0    
        trp = guk[2][0] #index 0 means there's only one value function

    for k in range(len(rewards)):
        rmdp.add_transition(s, a, rewards[k][0], trp[k], rewards[k][1])

    return rmdp, th

### Iteratively improve over when an initial value function is known
def incrementally_improve_V(vf, threshold):
    is_multiple_v = False

    value_functions = [[]]
    value_functions[0].append(vf[0])
    for v in range(max_demand+1,0,-1):
        value_functions[0].append(vf[-v])
    
    #this loop iterates incrementally with the latest value function 
    #to further improve upon
    for i in range(num_iterations_for_vf):
        #print("iterative vf",value_functions)
        
        threshold = np.zeros((tuple_size, action_count))
        rmdp = crobust.MDP(0, discount_factor)

        pos=0
        for s in range(mdp.state_count()):
            actions = mdp.action_count(s)
            for a in range(actions):

                if len(mdp.get_toids(s,a))==0:
                    continue
                
                threshold[0,pos] = s
                threshold[1,pos] = a
                
                rmdp, threshold[2,pos] = construct_rmdp(s, a, value_functions, rmdp, is_multiple_v) #pythonic way of passing by reference. modify inside outer function, return modified object & reassign.
                pos += 1

        #print("MDP: ",rmdp.to_json())

        sol = rmdp.rsolve_vi("robust_l1".encode(),threshold)
        vf = sol.valuefunction
        value_functions = [[]]

        #As S,s policy is used as random policy to generate samples, 
        #the possible inventory levels are, o & (max_inventory - max_demand) 
        #to max_inventory, inclusive. So filter out the value functions
        #for possible states.
        value_functions[0].append(vf[0])
        for v in range(max_demand+1,0,-1):
            value_functions[0].append(vf[-v])
    return value_functions[0][0] #initial states value of 0th value function

### Improve with random new value functions
def randomly_improve_V(vf, threshold):
    is_multiple_v = True

    value_functions = [[]]
    value_functions[0].append(vf[0])
    for v in range(max_demand+1,0,-1):
        value_functions[0].append(vf[-v])

    #this loop iterates incrementally with the latest value function 
    #to further improve upon
    for i in range(num_iterations_for_vf):
        #print("random vf",value_functions)
        
        #threshold is set to zero & will be zero finally because construct_uset_known_value_function\
        #picks the robust value with threshold zero
        threshold = np.zeros((tuple_size, action_count))
    
        value_functions.append(np.random.randint(10, size=(max_demand-min_demand+2))) 
        rmdp = crobust.MDP(0, discount_factor)
        pos=0
        
        for s in range(mdp.state_count()):
            actions = mdp.action_count(s)
            for a in range(actions):

                if len(mdp.get_toids(s,a))==0:
                    continue
                threshold[0,pos] = s
                threshold[1,pos] = a
                rmdp, threshold[2,pos] = construct_rmdp(s, a, value_functions, rmdp, is_multiple_v) #pythonic way of passing by reference. modify inside outer function, return modified object & reassign.
                pos += 1
        #print("MDP: ",rmdp.to_json())

        sol = rmdp.rsolve_vi("robust_l1".encode(),threshold)
        vf = sol.valuefunction
        
    return vf[0]
    
###
        value_function = []

        #As S,s policy is used as random policy to generate samples, 
        #the possible inventory levels are, o & (max_inventory - max_demand) 
        #to max_inventory, inclusive. So filter out the value functions
        #for possible states.
        value_function.append(vf[0])
        for v in range(max_demand+1,0,-1):
            value_function.append(vf[-v])

### Compare methods for MDP with multiple states & actions
if __name__ == "__main__":
    # number of sampling steps
    num_iterations = 30
    # number of runs
    runs = 20
    sample_step = 5

    gauss_results = []
    known_ValueFunction = np.random.uniform(low=0, high=10, size=(max_demand-min_demand+1))
    improve_ValueFunction, addRandom_ValueFunction = known_ValueFunction, [known_ValueFunction]
    
    print("known_ValueFunction",known_ValueFunction)
    
    sample_steps = np.arange(sample_step,sample_step*num_iterations+1, step = sample_step)
    
    #In thresholds, the first dimension is methods (e.g Bayesian, EM etc.)
    #for each method, there are 3 lists containing state-action-threshold packed into a list
    thresholds = [ [[] for _ in range(3)] for _ in range(Methods.NUM_METHODS.value) ]
    calc_return = [[] for _ in range(Methods.NUM_METHODS.value)]
    
    for pos, num_samples in tqdm.tqdm(enumerate(tqdm.tqdm(sample_steps))):
        
        rmdps = []
        for m in range(Methods.NUM_METHODS.value):
            rmdp = crobust.MDP(0, discount_factor)
            rmdps.append(rmdp)
        
        for s in range(mdp.state_count()):
            actions = mdp.action_count(s)
            for a in range(actions):
                
                rewards = compute_rewards(min_demand, max_demand, a, s, sale_price, purchase_cost)
                
                gu = evaluate_gaussian_uncertainty(num_samples, sa_confidence, runs,\
                known_ValueFunction, improve_ValueFunction, addRandom_ValueFunction,\
                min_demand, max_demand, demand_mean_prior_mean, demand_mean_prior_std,\
                true_demand_std)
                
                for m in range(Methods.NUM_METHODS.value):
                    trp = gu[m][7]
                    th = gu[m][2]
                    if LI_METHODS[m] is Methods.ADDRANDOMV:
                        trp = trp[0]
                        th = th[0]
                        
                    #print(s,a,"trp:",len(trp), "len(rewards)", len(rewards))
                    for k in range(len(rewards)):
                        rmdps[m].add_transition(s, a, rewards[k][0], trp[k], rewards[k][1])
                    thresholds[m][0].append(s)
                    thresholds[m][1].append(a)
                    thresholds[m][2].append(th)

        for m in range(Methods.NUM_METHODS.value):
            #print(m,thresholds[m])
            sol = rmdps[m].rsolve_vi("robust_l1".encode(),np.asarray(thresholds[m]))
            #print(sol.valuefunction)
            if LI_METHODS[m] is Methods.IMPROVEV:
                calc_return[m].append(incrementally_improve_V(sol.valuefunction,thresholds[m]))
            elif LI_METHODS[m] is Methods.ADDRANDOMV:
                calc_return[m].append(randomly_improve_V(sol.valuefunction,thresholds[m]))
            else:
                calc_return[m].append(sol.valuefunction[0])
            #print(LI_METHODS[m].value,sol.valuefunction)

###
print(calc_return)
generic_plot(sample_steps, calc_return, "Number of samples", "Returned value to initial state")

###
    print(X,Y)
    #print("knownV_nominal_points",knownV_nominal_points[(0,0)][0])
    simple_generic_plot(X, Y, "Iteration over Value Function", "Return on the initial state", "lower right", "Number of interations vs. return", "MDP_return_1")

