import Bayesian_Uncertainty_Set
import Gaussian_Uncertainty_Set
from craam import crobust
import Utils
import Plot
import numpy as np
import tqdm

initial, max_inventory, purchase_cost, sale_price = 0, 50, 2.0, 3.0,
prior_mean, prior_std, demand_std, rand_seed =  10.0, 5.0, 6.0, 3
horizon, runs = 5, 10
discount_factor = 0.9
num_samples = 5

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
    
def construct_rmdp(position, value_functions, rmdp, is_multiple_v=False):
    rewards = compute_rewards(min_demand, max_demand, a, s,\
        sale_price, purchase_cost)

    #Computes the return, threshold, nominal point etc. for 
    #current state & action
    guk = evaluate_gaussian_knownV(num_samples, sa_confidence, runs, value_functions, min_demand, max_demand, demand_mean_prior_mean, demand_mean_prior_std, true_demand_std)

    #Stack state-action-threshold to pass to the mdp solver for robust solution
    knownV_threshold[0,position] = s
    knownV_threshold[1,position] = a
    
    trp = None
    if is_multiple_v:
        #Find the center of the L1 ball for the nominal points with different value functions
        trp = find_nominal_point(guk[2])
        
        #Find the maximum distance from center of the L1 ball to the nominal points
        #This is the size of the L1 ball, set it as threshold
        knownV_threshold[2,position] = get_uset(guk[2], trp, len(guk[2]))[1]
    else:
        knownV_threshold[2,position] = guk[0] # threshold is 0    
        trp = guk[2][0] #index 0 means there's only one value function

    for k in range(len(rewards)):
        rmdp.add_transition(s, a, rewards[k][0], trp[k], rewards[k][1])
    position+=1
    return rmdp

### Improve over when an initial value function is known
if __name__ == "__main__":
    #initially assign random value function to each state
    value_function = [np.random.randint(10, size=(max_demand-min_demand+2))]
    #print(len(value_function))
    tuple_size = 3 #s-a-th
    is_multiple_v = False

    knownV_threshold = np.zeros((tuple_size, action_count))
    #knownV_nominal_points = {}

    X = []
    Y = []

    #this loop iterates incrementally with the latest value function 
    #to further improve upon
    for i in tqdm.tqdm(range(5)):
        
        rmdp = crobust.MDP(0, discount_factor)
        
        #print(knownV_threshold)
        position=0
        for s in range(mdp.state_count()):
            actions = mdp.action_count(s)
            for a in range(actions):

                if len(mdp.get_toids(s,a))==0:
                    continue
                    
                rmdp = construct_rmdp(position, value_function, rmdp, is_multiple_v) #pythonic way of passing by reference. modify inside outer function, return modified object & reassign.
                position += 1

        #print("MDP: ",rmdp.to_json())

        sol = rmdp.rsolve_vi("robust_l1".encode(),knownV_threshold)
        vf = sol.valuefunction
        value_function = [[]]

        #As S,s policy is used as random policy to generate samples, 
        #the possible inventory levels are, o & (max_inventory - max_demand) 
        #to max_inventory, inclusive. So filter out the value functions
        #for possible states.
        value_function[0].append(vf[0])
        for v in range(max_demand+1,0,-1):
            value_function[0].append(vf[-v])
        X.append(i)
        Y.append(vf[0])
        print(i,"value_function",vf,value_function)
###
    print(X,Y)
    #print("knownV_nominal_points",knownV_nominal_points[(0,0)][0])
    simple_generic_plot(X, Y, "Iteration over Value Function", "Return on the initial state", "lower right", "Number of interations vs. return", "MDP_return_1")

### The case when we have multiple value functions & multiple state MDP
if __name__ == "__main__":
    num_v = 20
    value_functions = []
    tuple_size = 3 #s-a-th
    is_multiple_v = True

    knownV_threshold = np.zeros((tuple_size, action_count))
    #knownV_nominal_points = {}

    X = []
    Y = []

    #this loop iterates incrementally with the latest value function 
    #to further improve upon
    for i in tqdm.tqdm(range(num_v)):
        
        value_functions.append(np.random.randint(10, size=(max_demand-min_demand+2))) 
        rmdp = crobust.MDP(0, discount_factor)
        position=0
        
        for s in range(mdp.state_count()):
            actions = mdp.action_count(s)
            for a in range(actions):

                if len(mdp.get_toids(s,a))==0:
                    continue
                
                rmdp = construct_rmdp(position, value_functions, rmdp, is_multiple_v) #pythonic way of passing by reference. modify inside outer function, return modified object & reassign.
                position += 1
        #print("MDP: ",rmdp.to_json())

        sol = rmdp.rsolve_vi("robust_l1".encode(),knownV_threshold)
        vf = sol.valuefunction
        value_function = []

        #As S,s policy is used as random policy to generate samples, 
        #the possible inventory levels are, o & (max_inventory - max_demand) 
        #to max_inventory, inclusive. So filter out the value functions
        #for possible states.
        value_function.append(vf[0])
        for v in range(max_demand+1,0,-1):
            value_function.append(vf[-v])
        X.append(i+1)
        Y.append(vf[0])
        print(i,"value_function",vf,value_function)

###    

print(X,Y)
simple_generic_plot(X, Y, "Number of Value Functions", "Return on the initial state", "lower right", "Number of value functions vs. return", "MDP_return_1")

###
                if (s,a) not in knownV_nominal_points:
                    knownV_nominal_points[(s,a)] = []

                if len(knownV_nominal_points[(s,a)])>=2:
                    #find the nominal point of all the nominal points
                    nominalp_of_nominal = find_nominal_point(np.asarray\
                                            (knownV_nominal_points[(s,a)]))
                    #uset, threshold = get_uset(np.asarray(knownV_nominal_points[(s,a)]), np.asarray(nominalp_of_nominal), len(knownV_nominal_points[(s,a)]))

                    #Compute the distance between the current nominal point & 
                    #the nominal point of all the previous nominal points
                    #dist = np.linalg.norm(guk[0][7] - nominalp_of_nominal, ord = 1)

                    #if the new nominal point lies inside the previously
                    #constructed l1-ball, a reasonable estimation is found 
                    #& continue without updating the threshold for this 
                    #state-action
                    if dist<threshold:
                        #print("dist<threshold",s,a,knownV_threshold[2,position])
                        position+=1
                        continue
                       
                       
                        
                rewards = compute_rewards(min_demand, max_demand, a, s,\
                sale_price, purchase_cost)

                #Computes the return, threshold, nominal point etc. for 
                #current state & action
                guk = evaluate_gaussian_knownV(num_samples, sa_confidence, runs, value_functions, min_demand, max_demand, demand_mean_prior_mean, demand_mean_prior_std, true_demand_std)
                
                #print(guk[2])
                nominalp_of_nominal = find_nominal_point(guk[2])
                trp = nominalp_of_nominal #guk[2]


                for k in range(len(rewards)):
                    rmdp.add_transition(s, a, rewards[k][0], trp[k], rewards[k][1])