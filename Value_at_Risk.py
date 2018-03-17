import Utils

### compute V@R
def value_at_risk(values, confidence):
    return np.percentile(values, 1-confidence)

### V@R Dirichlet
def evaluate_var_Dirichlet(num_points, num_next_states, reward, confidence_level, num_simulation, threshold):
    bayes_ret = np.zeros(num_simulation)
    hoeff_ret = np.zeros(num_simulation)
    tight_hoeff_ret = np.zeros(num_simulation)
    em_ret = np.zeros(num_simulation)
    
    # number of samples of the true distribution to take when estimating the Bayes samples
    bayes_samples = 25
    prior = np.ones(num_next_states)

    for i in range(num_simulation):
        # construct the true distribution
        true_distribution = np.random.dirichlet(prior, 1)[0]

        # get samples from multinomial distribution
        mult = np.random.multinomial(num_points, true_distribution)

        dir_points = np.random.dirichlet(mult + prior, bayes_samples) 
        
        nominal_prob_bayes = np.mean(dir_points, axis=0)
        nominal_prob_bayes /= np.sum(nominal_prob_bayes)
        nominal_prob_freq = mult / np.sum(mult)

        em_nominal, emthreshold = calc_EM_rand(dir_points, confidence_level, nominal_prob_bayes)
        em_nominal /= np.sum(em_nominal)
        
        #construct_uset_known_value_function(dir_points, reward, confidence_level)
        
        true_ret = true_distribution @ reward
        
        bayes_ret[i] = crobust.worstcase_l1(reward, nominal_prob_bayes, threshold)
        hoeff_ret[i] = crobust.worstcase_l1(reward, nominal_prob_freq, threshold)
        tight_hoeff_ret[i] = crobust.worstcase_l1(reward, nominal_prob_freq, threshold)
        em_ret[i] = crobust.worstcase_l1(reward, em_nominal, threshold)
    
    #print("Bayes Ret",bayes_ret)
    
    bayes_var = value_at_risk(bayes_ret, confidence_level)   
    """
    hoeff_var = value_at_risk(hoeff_ret, confidence_level)
    tight_hoeff_var = value_at_risk(tight_hoeff_ret, confidence_level)
    em_var = value_at_risk(em_ret, confidence_level)
    """
    return ("bayes_simple", bayes_var, np.mean(bayes_ret), threshold, true_ret)
"""
    return [("bayes_simple", bayes_var, np.mean(bayes_ret), threshold, true_ret),\
            ("hoeffding", hoeff_var, np.mean(hoeff_ret), threshold, true_ret),\
            ("hoeffding_tight", tight_hoeff_var, np.mean(tight_hoeff_ret), threshold, true_ret),\
            ("mm", em_var, np.mean(em_ret), threshold, true_ret)]
"""

threshold_values = np.arange(0,1.1,0.3)
num_actions = 5
num_samples = 5 #number of samples to approximate the true distribution, replacing the sample_steps from before
num_iters = 5
confidence_level = 0.9
results_var = []

for pos, val in enumerate(tqdm.tqdm(threshold_values)):
    policy_results = []
    for _ in range(num_iters):
        val_ret = []
        for action in range(num_actions):
            val_ret.append(evaluate_var_Dirichlet(num_samples, num_next_states, reward, confidence_level, runs, val))
        policy = np.argmax([x[2] for x in val_ret])
        #print("policy",policy)
        policy_results.append(val_ret[policy])
    var = value_at_risk([x[4] for x in policy_results] ,confidence_level)
    expected_ret = np.mean([x[2] for x in policy_results])
    print("var", var, "expected_ret", expected_ret, "threshold", val)
    results_var.append( (var, expected_ret, val) )
print(results_var)


###V@R Gaussian
def evaluate_var_Gaussian(num_samples, confidence_level, num_simulation, threshold):
    
    # prior distribution over the demand mean.
    demand_mean_prior_mean = 50
    demand_mean_prior_std = 15
    
    # demand parameters
    true_demand_std = 25
    
    # discretization of demands
    min_demand = 0 
    max_demand = 100
    demand_values = np.arange(min_demand, max_demand + 1, dtype=np.double)
    # number of next steps depends on the demands
    num_next_states = max_demand - min_demand + 1
    # rewards - an increasing sequence
    reward = np.arange(min_demand, max_demand + 1, dtype=np.double)
    
    bayes_ret = np.zeros(num_simulation)
    hoeff_ret = np.zeros(num_simulation)
    tight_hoeff_ret = np.zeros(num_simulation)
    em_ret = np.zeros(num_simulation)
    
    # number of samples of the true distribution to take when estimating the Bayes samples
    bayes_samples = 25

    for i in range(num_simulation):
        # construct the true distribution
        true_demand_mean = np.random.normal(demand_mean_prior_mean, demand_mean_prior_std)

        # construct the true distribution
        true_distribution = discretize_gaussian(min_demand, max_demand, true_demand_mean, true_demand_std)

        # get samples from multinomial distribution, 3 next states with uniform transition kernel
        mult = np.random.multinomial(num_samples, true_distribution)
        
        # sample from the posterior over samples
        # *** this is the main point of difference ***
        # !!! assume that the state index is the demand !!!
        # this is the distribution over the mean of the demand! not the distribution of the demand
        estmean_demand_mean, estmean_demand_std = normal_aposteriori(demand_values, mult, \
                                        true_demand_std, demand_mean_prior_mean, demand_mean_prior_std)
                             
        
        dir_points = np.array(\
            [discretize_gaussian(min_demand, max_demand, 
                    np.random.normal(estmean_demand_mean, estmean_demand_std), true_demand_std) \
                    for k in range(bayes_samples)])
        
        nominal_prob_bayes = np.mean(dir_points, axis=0)
        nominal_prob_bayes /= np.sum(nominal_prob_bayes)
        nominal_prob_freq = mult / np.sum(mult)
        
        em_nominal, emthreshold = calc_EM_rand(dir_points, confidence_level, nominal_prob_bayes)
        em_nominal /= np.sum(em_nominal)
                      
        true_ret = true_distribution @ reward
        
        bayes_ret[i] = crobust.worstcase_l1(reward, nominal_prob_bayes, threshold)
        hoeff_ret[i] = crobust.worstcase_l1(reward, nominal_prob_freq, threshold)
        tight_hoeff_ret[i] = crobust.worstcase_l1(reward, nominal_prob_freq, threshold)
        em_ret[i] = crobust.worstcase_l1(reward, em_nominal, threshold)
    
    print("Bayes Ret",bayes_ret)
    
    portfolio_val = np.sum(reward)
    #Compute value at risk
    quantile = 1 - confidence_level
    bayes_var = var_cov_var(portfolio_val, confidence_level, np.mean(bayes_ret), np.std(bayes_ret))
    #np.percentile(np.sort(bayes_ret),quantile)
    hoeff_var = var_cov_var(portfolio_val, confidence_level, np.mean(hoeff_ret), np.std(hoeff_ret))
    #np.percentile(hoeff_ret,quantile)
    tight_hoeff_var = var_cov_var(portfolio_val, confidence_level, np.mean(tight_hoeff_ret), np.std(tight_hoeff_ret))
    #np.percentile(tight_hoeff_ret,quantile)
    em_var = var_cov_var(portfolio_val, confidence_level, np.mean(em_ret), np.std(em_ret))
    #np.percentile(em_ret,quantile)
    
    return [("bayes_simple", bayes_var, np.mean(bayes_ret), threshold),\
            ("hoeffding", hoeff_var, np.mean(hoeff_ret), threshold),\
            ("hoeffding_tight", tight_hoeff_var, np.mean(tight_hoeff_ret), threshold),\
            ("em", em_var, np.mean(em_ret), threshold)]

threshold_values = np.arange(0,2.1,0.1)
num_samples = 20 #number of samples to approximate the true distribution, replacing the sample_steps from before
results_var = []
for pos, val in enumerate(tqdm.tqdm(threshold_values)):
    results_var.append(evaluate_var_Gaussian(num_samples, confidence_level, runs, val))
print(results_var)
