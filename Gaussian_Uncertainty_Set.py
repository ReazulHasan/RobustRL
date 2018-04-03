import random
import math
import numpy as np
from scipy.stats import norm
from craam import crobust
from scipy import stats

### Scripts for computing with Gaussian distribution


def discretize_gaussian(min_value, max_value, mean, std):
    """ 
    Computes a discrete approximation of the Gaussian distribution. The distribution
    is bounded by min_value and max_value.
    Both bounds are inclusive.
    """
    d = stats.norm(mean, std)
    dist = np.array([d.pdf(v) for v in range(min_value, max_value + 1)])
    # normalize it
    dist /= np.sum(dist)
    return dist
    
def normal_aposteriori(values, weights, std, prior_mean, prior_std):
    """ 
    Estimates the aposteriori Gaussian distribution:
    see: https://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions
    
    Assumes that the standard deviation of the demands is known but the 
    mean is distributed according to the prior Gaussian distribution. std is the
    standard deviation of the demand (known).
    
    The parameter weights represents an un-normalized weight on each value. 
    """
    n = np.sum(weights)
    sum = np.dot(weights, values)
    precision = (1 / prior_std**2 + n / std**2)
    expected_mean = (prior_mean / prior_std**2 + sum / std**2) / precision
    expected_std = 1/precision
    return expected_mean, expected_std

###
print(discretize_gaussian(0, 10, 3, 2))

### construct & evaluate uncertainty with Gaussian distributed data points. Calculate L1 worstcase return
def evaluate_gaussian_uncertainty(num_samples, confidence_level, num_simulation, known_ValueFunction, improve_ValueFunction, addRandom_ValueFunction, min_demand,\
                        max_demand, demand_mean_prior_mean = 50, demand_mean_prior_std = 15, true_demand_std = 25):
    """
    Runs the evaluation assuming that the next state represents a demand level
    and that it is distributed according to a normal distribution. The prior on
    the mean of this distribution is also Gaussian, and the standard deviation is known.
    
    @param num_samples Number of samples from the multinomial distribution
    @param confidence_level required confidence level
    @param num_simulation number of simulations
    @param min_demand minimum demand level
    @param max_demand maximum demand level
    @param demand_mean_prior_mean prior demand mean obtained from the inventory samples
    @param demand_mean_prior_std prior demand std obtained from the inventory samples
    @param true_demand_std known true std of the demand
    
    @return bayes_return, bayes_threshold, hoeffding_return, hoeffding_threshold
    """     
    
    demand_values = np.arange(min_demand, max_demand + 2, dtype=np.double)
    # number of next steps depends on the demands
    num_next_states = max_demand - min_demand + 2
    # rewards - an increasing sequence
    reward = np.arange(min_demand, max_demand + 2, dtype=np.double)
    
    bayes_th = np.zeros(num_simulation)
    bayes_ret = np.zeros(num_simulation)
    bayes_ret_err = np.zeros(num_simulation)
    
    hoeff_th = np.zeros(num_simulation)
    hoeff_ret = np.zeros(num_simulation)
    hoeff_ret_err = np.zeros(num_simulation)
    
    tight_hoeff_th = np.zeros(num_simulation)
    tight_hoeff_ret = np.zeros(num_simulation)
    tight_hoeff_ret_err = np.zeros(num_simulation)
    
    em_th = np.zeros(num_simulation)
    em_ret = np.zeros(num_simulation)
    em_ret_err = np.zeros(num_simulation)
    
    knownV_th = np.zeros(num_simulation)
    knownV_ret = np.zeros(num_simulation)
    knownV_ret_err = np.zeros(num_simulation)
    
    num_v = len(addRandom_ValueFunction)
    
    improveV_th = np.zeros(num_simulation) #np.zeros((num_v, num_simulation))
    improveV_ret = np.zeros(num_simulation) #np.zeros(num_simulation)
    improveV_ret_err = np.zeros(num_simulation) #np.zeros(num_simulation)
    
    addRandomV_th = np.zeros((num_v, num_simulation))
    addRandomV_ret = np.zeros((num_v, num_simulation))
    addRandomV_ret_err = np.zeros((num_v, num_simulation))
    
    bayes_nominalPoints = []
    em_nominalPoints = []
    knownV_nominalPoints = []
    improveV_nominalPoints = []
    addRandomV_nomianlPoints = [ [] for _ in range(num_v)]
    
    # number of samples of the true distribution to take when estimatng the Bayes samples
    bayes_samples = 25
    
    accumulate_dir_points = []

    for i in range(num_simulation):
        # construct the true distribution
        true_demand_mean = np.random.normal(demand_mean_prior_mean, demand_mean_prior_std)

        # construct the true distribution
        true_distribution = discretize_gaussian(min_demand, max_demand+1, true_demand_mean, true_demand_std)

        # get samples from multinomial distribution, 3 next states with uniform transition kernel
        mult = np.random.multinomial(num_samples, true_distribution)
        
        # sample from the posterior over samples
        # *** this is the main point of difference ***
        # !!! assume that the state index is the demand !!!
        # this is the distribution over the mean of the demand! not the distribution of the demand
        estmean_demand_mean, estmean_demand_std = normal_aposteriori(demand_values, mult, \
                                    true_demand_std, demand_mean_prior_mean, demand_mean_prior_std)

        dir_points = np.array(\
            [discretize_gaussian(min_demand, max_demand+1, 
                    np.random.normal(estmean_demand_mean, estmean_demand_std), true_demand_std) \
                    for k in range(bayes_samples)])
        
        accumulate_dir_points.append(dir_points)
        #print("Gaussian: ",min_demand, max_demand, dir_points[0])

        # calc mean probability p_hat 
        # TODO: marek changed from: nominal_prob = np.mean(dir_points, axis=0)
        # TODO: that may not result in a valid probability distribution, take the mean of samples instead
        nominal_prob_bayes = np.mean(dir_points, axis=0)
        nominal_prob_bayes /= np.sum(nominal_prob_bayes)
        # TODO: marek: also tried but does not seem to work
        nominal_prob_freq = mult / np.sum(mult)
        
        bayes_nominalPoints.append(nominal_prob_bayes)
        
        #get uncertainty set & threshold
        bayes_th[i] = compute_bayesian_threshold(dir_points,nominal_prob_bayes, confidence_level)        
        
        #calc threshold from hoeffding bound equation
        hoeff_th[i] = np.sqrt((2 / num_samples )*np.log((2**num_next_states-2) / (1-confidence_level) ))   
        
        # ** calculate the tight hoeffding bound
        tight_hoeff_th[i]= np.sqrt((2 / num_samples )*np.log((num_next_states-1) / (1 - confidence_level) ))   # TODO: should be -1 or -2?
        
        em_nominal, emthreshold = calc_EM_rand(dir_points, confidence_level, nominal_prob_bayes)
        em_nominal /= np.sum(em_nominal)
        em_th[i] = emthreshold
        
        em_nominalPoints.append(em_nominal)
        
        knownV = construct_uset_known_value_function(dir_points, known_ValueFunction, confidence_level)
        knownV_th[i] = knownV[1]#[0]
        knownV_nominalPoints.append(knownV[2])#[0])
        
        ivf = construct_uset_known_value_function(dir_points, improve_ValueFunction, confidence_level)
        improveV_th[i] = ivf[1]
        improveV_nominalPoints.append(ivf[2])
        
        true_ret = true_distribution @ known_ValueFunction
        
        for vi, value_function in enumerate(addRandom_ValueFunction):
            rvf = construct_uset_known_value_function(dir_points, value_function,\
            confidence_level)
            addRandomV_th[vi,i] = rvf[1]
            addRandomV_nomianlPoints[vi].append(rvf[2])
            addRandomV_ret[vi,i] = rvf[0]
            addRandomV_ret_err[vi, i] = (true_ret - addRandomV_ret[vi,i])/true_ret
            
        bayes_ret[i] = crobust.worstcase_l1(known_ValueFunction, nominal_prob_bayes, bayes_th[i])
        hoeff_ret[i] = crobust.worstcase_l1(known_ValueFunction, nominal_prob_freq, hoeff_th[i])
        tight_hoeff_ret[i] = crobust.worstcase_l1(known_ValueFunction, nominal_prob_freq, tight_hoeff_th[i])
        em_ret[i] = crobust.worstcase_l1(known_ValueFunction, em_nominal, em_th[i])
        knownV_ret[i] = knownV[0]
        improveV_ret[i] = ivf[0]
        #addRandomV_ret[i] = rvf[0]
        
        #print("addRandomV_ret", addRandomV_ret)
        
        bayes_ret_err[i] = (true_ret - bayes_ret[i]) / true_ret
        hoeff_ret_err[i] = (true_ret - hoeff_ret[i]) /true_ret
        tight_hoeff_ret_err[i] = (true_ret - tight_hoeff_ret[i]) /true_ret
        em_ret_err[i] = (true_ret - em_ret[i]) /true_ret
        knownV_ret_err[i] = (true_ret - knownV_ret[i])/true_ret
        improveV_ret_err[i] = (true_ret - improveV_ret[i])/true_ret
        
    #print("np.mean(addRandomV_ret[:,1]:",np.mean(addRandomV_ret[:,1]))
    
    # make sure to not count negative return errors to improve the mena
    return [(Methods.BAYES, np.mean(np.maximum(0,bayes_ret_err)), np.mean(bayes_th), np.mean(bayes_ret_err < 0), np.mean(bayes_ret), np.std(np.maximum(0,bayes_ret_err)), np.std(bayes_th), nominal_prob_bayes ),\
            (Methods.HOEFF, np.mean(np.maximum(0,hoeff_ret_err)), np.mean(hoeff_th), np.mean(hoeff_ret_err < 0), np.mean(hoeff_ret), np.std(np.maximum(0,hoeff_ret_err)), np.std(hoeff_th), nominal_prob_bayes ),\
            (Methods.HOEFFTIGHT, np.mean(np.maximum(0,tight_hoeff_ret_err)), np.mean(tight_hoeff_th), np.mean(tight_hoeff_ret_err < 0), np.mean(tight_hoeff_ret), np.std(np.maximum(0,tight_hoeff_ret_err)), np.std(tight_hoeff_th), nominal_prob_bayes ),\
            (Methods.EM, np.mean(np.maximum(0,em_ret_err)), np.mean(em_th), np.mean(em_ret_err < 0), np.mean(em_ret), np.std(np.maximum(0,em_ret_err)), np.std(em_th), em_nominal ),\
            (Methods.KNOWNV, np.mean(np.maximum(0,knownV_ret_err)), np.mean(knownV_th), np.mean(knownV_ret_err<0), np.mean(knownV_ret), np.std(np.maximum(0,knownV_ret_err)), np.std(knownV_th), np.mean(knownV_nominalPoints,axis=0)),\
            (Methods.IMPROVEV, np.mean(np.maximum(0,improveV_ret_err)), np.mean(improveV_th), np.mean(improveV_ret_err<0), np.mean(improveV_ret), np.std(np.maximum(0,improveV_ret_err)), np.std(improveV_th), np.mean(improveV_nominalPoints,axis=0)),\
            (Methods.ADDRANDOMV, np.mean(np.maximum(0,addRandomV_ret_err)), np.mean(addRandomV_th, axis=1), np.mean(addRandomV_ret_err<0), np.mean(addRandomV_ret, axis=1), np.std(np.maximum(0,addRandomV_ret_err)), np.std(addRandomV_th,axis=1), np.mean(addRandomV_nomianlPoints,axis=1))], np.mean(accumulate_dir_points, axis=0)


### construct & evaluate uncertainty with Gaussian Distribution & Known Value Function
def evaluate_gaussian_knownV(num_samples, confidence_level, num_simulation, value_functions, min_demand = 0, max_demand = 100, demand_mean_prior_mean = 50, demand_mean_prior_std = 15, true_demand_std = 25):
    """
    Runs the evaluation assuming that the next state represents a demand level
    and that it is distributed according to a normal distribution, & the value function for the next states is known. The prior on
    the mean of this distribution is also Gaussian, and the standard deviation is known.
    
    @param num_samples Number of samples from the multinomial distribution
    @param confidence_level required confidence level
    @param num_simulation number of simulations
    @param min_demand minimum demand level
    @param max_demand maximum demand level
    @param demand_mean_prior_mean prior demand mean obtained from the inventory samples
    @param demand_mean_prior_std prior demand std obtained from the inventory samples
    @param true_demand_std known true std of the demand
    
    @return KnownV paramters
    """     
    
    demand_values = np.arange(min_demand, max_demand + 2, dtype=np.double)
    
    # number of samples of the true distribution to take when estimatng the Bayes samples
    bayes_samples = 25
    num_v = len(value_functions)
    
    knownV_th = np.zeros((num_v, num_simulation))
    #knownV_ret = np.zeros((num_v, num_simulation))
    KnownV_nomianl_point = [ [] for _ in range(num_v)]#np.zeros(num_simulation)
    
    for i in range(num_simulation):
        # construct the true distribution
        true_demand_mean = np.random.normal(demand_mean_prior_mean, demand_mean_prior_std)

        # construct the true distribution
        true_distribution = discretize_gaussian(min_demand, max_demand+1, true_demand_mean, true_demand_std)

        # get samples from multinomial distribution, 3 next states with uniform transition kernel
        mult = np.random.multinomial(num_samples, true_distribution)

        # sample from the posterior over samples
        # *** this is the main point of difference ***
        # !!! assume that the state index is the demand !!!
        # this is the distribution over the mean of the demand! not the distribution of the demand
        estmean_demand_mean, estmean_demand_std = normal_aposteriori(demand_values, mult, true_demand_std, demand_mean_prior_mean, demand_mean_prior_std)

        dir_points = np.array([discretize_gaussian(min_demand, max_demand+1, 
                        np.random.normal(estmean_demand_mean, estmean_demand_std), true_demand_std) for k in range(bayes_samples)])
        #print(i,min_demand,max_demand,len(dir_points[0]))
        for vi, value_function in enumerate(value_functions):
            knownV = construct_uset_known_value_function(dir_points, value_function,\
            confidence_level)
            
            knownV_th[vi,i] = knownV[1]
            
            #knownV_ret[vi,i] = knownV[0]
            
            KnownV_nomianl_point[vi].append(knownV[2])
            
    return (np.mean(knownV_th,axis=1), np.std(knownV_th,axis=1), np.mean(KnownV_nomianl_point,axis=1))

def evaluate_gaussian_knownV(dir_points, confidence_level, value_functions):
    """
    Runs the evaluation assuming that the next state represents a demand level
    and that it is distributed according to a normal distribution, & the value function for the next states is known. The prior on
    the mean of this distribution is also Gaussian, and the standard deviation is known.
    @param confidence_level required confidence level

    @return KnownV paramters
    """
    knownV_th = []#np.zeros((num_v, num_simulation))
    KnownV_nomianl_point = []#[ [] for _ in range(num_v)]
    
    for vi, value_function in enumerate(value_functions):
        knownV = construct_uset_known_value_function(dir_points, value_function,\
        confidence_level)
        
        knownV_th.append(knownV[1])
        
        KnownV_nomianl_point.append(knownV[2])
            
    return knownV_th, KnownV_nomianl_point
###
x = np.zeros((2,3))
print(np.mean(x, axis=1))

l = []
l.append([1,2,3])
l.append([0,1,5])
print(np.mean(l,axis=0))


