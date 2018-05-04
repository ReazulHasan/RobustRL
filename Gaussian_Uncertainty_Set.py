import random
import math
import numpy as np
from scipy.stats import norm
from craam import crobust
from scipy import stats

### construct & evaluate uncertainty with Gaussian distributed data points. Calculate L1 worstcase return
def evaluate_gaussian_uncertainty(num_samples, confidence_level, num_simulation, value_function, min_demand = 0,\
                        max_demand = 100, demand_mean_prior_mean = 50, demand_mean_prior_std = 15, true_demand_std = 25):
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
    
    demand_values = np.arange(min_demand, max_demand + 1, dtype=np.double)
    # number of next steps depends on the demands
    num_next_states = max_demand - min_demand + 1
    # rewards - an increasing sequence
    reward = np.arange(min_demand, max_demand + 1, dtype=np.double)
    
    bayes_th = np.zeros(num_simulation)
    bayes_ret = np.zeros(num_simulation)
    bayes_ret_err = np.zeros(num_simulation)
    
    mean_th = np.zeros(num_simulation)
    mean_ret = np.zeros(num_simulation)
    mean_ret_err = np.zeros(num_simulation)
    
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
    
    # number of samples of the true distribution to take when estimatng the Bayes samples
    bayes_samples = 25

    for i in range(num_simulation):
        # construct the true distribution
        true_demand_mean = np.random.normal(demand_mean_prior_mean, demand_mean_prior_std)

        # construct the true distribution
        true_distribution = discretize_gaussian(min_demand, max_demand, true_demand_mean, true_demand_std)

        # get samples from multinomial distribution, 3 next states with uniform transition kernel
        mult = np.random.multinomial(num_samples, true_distribution)
        
        mean_transition_prob = mult / np.sum(mult)
        
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
        
        #print("Gaussian: ",min_demand, max_demand, dir_points[0])
        
        # calc mean probability p_hat 
        # TODO: marek changed from: nominal_prob = np.mean(dir_points, axis=0)
        # TODO: that may not result in a valid probability distribution, take the mean of samples instead
        nominal_prob_bayes = np.mean(dir_points, axis=0)
        nominal_prob_bayes /= np.sum(nominal_prob_bayes)
        # TODO: marek: also tried but does not seem to work
        nominal_prob_freq = mult / np.sum(mult)
        
        #get uncertainty set & threshold
        bayes_th[i] = compute_bayesian_threshold(dir_points,nominal_prob_bayes, confidence_level)        
        
        #calc threshold from hoeffding bound equation
        hoeff_th[i] = np.sqrt((2 / num_samples )*np.log((2**num_next_states-2) / (1-confidence_level) ))   
        
        # ** calculate the tight hoeffding bound
        tight_hoeff_th[i]= np.sqrt((2 / num_samples )*np.log((num_next_states-1) / (1 - confidence_level) ))   # TODO: should be -1 or -2?

        em_nominal, emthreshold = calc_EM_rand(dir_points, confidence_level, nominal_prob_bayes)
        em_nominal /= np.sum(em_nominal)
        em_th[i] = emthreshold
        
        knownV = construct_uset_known_value_function(dir_points, value_function, confidence_level)
        knownV_th[i] = knownV[1]
        
        true_ret = true_distribution @ reward
        bayes_ret[i] = crobust.worstcase_l1(reward, nominal_prob_bayes, bayes_th[i])
        mean_ret[i] = crobust.worstcase_l1(reward, mean_transition_prob, 0)
        hoeff_ret[i] = crobust.worstcase_l1(reward, nominal_prob_freq, hoeff_th[i])
        tight_hoeff_ret[i] = crobust.worstcase_l1(reward, nominal_prob_freq, tight_hoeff_th[i])
        em_ret[i] = crobust.worstcase_l1(reward, em_nominal, em_th[i])
        knownV_ret[i] = knownV[0]
        
        bayes_ret_err[i] = (true_ret - bayes_ret[i]) / true_ret
        mean_ret_err[i] = (true_ret - mean_ret[i]) / true_ret
        hoeff_ret_err[i] = (true_ret - hoeff_ret[i]) /true_ret
        tight_hoeff_ret_err[i] = (true_ret - tight_hoeff_ret[i]) /true_ret
        em_ret_err[i] = (true_ret - em_ret[i]) /true_ret
        knownV_ret_err[i] = (true_ret - knownV_ret[i])/true_ret
        
    # make sure to not count negative return errors to improve the mena
    return [(Methods.BAYES, np.mean(np.maximum(0,bayes_ret_err)), np.mean(bayes_th), np.mean(bayes_ret_err < 0), np.mean(bayes_ret), np.std(np.maximum(0,bayes_ret_err)), np.std(bayes_th) ),\
            (Methods.CENTROID, np.mean(np.maximum(0,mean_ret_err)), 0, np.mean(mean_ret_err < 0), np.mean(mean_ret), np.std(np.maximum(0,mean_ret_err)), 0 ),\
            (Methods.HOEFF, np.mean(np.maximum(0,hoeff_ret_err)), np.mean(hoeff_th), np.mean(hoeff_ret_err < 0), np.mean(hoeff_ret), np.std(np.maximum(0,hoeff_ret_err)), np.std(hoeff_th) ),\
            (Methods.HOEFFTIGHT, np.mean(np.maximum(0,tight_hoeff_ret_err)), np.mean(tight_hoeff_th), np.mean(tight_hoeff_ret_err < 0), np.mean(tight_hoeff_ret), np.std(np.maximum(0,tight_hoeff_ret_err)), np.std(tight_hoeff_th)),\
            (Methods.EM, np.mean(np.maximum(0,em_ret_err)), np.mean(em_th), np.mean(em_ret_err < 0), np.mean(em_ret), np.std(np.maximum(0,em_ret_err)), np.std(em_th) ),\
            (Methods.INCR_REPLACE_V, np.mean(np.maximum(0,knownV_ret_err)), np.mean(knownV_th), np.mean(knownV_ret_err<0), np.mean(knownV_ret), np.std(np.maximum(0,knownV_ret_err)), np.std(knownV_th)),
            (Methods.INCR_ADD_V, np.mean(np.maximum(0,knownV_ret_err)), np.mean(knownV_th), np.mean(knownV_ret_err<0), np.mean(knownV_ret), np.std(np.maximum(0,knownV_ret_err)), np.std(knownV_th))]
            
            

### construct & evaluate uncertainty with Gaussian Distribution & Known Value Function
def evaluate_gaussian_knownV(num_samples, confidence_level, num_simulation, value_function, min_demand = 0, max_demand = 100, demand_mean_prior_mean = 50, demand_mean_prior_std = 15, true_demand_std = 25):
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
    # number of next steps depends on the demands
    num_next_states = max_demand - min_demand + 2
    # rewards - an increasing sequence
    reward = np.arange(min_demand, max_demand + 2, dtype=np.double)
    
    # number of samples of the true distribution to take when estimatng the Bayes samples
    bayes_samples = 25
    
    knownV_th = np.zeros(num_simulation)
    knownV_ret = np.zeros(num_simulation)
    knownV_ret_err = np.zeros(num_simulation)
    KnownV_nomianl_point = []#np.zeros(num_simulation)
    
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
        knownV = construct_uset_known_value_function(dir_points, value_function, confidence_level)
            
        knownV_th[i] = knownV[1]
    
        knownV_ret[i] = knownV[0]
            
        KnownV_nomianl_point.append(knownV[2])
        
        true_ret = true_distribution @ reward
        knownV_ret_err[i] = (true_ret - knownV_ret[i])/true_ret
        
    # make sure to not count negative return errors to improve the mean
    return [(Methods.KNOWNV, np.mean(np.maximum(0,knownV_ret_err)), np.mean(knownV_th), np.mean(knownV_ret_err<0), np.mean(knownV_ret), np.std(np.maximum(0,knownV_ret_err)), np.std(knownV_th), np.mean(KnownV_nomianl_point,axis=0))]




