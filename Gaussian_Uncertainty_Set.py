import random
import math
import numpy as np
from scipy.stats import norm
from craam import crobust
from scipy import stats
from Utils import *

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
    reward = np.arange(min_demand, max_demand + 1, dtype=np.double) #np.random.randint(10, size=max_demand-min_demand+1)
    reward = np.array([float(i) for i in reward])
    
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
    bayes_samples = 300

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
                    np.random.normal(estmean_demand_mean, estmean_demand_std+true_demand_std), true_demand_std) \
                    for k in range(bayes_samples)])
        
        #print("sum of probabilities",np.sum(dir_points, axis=1))
        #print("Gaussian: ",min_demand, max_demand, dir_points[0])
        
        # calc mean probability p_hat 
        nominal_prob_bayes = np.mean(dir_points, axis=0)
        nominal_prob_bayes /= np.sum(nominal_prob_bayes)
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
        
        knownV = construct_uset_known_value_function(dir_points, reward, confidence_level)
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
        
    # make sure to not count negative return errors to improve the mean
    return [(Methods.BAYES, np.mean(abs(bayes_ret_err)), np.mean(bayes_th), np.mean(bayes_ret_err < 0), np.mean(bayes_ret), np.std(abs(bayes_ret_err)), np.std(bayes_th) ),\
            (Methods.CENTROID, np.mean(abs(mean_ret_err)), 0, np.mean(mean_ret_err < 0), np.mean(mean_ret), np.std(abs(mean_ret_err)), 0 ),\
            (Methods.HOEFF, np.mean(abs(hoeff_ret_err)), np.mean(hoeff_th), np.mean(hoeff_ret_err < 0), np.mean(hoeff_ret), np.std(abs(hoeff_ret_err)), np.std(hoeff_th) ),\
            (Methods.HOEFFTIGHT, np.mean(abs(tight_hoeff_ret_err)), np.mean(tight_hoeff_th), np.mean(tight_hoeff_ret_err < 0), np.mean(tight_hoeff_ret), np.std(abs(tight_hoeff_ret_err)), np.std(tight_hoeff_th)),\
            (Methods.EM, np.mean(abs(em_ret_err)), np.mean(em_th), np.mean(em_ret_err < 0), np.mean(em_ret), np.std(abs(em_ret_err)), np.std(em_th) ),\
            (Methods.INCR_REPLACE_V, np.mean(abs(knownV_ret_err)), np.mean(knownV_th), np.mean(knownV_ret_err<0), np.mean(knownV_ret), np.std(abs(knownV_ret_err)), np.std(knownV_th)),
            (Methods.INCR_ADD_V, np.mean(abs(knownV_ret_err)), np.mean(knownV_th), np.mean(knownV_ret_err<0), np.mean(knownV_ret), np.std(abs(knownV_ret_err)), np.std(knownV_th))]
            

### construct & evaluate uncertainty with Gaussian Distribution & Known Value Function

import operator

def construct_uset_known_value_function_ext(transition_points, value_function, confidence):
    """
    Computes the robust return and a threshold that achieves the desired confidence level
    for a single state and action.

    @param transition_points Samples from the posterior distribution of the transition
                             probabilities
    @param value_function Assumed optimal value function
    @param confidence Desired confidence level, such as 0.99
    """
    points = []

    for p in transition_points:
        points.append( (p,p@value_function) )
    points.sort(key=lambda x: x[1])

    conf_rank = math.ceil(len(transition_points)*confidence)
    #print("confidence_rank", conf_rank, "len(trans_points)", len(transition_points), "confidence",confidence,"product",confidence*len(transition_points))
    robust_ret = points[conf_rank][1]
    robust_th = 0 #np.linalg.norm(points[-int(conf_rank)][0]-points[-int(conf_rank/2)][0], ord=1)
    nominal_point = points[conf_rank][0]
    
    return (robust_ret, robust_th, nominal_point, points[:-conf_rank])

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
    
    demand_values = np.arange(min_demand, max_demand + 1, dtype=np.double)
    # number of next steps depends on the demands
    num_next_states = max_demand - min_demand + 1
    # rewards - an increasing sequence
    reward = np.arange(min_demand, max_demand + 1, dtype=np.double)
    
    # number of samples of the true distribution to take when estimatng the Bayes samples
    bayes_samples = 300
    
    knownV_th = np.zeros(num_simulation)
    knownV_ret = np.zeros(num_simulation)
    knownV_ret_err = np.zeros(num_simulation)
    KnownV_nomianl_point = []#np.zeros(num_simulation)
    
    li_true_mean, li_est_mean, li_prior_mean = [], [], []
    
    li_out_of_ambiguity_set = []

    for i in range(num_simulation):
        # construct the true distribution
        true_demand_mean = np.random.normal(demand_mean_prior_mean, demand_mean_prior_std)
        
        li_prior_mean.append(demand_mean_prior_mean)
        li_true_mean.append(true_demand_mean)
        
        # construct the true distribution
        true_distribution = discretize_gaussian(min_demand, max_demand, true_demand_mean, true_demand_std)
        
        # get samples from multinomial distribution, 3 next states with uniform transition kernel
        mult = np.random.multinomial(num_samples, true_distribution)
        
        estmean_demand_mean, estmean_demand_std = normal_aposteriori(demand_values, mult, true_demand_std, demand_mean_prior_mean, demand_mean_prior_std)
        
        li_est_mean.append(estmean_demand_mean)
        
        dir_points = np.array([discretize_gaussian(min_demand, max_demand, 
                        np.random.normal(estmean_demand_mean, estmean_demand_std), true_demand_std) for k in range(bayes_samples)])
        
        print("dir_points", dir_points)
        
        knownV = construct_uset_known_value_function_ext(dir_points, reward, confidence_level)
        
        knownV_th[i] = knownV[1]
    
        knownV_ret[i] = knownV[0]
        
        KnownV_nomianl_point.append(knownV[2])
        
        true_ret = true_distribution @ reward
        knownV_ret_err[i] = (true_ret - knownV_ret[i])/true_ret
        
        points = knownV[3]
        print("len(points)", len(points))
        err = []
        for p in points:
            err.append((true_ret - p[1])/true_ret)
        li_out_of_ambiguity_set.append( err )
    
    print( np.mean( np.array(li_out_of_ambiguity_set)<0 ) )
    
    import matplotlib.pyplot as plt
    plt.plot(li_prior_mean, color='black', label="Prior mean")
    plt.plot(li_true_mean, color='r', label="True mean sampled from prior")
    plt.plot(li_est_mean, color='g', label="Estimation of true mean")
    plt.legend(loc='best', fancybox=True, framealpha=0.3)
    plt.show()
    
    # make sure to not count negative return errors to improve the mean
    return [(Methods.INCR_ADD_V, np.mean(abs(knownV_ret_err)), np.mean(knownV_th), np.mean(knownV_ret_err<0), np.mean(knownV_ret), np.std(abs(knownV_ret_err)), np.std(knownV_th), np.mean(KnownV_nomianl_point,axis=0))]

###
def test_evaluate_gaussian_knownV(num_samples, num_simulation, min_demand = 0, max_demand = 100, demand_mean_prior_mean = 50, demand_mean_prior_std = 15, true_demand_std = 25):
    demand_values = np.arange(min_demand, max_demand + 1, dtype=np.double)
    # number of next steps depends on the demands
    num_next_states = max_demand - min_demand + 1
    # rewards - an increasing sequence
    reward = np.arange(min_demand, max_demand + 1, dtype=np.double)
    
    # number of samples of the true distribution to take when estimatng the Bayes samples
    bayes_samples = 300
    
    knownV_th = np.zeros(num_simulation)
    knownV_ret = np.zeros(num_simulation)
    knownV_ret_err = np.zeros(num_simulation)
    KnownV_nomianl_point = []#np.zeros(num_simulation)
    
    li_true_mean, li_est_mean, li_est_std, li_prior_mean = [], [], [], []
    
    li_out_of_ambiguity_set = []

    for i in range(num_simulation):
        # construct the true distribution
        true_demand_mean = np.random.normal(demand_mean_prior_mean, demand_mean_prior_std)
        
        li_prior_mean.append(demand_mean_prior_mean)
        li_true_mean.append(true_demand_mean)
        
        # construct the true distribution
        true_distribution = discretize_gaussian(min_demand, max_demand, true_demand_mean, true_demand_std)
        
        # get samples from multinomial distribution, 3 next states with uniform transition kernel
        mult = np.random.multinomial(num_samples, true_distribution)
        
        estmean_demand_mean, estmean_demand_std = normal_aposteriori(demand_values, mult, true_demand_std,\
                                                                demand_mean_prior_mean, demand_mean_prior_std)
        li_est_mean.append(estmean_demand_mean)
        li_est_std.append(estmean_demand_std)
            
    return (np.mean(li_est_mean), np.mean(li_est_std), np.mean(li_true_mean), true_demand_std)

###
samples = np.arange(20,300,20)
true_mean, est_mean, true_std, est_std = [], [], [], []

for i in samples:
    res = test_evaluate_gaussian_knownV(i, 200)
    est_mean.append(res[0])
    est_std.append(res[1])
    true_mean.append(res[2])
    true_std.append(res[3])

###
import matplotlib.pyplot as plt
plt.plot(samples, true_mean, color='black', label="true mean")
plt.plot(samples, true_std, color='r', label="true std")
plt.plot(samples, est_mean, color='g', label="estimated mean")
plt.plot(samples, est_std, color='b', label="estimated std")
plt.legend(loc='best', fancybox=True, framealpha=0.3)
plt.xlabel("number of samples")
plt.show()










