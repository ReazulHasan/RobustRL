import random
import math
import numpy as np
from craam import crobust
import Utils

### Construct uncertainty set

def compute_bayesian_threshold(points, nominal_point, confidence_level):
    """
    Computes an empirical thresholds from samples from a posterior distriL1 threshold'bution
    Must adjust the confidence level when the MDP has more states 
    and actions using the union bound
    """
    assert(abs(np.sum(nominal_point) - 1) < 0.001)
    for p in points:
        assert(abs(np.sum(p) - 1) < 0.001)
    
    distances = [np.linalg.norm(p - nominal_point, ord = 1) for p in points]
    confidence_rank = math.floor(len(points) * confidence_level)
    dist = np.partition(distances, confidence_rank)[confidence_rank]
    
    return dist
    
def compute_hoeffding_threshold(num_next_states, num_samples, confidence_level):
    """
    Hoeffding threshold for a single state and action. Must adjust the confidence level
    when the MDP has more states and actions using the union bound
    """
    return np.sqrt( (2 / num_samples ) * np.log( (2**num_next_states-2) / confidence_level) )

### construct & evaluate uncertainty with Multinomial + dirichlet distributed data points. Calculate L1 worstcase return

def evaluate_bayesian_uncertainty(num_points, num_next_states, reward, confidence_level, num_simulation, value_function):
    """
    Compares Hoeffding and Bayesian methods for constructing uncertainty sets
    
    @returns list of results with tuples 
            (method_name, 
                error_fractions: error as a fraction of the true return, 
                L1 thresholds, 
                violations: fraction of instances in which the value is not a lower bound)
    """
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
    
    # number of samples of the true distribution to take when estimating the Bayes samples
    bayes_samples = 200
    prior = np.ones(num_next_states)
    
    #value_function = np.random.randint(10, size=num_next_states)
    #print(value_function)

    for i in range(num_simulation):
        # construct the true distribution
        true_distribution = np.random.dirichlet(prior, 1)[0]
        
        #print(i,"true_distribution",true_distribution)
        # get samples from multinomial distribution
        mult = np.random.multinomial(num_points, true_distribution)
        
        mean_transition_prob = mult / np.sum(mult)
        #print("mult",mult,"mean_transition_prob",mean_transition_prob)
        
        # ** calculate simple bayesian threshold
        # sample transition points from the posterior Dirichlet distribution        
        dir_points = np.random.dirichlet(mult + prior, bayes_samples) 
        
        #print("true_distribution",true_distribution,"mult",mult)
        #print("dir_points",dir_points)

        # calc mean probability p_hat 
        # TODO: marek changed from: nominal_prob = np.mean(dir_points, axis=0)
        # TODO: that may not result in a valid probability distribution, take the mean of samples instead
        nominal_prob_bayes = np.mean(dir_points, axis=0)
        nominal_prob_bayes /= np.sum(nominal_prob_bayes)
        # TODO: marek: also tried but does not seem to work
        nominal_prob_freq = mult / np.sum(mult)
        
        # TODO: marek: delta is 1 - confidence
        # get uncertainty set & threshold
        bayes_th[i] = compute_bayesian_threshold(dir_points, nominal_prob_bayes, confidence_level)        
        
        # TODO: marek: delta is 1 - confidence
        # ** calculate threshold from hoeffding bound equation
        hoeff_th[i] = np.sqrt((2 / num_points )*np.log((2**num_next_states-2)/ (1-confidence_level) ))   
        
        # ** calculate the tight hoeffding bound
        tight_hoeff_th[i]= np.sqrt((2 / num_points )*np.log((num_next_states-1)/ (1-confidence_level) ))   # TODO: marek needs to fige out whether this should be -1 or -2?
        
        em_nominal, emthreshold = calc_EM_rand(dir_points, confidence_level, nominal_prob_bayes)
        em_nominal /= np.sum(em_nominal)
        
        #print("nominal_prob_bayes",np.sum(nominal_prob_bayes), "em_nominal", np.sum(em_nominal))
        em_th[i] = emthreshold
        
        knownV = construct_uset_known_value_function(dir_points, value_function, confidence_level)
        knownV_th[i] = knownV[1]
        true_ret = true_distribution @ reward
        true_ret_knownV = true_distribution @ value_function
        
        bayes_ret[i] = crobust.worstcase_l1(reward, nominal_prob_bayes, bayes_th[i])
        mean_ret[i] = crobust.worstcase_l1(reward, mean_transition_prob, 0)
        hoeff_ret[i] = crobust.worstcase_l1(reward, nominal_prob_freq, hoeff_th[i])
        tight_hoeff_ret[i] = crobust.worstcase_l1(reward, nominal_prob_freq, tight_hoeff_th[i])
        em_ret[i] = crobust.worstcase_l1(reward, em_nominal, em_th[i])
        knownV_ret[i] = knownV[0]
        
        bayes_ret_err[i] = (true_ret - bayes_ret[i]) #/ true_ret
        mean_ret_err[i] = (true_ret - mean_ret[i]) #/ true_ret
        hoeff_ret_err[i] = (true_ret - hoeff_ret[i]) #/true_ret
        tight_hoeff_ret_err[i] = (true_ret - tight_hoeff_ret[i]) #/true_ret
        em_ret_err[i] = (true_ret - em_ret[i]) #/true_ret
        knownV_ret_err[i] = (true_ret_knownV - knownV_ret[i]) #/true_ret_knownV
        
        #print("true_ret_knownV", true_ret_knownV, "knownV_ret[i]", knownV_ret[i], "knownV_ret_err[i]", knownV_ret_err[i])
    #print("knownV_ret_err<0", knownV_ret_err<0, "np.mean(knownV_ret_err<0)", np.mean(knownV_ret_err<0))

    # make sure to not count negative return errors to improve the mena
    return [(Methods.BAYES, np.mean(abs(bayes_ret_err)), np.mean(bayes_th), np.mean(bayes_ret_err < 0), np.mean(bayes_ret), np.std(abs(bayes_ret_err)), np.std(bayes_th) ),\
            (Methods.CENTROID, np.mean(abs(mean_ret_err)), 0, np.mean(mean_ret_err < 0), np.mean(mean_ret), np.std(abs(mean_ret_err)), 0 ),\
            (Methods.HOEFF, np.mean(abs(hoeff_ret_err)), np.mean(hoeff_th), np.mean(hoeff_ret_err < 0), np.mean(hoeff_ret), np.std(abs(hoeff_ret_err)), np.std(hoeff_th) ),\
            (Methods.HOEFFTIGHT, np.mean(abs(tight_hoeff_ret_err)), np.mean(tight_hoeff_th), np.mean(tight_hoeff_ret_err < 0), np.mean(tight_hoeff_ret), np.std(abs(tight_hoeff_ret_err)), np.std(tight_hoeff_th)),\
            (Methods.EM, np.mean(abs(em_ret_err)), np.mean(em_th), np.mean(em_ret_err < 0), np.mean(em_ret), np.std(abs(em_ret_err)), np.std(em_th) ),\
            (Methods.INCR_REPLACE_V, np.mean(abs(knownV_ret_err)), np.mean(knownV_th), np.mean(knownV_ret_err<0), np.mean(knownV_ret), np.std(abs(knownV_ret_err)), np.std(knownV_th)),\
            (Methods.INCR_ADD_V, np.mean(abs(knownV_ret_err)), np.mean(knownV_th), np.mean(knownV_ret_err<0), np.mean(knownV_ret), np.std(abs(knownV_ret_err)), np.std(knownV_th))]
            
            


