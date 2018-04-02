import random
import math
import numpy as np
from gurobipy import *
import sys
from enum import Enum

### Enumerate experimenting methods
if __name__ == "__main__":
    class Methods(Enum):
        BAYES = "Bayes Simple"
        HOEFF = "Hoeffding"
        HOEFFTIGHT = "Hoeffding Tight"
        EM = "Expectation Maximization"
        KNOWNV = "Known Value Function"
        IMPROVEV = "Iteratively Improve Value Function"
        ADDRANDOMV = "Iteratively Add Random Value Function"
        
        NUM_METHODS = 7 #Number of methods
    
    LI_METHODS = [Methods.BAYES, Methods.HOEFF, Methods.HOEFFTIGHT, Methods.EM,\
                Methods.KNOWNV, Methods.IMPROVEV, Methods.ADDRANDOMV]

### Apply Linear Program to compute the nominal point & apply quick select to find the uncertainty set with the required confidence
def get_uset(points, nominal_point, conf_rank):
    """
    Sort the points based on the distance from nominal point in ascending order. 
    Taking N*confidence will  give the uncertainty set with nearest points 

    @param conf_rank: how many points to take 
    """
    distances = [( p, np.linalg.norm(p - nominal_point, ord = 1)) for p in points]
    distances.sort(key = lambda x : x[1])
    res_list = np.squeeze([distances[i][0] for i in range(conf_rank)])
    psi = distances[conf_rank-1][1]
    return res_list, psi

def find_nominal_point(p):
    """
    Find nominal point for the uncertainty set using LP
    """
    num_p = p.shape[0]
    num_d = p.shape[1]
    m = Model("nominal")
    u = m.addVar(name="u", lb=0)
    
    y = m.addVars(range(num_p*num_d), vtype=GRB.CONTINUOUS, obj=0.0, name="y")
    # new nominal point
    beta = m.addVars(range(num_d), vtype=GRB.CONTINUOUS, obj=0.0, name="beta", lb=0)
    
    m.setObjective(u, GRB.MINIMIZE)
    
    for i in range(num_p):
        m.addConstr(u, GRB.GREATER_EQUAL, quicksum(y[i*num_d+j] for j in range(num_d)), "u_"+str(i))

    for i in range(num_p):
        for j in range(num_d):
            m.addConstr(y[i*num_d+j], GRB.GREATER_EQUAL, p[i,j]-beta[j], "c1"+str(i))
            m.addConstr(y[i*num_d+j], GRB.GREATER_EQUAL, beta[j]-p[i,j], "c2"+str(i))

    m.setParam( 'OutputFlag', False )
    m.optimize()
    
    #print('Obj: %g' % m.objVal) 
    
    #for v in m.getVars():
    #    print('%s %g' % (v.varName, v.x))
    
    nominal_params = m.getAttr('x', beta)

    nominal_p = []
    for i in range(num_d):
        nominal_p.append(nominal_params[i])
    
    return nominal_p#tuple(nominal_p)

### Test EM
if __name__ == "__main__":
    num_points = 100
    confidence = 0.8
    confidence_rank = math.ceil(num_points * confidence)
    x = np.random.rand(num_points,1)*100
    y = np.random.rand(num_points,1)*100
    points = np.squeeze(np.stack((x,y),axis=1))
    
    num_iter = 20
    nominal_list = []
    psi_list = []
    
    rand_nominal_pos = np.random.randint(num_points)
    
    nominal_p = points[rand_nominal_pos]
    
    prev_psi = sys.float_info.max
    for i in range(num_iter):
        nominal_list.append(nominal_p)
        u_set, psi = get_uset(points, nominal_p, confidence_rank)
        
        assert(psi <= prev_psi + 1e-5)
        
        prev_psi = psi
        psi_list.append(psi)
        nominal_p = find_nominal_point(u_set)
    
    nominal_list = np.squeeze(nominal_list)
    
    print("nominal_list",nominal_list)
    print("psi_list",psi_list)

### calculate EM

def calc_EM(points, confidence_level, nominal_p):
    num_iter = 100
    num_count = 0
    nominal_list = []
    psi_list = []
    confidence_rank = math.ceil(len(points) * confidence_level)
    
    prev_psi = np.inf
    for i in range(num_iter):        
        u_set, psi = get_uset(points, nominal_p, confidence_rank)
        if psi >= prev_psi or num_count>=100:
            break
            
        assert(psi <= prev_psi + 1e-5)
        
        #print("i",i,"psi",psi)
        num_count += 1
        prev_psi = psi
        psi_list.append(psi)
        nominal_list.append(nominal_p)
        nominal_p = find_nominal_point(u_set)

    if len(nominal_list) > 1:
        nominal_list = np.squeeze(nominal_list)
    #print(nominal_list)
    return nominal_list[-1], psi_list[-1]

def calc_EM_rand(points, confidence_level, nominal_p):
    """
    A randomized version
    """
    
    new_nominal, new_psi = calc_EM(points, confidence_level, nominal_p)
    
    for i in range(10):
        next_nominal_p = points[np.random.randint(points.shape[0])]
        next_nominal, next_psi = calc_EM(points, confidence_level, next_nominal_p)
        if(next_psi < new_psi):
            new_psi = next_psi
            new_nominal = next_nominal
            
    return new_nominal, new_psi

### Construct Uset for known value function
import operator

def construct_uset_known_value_function(transition_points, value_function, confidence):
    """
    Computes the robust return and a threshold that achieves the desired confidence level
    for a single state and action.

    @param transition_points Samples from the posterior distribution of the transition
                             probabilities
    @param value_function Assumed optimal value function
    @param confidence Desired confidence level, such as 0.99
    """
    points = []
    #print("len(transition_points)",len(transition_points[0]), "len(value_function)",len(value_function))
    #print("transition_points",transition_points[0], "value_function",value_function)
    #print("transition points",transition_points, "value_function", value_function)
    for p in transition_points:
        points.append( (p,p@value_function) )
    points.sort(key=lambda x: x[1])
    #print("points with value_function",points)
    conf_rank = math.ceil(len(transition_points)*confidence)
    robust_ret = points[-conf_rank][1]
    robust_th = 0 #np.linalg.norm(points[-int(conf_rank)][0]-points[-int(conf_rank/2)][0], ord=1)
    nominal_point = points[-conf_rank][0]
    
    #print("centroid",points[-int(conf_rank/2)], "ret", robust_ret, "th", robust_th)
    return (robust_ret, robust_th, nominal_point)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
