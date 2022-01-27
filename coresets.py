''' Coreset construction algorithms from the paper 
    "Sets of lines clustering" '''

import numpy as np
from itertools import chain
from collections import Counter
from utils import pack_colored_points, pack_lines, unpack_lines
from median import (
    closest_points, closest_lines, closest_colored_points,
    median_colored_point_sets_to_points,
    closest_colored_point_sets_to_points,
    median_line_sets_to_points, closest_line_sets_to_points)


''' Translation and projection functions '''

def translation_point(p1, p2):
    if p1.ndim == p2.ndim == 1:
        return p2
    elif p1.ndim == 2 and p2.ndim == 1:
        return np.repeat(np.expand_dims(p2, axis=0), len(p1), axis=0)

def translation_line(l, p):
    p1, d1 = unpack_lines(l)
    return pack_lines(translation_point(p1, p), d1) # T(l, p) as in paper

    
def project(P, B, closest_f, translation_f): # todo: handle empty sets
    P_remaining = P #np.copy(P)
    P_projected = []
    for b in B:
        p, _, P_remaining, _ = closest_f(P_remaining, [b], return_remaining=True)
        p_proj = translation_f(p[0], b)
        P_projected.append(p_proj)
    return P_projected, P_remaining

''' Wrappers for projection of different objects '''

def project_point_(P, B):
    return project(P, B, closest_points, translation_point)
def project_colored_point_(P, B):
    return project(P, B, closest_colored_points, translation_point)
def project_line_(P, B):
    return project(P, B, closest_lines, translation_line)

def project_point(P, B):
    return project_point_(P, B)[0]
def project_colored_point(P, B):
    return project_colored_point_(P, B)[0]
def project_line(P, B):
    return project_line_(P, B)[0]

''' Counter ("hat") projections ''' 

def proj_hat_point(P, B):
    return project_point_(P, B)[-1]
def proj_hat_colored_point(P, B):
    return project_colored_point_(P, B)[-1]
def proj_hat_line(P, B):
    return project_line_(P, B)[-1]

''' Geometry functions ''' 

def intersect_spheres(L, B):
    ''' Assumes that lines in L are centered around given points in B '''
    def intersect_sphere(l, b, color):
        c, d = unpack_lines(l)
        points = [b + d, b - d]
        return [pack_colored_points(p, color) for p in points]

    result_lists = [intersect_sphere(l_set, b, color) 
                    for l_set, (color, b) in zip(L, enumerate(B))]
    result = np.asarray(list(chain.from_iterable(result_lists)))
    return result

''' Helper functions ''' 

def set_in_setofsets(p, Q):
    for q in Q:
        if np.all(p == q):
            return True
    return False

def to_tuple(P_array):
    return tuple(P_array.flatten()) + (P_array.shape,)
def from_tuple(P_tuple):
    data = P_tuple[:-1]
    shape = P_tuple[-1]
    return np.asarray(data).reshape(shape)
   
def isin_all(A, B):
    PRECISION = 5 # digits
    return np.isin(np.around(A, PRECISION), 
                   np.around(B, PRECISION)).all()

''' 
    Main algorithms from the paper "Sets of lines clustering"
'''

def CS_dense(P, k, m_CS=2, tau=1/20., k_closest=2):
    ''' P is (n, m)-ordered set. computes recursive robust median 
        Reduces data by ((1 - tau) / (4 * k)) ** m_CS '''
    m = len(P[0])
    m_CS = min(m_CS, len(P[0])) # TODO: check m_CS
    k_closest = min(k, 2) #min(k_closest, 4 * k)
    k_iterations = min(k, 2)
    P_prev = P
    delta = tau
    stopCondition = False
    P_prev_size_prev = len(P)
    for r in range(k_iterations):
        if stopCondition:
            break
        B_prev = []
        #print("r=",r)
        for l in range(m_CS):
            if stopCondition:
                break
            #print("l=", l)
            P_prev_hat = np.asarray([proj_hat_colored_point(P, B_prev) 
                                     for P in P_prev])
            b, _ = median_colored_point_sets_to_points(P_prev_hat, k, delta) 
            P_hat_closest, _ = closest_colored_point_sets_to_points(
                P_prev_hat, [b], (1 - tau) / (1.1 * max(2, k_closest))) # TODO: 2 * k
                
            P_prev = np.asarray([
                P for P in P_prev if isin_all(
                    proj_hat_colored_point(P, B_prev), P_hat_closest)])
                                 
            #                     if set_in_setofsets(
            #                             proj_hat_colored_point(P, B_prev), 
            #                             P_hat_closest)])
            B_prev.append(b)       
            stopCondition = ((len(P_prev) <= 1) or 
                             (len(P_prev) == P_prev_size_prev))
            P_prev_size_prev = len(P_prev)
            print("r={}, l={}, P_hat_closest: {}, P_prev: {}".format(
                r, l, len(P_hat_closest), len(P_prev)))
    return P_prev, np.asarray(B_prev)

def Grouped_sensitivity(L, B, k, k_CS=2, tau=1/20., k_closest=2):    
    ''' Reduces data by ((1 - tau) / (4 * k)) ** m_CS '''
    m = len(L[0])
    P_m = P_all = P_L = np.asarray([intersect_spheres(l, B) for l in L])
    s = dict()
    if k_CS is None:
        k_CS = m * k
    b = 1#2 * k_CS  #2 * m * k

    print("Grouped sensitivity: len(P_m) = {}, 2 * k_CS = {}".format(
          len(P_m), b))
    if 1:#while len(P_m) > b: 
        P_m, B_m = CS_dense(P_L, k_CS, tau=tau, k_closest=k_closest)
        for P in P_m:
            s[to_tuple(P)] = b / len(P_m) 
        P_L = [P for P in P_L if not set_in_setofsets(P, P_m)]
        print("Grouped sensitivity: len(P_m) = {}, 2 * k_CS = {}".format(
              len(P_m), b))

    for q in P_L:
        s[to_tuple(q)] = 1.01
    s_lines = [np.sqrt(2) * s[to_tuple(P_all[idx])] 
               for idx, l in enumerate(L)]
    return np.asarray(s_lines), P_all, s


def LS_dense(L, k, k_closest=None):
    m = min(2, len(L[0]))
    delta = tau = 1/20.0
    L_prev = L
    B_prev = []
    if k_closest is None:
        k_closest = 2.2#(1.1 * max(2, k)) # 2 * k
    
    for i in range(m):
        L_prev_hat = np.asarray([proj_hat_line(L, B_prev) for L in L_prev])
        b, _ = median_line_sets_to_points(L_prev_hat, k, delta) # TODO: cache m**2
        L_hat_closest, _ = closest_line_sets_to_points(
            L_prev_hat, [b], (1 - tau) / k_closest)
        L_prev = np.asarray([
            L for L in L_prev if isin_all(
                proj_hat_line(L, B_prev), L_hat_closest)])
        B_prev.append(b)            
        print("i={}, L_hat_closest: {}, L_prev: {}".format(
            i, len(L_hat_closest), len(L_prev)))
    
    if len(L_prev) == 1:
        return L_prev, np.asarray(B_prev)
    
    L_new = np.asarray([project_line(L, B_prev) for L in L_prev])
    
    sensitivities, _, _ = Grouped_sensitivity(L_new, B_prev, k)
    #sensitivities = np.asarray([s[L_idx] for L_idx, L in enumerate(L_new)])
    min_sensitivity = np.min(sensitivities)
    min_sensitivity_idx = np.isclose(sensitivities, min_sensitivity)
    L_m_plus_one = L_prev[min_sensitivity_idx]
    return L_m_plus_one, np.asarray(B_prev)
    
def coreset(L, k, f_dense=LS_dense, 
            hash_to_f=to_tuple, hash_from_f=from_tuple):
    ''' Computes sensitivities.
        For using with lines:
            - L is set of sets of lines,
            - f_dense = LS_dense
        For using with colored points:
            - L is set of sets of colored points,
            - f_dense=CS_dense. 
        hash_to_f, hash_from_f 
            - transformation functions
              mapping objects from L to hasheable objects such as tuples '''
    L_m_plus_one = L_0 = L
    b = 1 #2 * k
    s = dict()
    print("Coreset: len(L_m_plus_one) = {}, b = {}".format(
          len(L_m_plus_one), b))
        
    stopCondition = False
    #prevSize = float("inf")
    while not stopCondition: #len(L_0) > b:
        L_m_plus_one, B_m_plus_one = f_dense(L_0, k)
        currSize = len(L_m_plus_one)
        stopCondition = currSize <= b # or  currSize > prevSize * 16 
        if not stopCondition:
            for L_set in L_m_plus_one:
                s[hash_to_f(L_set)] = b / currSize
            L_0 = [L_set for L_set in L_0 
                   if not isin_all(L_set, L_m_plus_one)] # subtract sets
            print("Coreset: len(L_m_plus_one) = {}, len(L_0) = {}; b = {}, s = {}, s+l = {}".format(
                  currSize, len(L_0), b, len(s), len(s) + len(L_0)))
            stopCondition = (len(L_0) <= b)
            #prevSize = currSize
    for L_set in L:
        L_hash = hash_to_f(L_set)
        if not L_hash in s:
            s[L_hash] = 1.0
    lines, sensitivities = zip(*s.items())
    lines = np.asarray([hash_from_f(line) for line in lines])
    
    print("s:", len(L), len(s))    
    sensitivities = np.asarray([s[hash_to_f(L_set)] for L_set in L])
    print("Coreset sensitivities counts", Counter(sensitivities))
    return sensitivities

def coreset_sample_biased(L, sensitivities, size):
    ''' Samples points or lines according to given sensitivities '''
    sensitivities = sensitivities + 1 / len(L)
    t = sensitivities.sum()
    sensitivities_normalized = sensitivities / t
    M_idx_big = np.arange(len(L))[sensitivities_normalized >= 1 / size]
    M_idx_normal = np.arange(len(L))[sensitivities_normalized < 1 / size]  
    
    coreset = L[M_idx_big]
    weights = np.ones(len(coreset))
        
    sensitivities_filtered = sensitivities[M_idx_normal]
    t_filtered = sensitivities_filtered.sum()
    p_sampling = sensitivities_filtered / t_filtered

    M_idx_in_idx = np.random.choice(
        np.arange(len(M_idx_normal)), size=size - len(M_idx_big), 
        p=p_sampling, replace=True) 
    M_idx_filtered = M_idx_normal[M_idx_in_idx]
    
    p_additional = p_sampling[M_idx_in_idx]
    coreset_additional = L[M_idx_filtered]
    weights_additional = 1 / (size * p_additional)
    
    coreset = np.concatenate((coreset, coreset_additional))
    weights = np.concatenate((weights, weights_additional))
    return coreset, weights #* len(L) / weights.sum()
    

def coreset_sample(L, sensitivities, size):
    ''' Samples points or lines according to given sensitivities '''
    sensitivities = sensitivities + 1 / len(L)
    t = sensitivities.sum()
    p_sampling = sensitivities / t

    M_idx = np.random.choice(
        np.arange(len(L)), size=size, 
        p=p_sampling, replace=True)
    
    coreset = L[M_idx]
    weights = 1 / (size * p_sampling[M_idx])

    return coreset, weights
    