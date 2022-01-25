'''
Implementation of Robust Median algorithms from D. Feldman and L. J. Schulman, 
Data reduction for weighted and outlier-resistant clustering. 

The algorithm is implemented in a general way to allow finding 
medians of various objects, in spaces with various distance functions.
'''

import numpy as np
from joblib import Parallel, delayed
from utils import unpack_colored_points, unpack_lines

''' Distance functions '''

def dist_points(p, q):
    return np.linalg.norm(p - q)
def dist_colored_points(p, q):
    p, pcolor = unpack_colored_points(p)
    q, qcolor = unpack_colored_points(q)
    distance = dist_points(p, q)
    same_color = pcolor == qcolor
    return np.where(same_color, distance, np.inf)
def dist_lines(L, q):
    p, d = unpack_lines(L)
    qp = q - p
    return np.linalg.norm(qp - (qp @ d) * d)

''' Wrapper distance functions for sets of objects '''

def dist_min_set_to_point(P, q, dist_f):
    cost = float("inf")
    for p in P:
        d = dist_f(p, q)
        if d < cost:
            cost = d
    return cost
def dist_min_set_to_set(P, Q, dist_f):
    cost = float("inf")
    for q in Q:
        d = dist_min_set_to_point(P, q, dist_f)
        if d < cost:
            cost = d
    return cost

def dist_points_min_set_to_point(P, Q):
    return dist_min_set_to_point(P, Q, dist_points)
def dist_points_min_p_to_set(P, Q):
    return dist_points_min_set_to_point(Q, P)
def dist_points_min_set_to_set(P, Q):
    return dist_min_set_to_set(P, Q, dist_points)

def dist_colored_points_min_set_to_point(P, Q):
    return dist_min_set_to_point(P, Q, dist_colored_points)
def dist_colored_points_min_p_to_set(P, Q):
    return dist_colored_points_min_set_to_point(Q, P)
def dist_colored_points_min_set_to_set(P, Q):
    return dist_min_set_to_set(P, Q, dist_colored_points)

def dist_lines_min_set_to_point(P, Q):
    return dist_min_set_to_point(P, Q, dist_lines)
def dist_lines_min_p_to_set(P, Q):
    return dist_colored_points_min_set_to_point(Q, P)
def dist_lines_min_set_to_set(P, Q):
    return dist_min_set_to_set(P, Q, dist_lines)

def dist_p_to_set(p, S, dist_f=dist_points):
    cost = float("inf")
    for q in S:
        cost_curr = dist_f(p, q)
        if cost_curr < cost:
            cost = cost_curr
    return cost
def sum_dist_p_to_set(p, S, dist_f=dist_points):
    cost = sum(dist_f(p, q) for q in S)
    return cost

def dist_set_to_set(P, Q, dist_f=dist_points):
    cost = sum(dist_p_to_set(p, Q, dist_f) for p in P)
    return cost

''' Enumeration functions to allow iteration over special objects in sets '''

def enumerate_set_of_sets(P):
    idx = 0
    for subset in P:
        for element in subset:
            yield idx, element
            idx += 1
          
''' Geometrical functions '''

def closest_point_to_two_lines(l1, l2):
    ''' Triangulation: finding closest points to two lines. '''
    p1, d1 = unpack_lines(l1)
    p2, d2 = unpack_lines(l2)
    if np.isclose(np.abs(d1 @ d2), 1):
        # parallel lines, returning any middle point inbetween
        return (p1 + p2) / 2
    else:
        # intersecting or skew lines, returning the point with minimal distance
        d = p1.shape[-1]
        I = np.eye(d)
        diff1 = I - np.outer(d1, d1)
        diff2 = I - np.outer(d2, d2)
        result = np.linalg.inv(diff1 + diff2) @ (diff1 @ p1 + diff2 @ p2)
        return result
    
def enumerate_set_of_sets_centroids(L):
    ''' Implementation of Centroid Set algorithm from Y. Marom and D. Feldman, 
        k-Means Clustering of Lines for Big Data 
        Enumerates all centroids given set of sets of lines. '''
    idx = 0
    L_all = L.reshape((-1, L.shape[-1]))
    for element1 in L_all:
        for element2 in L_all:
            if np.any(element1 != element2):
                p = closest_point_to_two_lines(element1, element2)
                yield idx, p
                idx += 1

    '''
    for subset in L:
        if len(subset) == 1:
            element = subset[0]
            p, d = unpack_lines(element)
            yield idx, p
            idx += 1
        else:
            for element1 in subset:
                for element2 in subset:
                    if np.any(element1 != element2):
                        p = closest_point_to_two_lines(element1, element2)
                        yield idx, p
                        idx += 1
    '''
    
''' Implementation of the "closest" operation '''

def closest(P, B, gamma=0, dist_f=dist_points, n_closest=None, 
            return_remaining=False):
    ''' Finds max(1, N) objects closest to the set B.
        where N = n_closest if n_closest is not None,
        otherwise N = gamma * |P|'''
    pairs = [(dist_p_to_set(p, B, dist_f), p) for p in P]
    pairs = sorted(pairs, key=lambda x:x[0])
    N = n_closest if n_closest is not None else int(np.ceil(gamma * len(P)))
    pairs_top = pairs[:max(1, N)]
    distances, points = zip(*pairs_top)
    if not return_remaining:
        return np.asarray(points), np.sum(distances)
    else:
        pairs_remaining = pairs[max(1, N):]
        if len(pairs_remaining) > 0:
            distances_remaining, points_remaining = zip(*pairs_remaining)
        else:
            distances_remaining, points_remaining = [], []
        return (np.asarray(points), np.sum(distances),
                np.asarray(points_remaining), np.sum(distances_remaining))

''' Wrappers of the "closest" operation for different objects and distance 
    functions '''

def closest_points(points, point, gamma=0, n_closest=None,
                   return_remaining=False):
    return closest(points, point, gamma=gamma, dist_f=dist_points,
                   n_closest=n_closest, return_remaining=return_remaining)
def closest_colored_points(points, point, gamma=0, n_closest=None,
                           return_remaining=False):
    return closest(points, point, gamma=gamma, dist_f=dist_colored_points,
                   n_closest=n_closest, return_remaining=return_remaining)
def closest_point_sets(points, point, gamma=0, n_closest=None,
                       return_remaining=False):
    return closest(points, point, gamma=gamma, 
                   dist_f=dist_points_min_set_to_set,
                   n_closest=n_closest, return_remaining=return_remaining)
def closest_colored_point_sets(points, point, gamma=0, n_closest=None,
                               return_remaining=False):
    return closest(points, point, gamma=gamma, 
                   dist_f=dist_colored_points_min_set_to_set,
                   n_closest=n_closest, return_remaining=return_remaining)
def closest_colored_point_sets_to_points(
        points, point, gamma=0, n_closest=None, return_remaining=False):
    return closest(points, point, gamma=gamma, 
                   dist_f=dist_colored_points_min_set_to_point,
                   n_closest=n_closest, return_remaining=return_remaining)
def closest_lines(lines, point, gamma=0, n_closest=None, 
                  return_remaining=False):
    return closest(lines, point, gamma=gamma, dist_f=dist_lines,
                   n_closest=n_closest, return_remaining=return_remaining)
def closest_line_sets_to_points(
        lines, point, gamma=0, n_closest=None, return_remaining=False):
    return closest(lines, point, gamma=gamma, 
                   dist_f=dist_lines_min_set_to_point,
                   n_closest=n_closest, return_remaining=return_remaining)

''' Median computation algorithms '''

def median_exhaustive(P, gamma, 
                      dist_f=dist_points, enumerate_f=enumerate,
                      dist_to_set_f=None):
    ''' Finds an approximation to 1-median using exhaustive search.
        dist_f - basic distance function of the space
        dist_to_set_f - wraper of the distance function for sets
        enumerate_f - enumeration function for possible solutions, either
                      returning sub-elements from P or an approximation
                      such as the "centroid set" for lines '''
    if dist_to_set_f is None:
        dist_to_set_f = dist_f
    cost = float("inf")
    center = None
    
    def get_cost_from_center(c):
        closest_c, _ = closest(P, [c], gamma, dist_f)
        cost_curr = sum_dist_p_to_set(c, closest_c, dist_to_set_f)
        return cost_curr, c
     
    result = Parallel()([
        delayed(get_cost_from_center)(c) for _, c in enumerate_f(P)])
    cost, center = sorted(result, key=lambda x: x[0])[0]
    '''
    for idx, c in enumerate_f(P):
        closest_c, _ = closest(P, [c], gamma, dist_f)
        cost_curr = sum_dist_p_to_set(c, closest_c, dist_to_set_f)
        if center is None or cost_curr < cost:
            cost = cost_curr
            center = c
    '''
    
    if cost < np.inf:
        pass
    
    return center, cost

def robust_median(P, k, delta=1/20.0, 
                  dist_f=dist_points, enumerate_f=enumerate,
                  dist_to_set_f=None):
    ''' Implementation of a robust k-median algorithm. 
        Parameters meaning is same as in median_exhaustive '''
    b = 4 / (1-delta) # constant that can be found from proofs of the lemmas
    n = len(P)
    size = max(1, int(np.ceil(b * k**2 * np.log(1/delta)))) # 120
    if size < len(P):
        idx = np.random.choice(n, size=min(size, n), replace=True) 
        S = P[idx]
    else:
        S = P
    q , cost = median_exhaustive(S, 15 / (16*k), 
                                 dist_f=dist_f, enumerate_f=enumerate_f,
                                 dist_to_set_f=dist_to_set_f)
    return q, cost

def recursive_robust_median(P, k, tau=1/10.0, delta=1/20.0, 
                            dist_f=dist_points, enumerate_f=enumerate,
                            dist_to_set_f=None):
    ''' Implementation of the recursive robust k-median algorithm. 
        Parameters meaning is same as in median_exhaustive '''
    Q = P
    #print("len(P):", len(P))
    for i in range(k):
        #print("i = {}".format(i))
        q, _ = robust_median(Q, k, delta=delta, 
                             dist_f=dist_f, enumerate_f=enumerate_f,
                             dist_to_set_f=dist_to_set_f) 
        Q, cost = closest(Q, [q], (1 - tau) / (2 * k), 
                          dist_f=dist_f)
        #print("len(Q): {}; cost: {}".format(len(Q), cost))
        if len(Q) == 1:
            break
    return Q, q, cost

''' Wrappers for median computation algorithms for different objects and
    distance functions '''

def median_points(P, k=1, delta=1/20.0):
    return robust_median(P, k, delta, dist_points)
def median_point_sets(P, k=1, delta=1/20.0):
    return robust_median(P, k, delta, dist_points_min_set_to_set)
def median_colored_point_sets(P, k=1, delta=1/20.0):
    return robust_median(P, k, delta, dist_colored_points_min_set_to_set)
def median_colored_point_sets_to_points(P, k=1, delta=1/20.0):
    return robust_median(P, k, delta, 
                         dist_f=dist_colored_points_min_set_to_point,
                         enumerate_f=enumerate_set_of_sets,
                         dist_to_set_f=dist_colored_points_min_p_to_set)
def median_lines(L, k=1, delta=1/20.0):
    return robust_median(L, k, delta, dist_lines)    
def median_line_sets_to_points(L, k=1, delta=1/20.0):    
    return robust_median(L, k, delta, 
                         dist_f=dist_lines_min_set_to_point,
                         enumerate_f=enumerate_set_of_sets_centroids,
                         dist_to_set_f=dist_lines_min_p_to_set)
    