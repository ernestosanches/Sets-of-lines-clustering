import numpy as np
from median import (
    robust_median, recursive_robust_median,
    closest_lines, closest_points, dist_points,
    closest_point_sets, pack_lines, unpack_lines,
    pack_colored_points, dist_colored_points, dist_lines_min_set_to_set,
    closest_colored_points, unpack_colored_points,
    closest_colored_point_sets, dist_colored_points_min_set_to_set,
    dist_colored_points_min_set_to_point, dist_colored_points_min_p_to_set,
    enumerate_set_of_sets, closest_colored_point_sets_to_points)



''' Data generation functions '''

def generate_points(n):
    mu = np.random.normal(0, 5, 2)
    P = np.random.multivariate_normal(mu + np.array([100, 0]), 
                                      np.eye(2) * 2, int(n * 1/100))
    Q = np.random.multivariate_normal([0,0], np.array([[5,0],[0,1]]) * 5, 
                                      (n - len(P)) // 2)
    R = np.random.multivariate_normal([0,0], np.array([[1,0],[0,5]]) * 5, 
                                      n - len(P) - len(Q))
    P = np.vstack([P, Q, R])
    #np.random.shuffle(P)
    return P

def color_the_points(P_points, color=None):
    if color is None:
        P_color = np.abs(np.asarray(P_points.sum(
            axis=1, keepdims=True), dtype=int)) % 3
    else:
        P_color = np.full((len(P_points), 1), color)        
    return pack_colored_points(P_points, P_color)
    
def generate_colored_points(n):
    P = generate_points(n)
    return color_the_points(P)

def color_set_of_sets_points(P):
    result = []
    for p_set in P:
        result.append([])
        for color, p in enumerate(p_set):
            result[-1].append(color_the_points(p, color))
    return result

def stack_point_sets(point_sets):
    m = len(point_sets)
    if m == 0:
        return None
    n, d = point_sets[0].shape
    return np.concatenate([np.expand_dims(ps, axis=1) for ps in point_sets],
                          axis=1)

def generate_points_sets(n):
    P = np.random.multivariate_normal([0,0], np.eye(2) * 5, n)
    Q = np.random.multivariate_normal([5,1], np.eye(2) * 2, n)
    n, d = P.shape    
    return stack_point_sets((P, Q)) #list(zip(P, Q))

import pandas as pd
from SetsClustering.Utils import createFlowerDataset, to_array

def generate_colored_points_sets(n, m, r=1, is_colored=True):
    
    set_P = createFlowerDataset(r=r, n = n - 90, m=m)
    #set_P_indiced = [(P, idx) for (idx, P) in enumerate(set_P)] 
    P, w = to_array(set_P)
    n, m, d = P.shape
    colors = np.zeros((n, m, 1))
    for i in range(m):
        colors[:, i, 0] = i if is_colored else 0
    P = np.concatenate((P, colors), axis=-1)
    return np.unique(np.around(P, 5), axis=0)
    
    '''
    LOAD_3D_MODEL = False
    if LOAD_3D_MODEL:
        # Real 3d 
        assert(m == 1)
        fpath = "/home/ernesto/projects/tracxpoint/sfm_postprocessing/"
        fname = fpath + "points3DWithDescriptors_front (4).txt"
        data = pd.read_csv(
            fname, delimiter="|", usecols=range(1,1+3), 
            header=0, names=["x", "y", "z"])#, "r", "g", "b"])  
        P = data.values
        n = len(P)
        P = np.hstack((P, np.zeros((n, 1))))        
        return np.expand_dims(P, axis=1)

    LOAD_COUNTEREXAMPLE = False
    if LOAD_COUNTEREXAMPLE:
        # COUNTEREXAMPLE
        P = np.random.uniform(0, 1, size = (n, m, 3))
        P[:, :, -1] = np.random.randint(0, m, size=(n, 1))
        return P

    MANUALLY_GENERATED = False
    if MANUALLY_GENERATED:
        if 1:
            P1 = np.random.multivariate_normal([0,0], np.eye(2) * 0.2, n)
            P2 = np.random.multivariate_normal([3,2], np.eye(2) * 0.2, n)
            P3 = np.random.multivariate_normal([5,-2], np.eye(2) * 0.2, n)
            P4 = np.random.multivariate_normal([7,0], np.eye(2) * 0.2, n)
        else:
            P1 = np.random.multivariate_normal([0,0], np.array([[0.02,1],[1,0.02]]), n)
            P2 = np.random.multivariate_normal([3,2], np.array([[0.02,-1],[-1,0.02]]), n)
            P3 = np.random.multivariate_normal([5,-2], np.array([[0.02,-1],[-1,0.02]]), n)
            P4 = np.random.multivariate_normal([7,0], np.array([[0.02,1],[1,0.02]]), n)
        colored_points = [color_the_points(P, color) 
                          for color, P in enumerate([P1, P2, P3, P4][:m])]
    else:
        # CURRENT EXPERIMENT
        point_groups = [generate_points(n) + np.random.normal(0, 10, (1,2))
                           for i in range(m)]
        colored_points = [color_the_points(P, color) 
                          for color, P in enumerate(point_groups)]
    '''
    #return stack_point_sets(colored_points)


def generate_set_of_lines_new(n, offset=np.zeros((1, 2))):
    size = n//50
    c = np.repeat(offset, size, axis=0)
    d = np.hstack([3 * np.ones((size, 1)), np.random.normal(0, 1, size=(size, 1))])
    print("c,d:", c.shape, d.shape)
    P = pack_lines(c, d)    

    size = n - n//50
    c = np.repeat(offset, size, axis=0)
    d = np.hstack([np.random.normal(0, 0.5, size=(size, 1)), 3 * np.ones((size, 1))])
    Q = pack_lines(c, d)    

    return np.vstack((P, Q))

def generate_set_of_lines(n, offset=np.zeros((1,2))):
    c = generate_points(n) + offset
    d = generate_points(n)
    print(d.shape)
    #d = np.hstack([np.random.normal(0, 0.03, size=(n, 1)), 0.1 * np.ones((n, 1))])
    #print(d.shape)
    #d = d / np.linalg.norm(d, axis=-1)[:, np.newaxis]
    return pack_lines(c, d)

def generate_set_of_sets_of_lines(n, m):
    L_set = np.concatenate([
        np.expand_dims(generate_set_of_lines(
            n, offset=np.random.normal(0, 100, size=(1, 2)) * np.array([1, 0])),
            axis=1)
        for i in range(m)], axis=1)
    
    '''
    #!!
    a = np.pi/12
    sina = np.sin(a)
    cosa = np.cos(a)
    rotation = np.array([[cosa, -sina],[sina, cosa]])
    L_set[:, 1, 2:4] = np.dot(L_set[:, 0, 2:4], rotation.T)
    '''
    return L_set