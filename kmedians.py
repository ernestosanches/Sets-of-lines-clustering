import numpy as np
from functools import partial
from median import (
    robust_median, dist_points, dist_colored_points_min_set_to_point,
    dist_lines_min_set_to_point, enumerate_set_of_sets,
    enumerate_set_of_sets_centroids)
from drawing import (
    draw_colored_point_set, draw_colored_point_sets, draw_point_set)

def random_init(P, k):
    idx = np.random.choice(len(P), k, replace=False)
    return P[idx]

def kmedians(P, k, epsilon = 0.01, delta=1/20.0, 
             centroids_init_f=random_init,
             centroids_init=None,
             draw_f=draw_point_set, draw_centroids_f=None,
             dist_f=dist_points, enumerate_f=enumerate):
    def find_assignment(centroids):
        groups = [[] for i in range(k)]
        for p in P:
            best_dist = np.inf
            best_idx = np.random.randint(0, k)
            for idx, centroid in enumerate(centroids):
                dist = dist_f(p, centroid)
                if dist < best_dist:
                    best_idx = idx
                    best_dist = dist
            groups[best_idx].append(p)
        return [np.asarray(group) for group in groups]
    def find_centroids(groups, centroids_prev):
        centroids = []
        costs = 0
        for group, centroid_prev in zip(groups, centroids_prev):
            if len(group) == 0:
                continue
            centroid, cost = robust_median(group, k, delta, dist_f, 
                                           enumerate_f,
                                           max_points=10) #, dist_to_set_f)
            centroids.append(centroid)
            costs += cost
        return np.sum(costs), centroids
    prev_cost = 0
    curr_cost = np.inf
    iteration = 0
    if centroids_init is None:
        centroids = centroids_init_f(P, k)
    else:
        centroids = centroids_init
    if draw_centroids_f is None:
       draw_centroids_f = draw_f
    while iteration < 1 or prev_cost - curr_cost >= epsilon * curr_cost:
        prev_cost = curr_cost
        groups = find_assignment(centroids)
        curr_cost, centroids = find_centroids(groups, centroids)
        
        iteration += 1
    print("KMedians iterations: {}, cost: {}, group sizes: {}".format(
        iteration, curr_cost, [len(g) for g in groups]))
    return curr_cost, centroids
   
def centroids_set_init(P, k, is_lines=False):
    n, m, d = P.shape
    centroids = []
    k_remaining, m_remaining = k, m
    for set_idx in range(m):
        count = k_remaining // m_remaining
        k_remaining -= count
        m_remaining -= 1
        idx = np.random.choice(len(P), count)
        centroids.append(P[idx, set_idx])
    result = np.concatenate(centroids, axis=0)
    if is_lines:
        result = result[..., :d//2]
    return result

if __name__ == "__main__":
    from generation import generate_data_set_of_sets
    from parameters import Datasets
    n = 500
    m = 2
    k = 4
    
    '''
    P = generate_data_set_of_sets(n, m, Datasets.POINTS_RANDOM)
    cost, centroids = kmedians(P, k,
                               draw_f=draw_colored_point_sets,
                               draw_centroids_f=draw_colored_point_set,
                               dist_f=dist_colored_points_min_set_to_point,
                               enumerate_f=enumerate_set_of_sets,
                               centroids_init_f=centroids_set_init,
                               #centroids_init=np.array(
                               #    [[0, 10,0], [0,-10,1], [100,10,0], [100,-10,1]])
                               )
    '''
    P = generate_data_set_of_sets(n, m, Datasets.LINES_RANDOM)
    cost, centroids = kmedians(P, k,
                               draw_f=None,
                               draw_centroids_f=None,
                               dist_f=dist_lines_min_set_to_point,
                               enumerate_f=enumerate_set_of_sets_centroids,
                               centroids_init_f=partial(
                                   centroids_set_init, is_lines=True)
                               )
