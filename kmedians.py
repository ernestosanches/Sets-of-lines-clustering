import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from median import (
    robust_median, dist_points, dist_colored_points_min_set_to_point,
    enumerate_set_of_sets, dist_colored_points_min_p_to_set)
from generation import (
    generate_points, generate_colored_points_sets)
from drawing import (
    draw_colored_point_set, draw_point_set, draw_colored_point_set, 
    draw_colored_point_sets)

def random_init(P, k):
    idx = np.random.choice(len(P), k, replace=False)
    return P[idx]

def kmedians(P, k, epsilon = 0.01, delta=1/20.0, 
             centroids_init_f=random_init,
             centroids_init=None,
             draw_f=draw_point_set, draw_centroids_f=None,
             dist_f=dist_points, enumerate_f=enumerate,
             dist_to_set_f=None):
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
                                           enumerate_f, dist_to_set_f)
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
        '''
        plt.cla()
        for group in groups:
            if len(group) > 0:
                draw_f(np.asarray(group))
        draw_centroids_f(np.asarray(centroids), s=500)
        plt.pause(0.05)
        '''
    print("KMedians iterations: {}, cost: {}, group sizes: {}".format(
        iteration, curr_cost, [len(g) for g in groups]))
    return curr_cost, centroids
   
def centroids_set_init(P, k):
    m = P.shape[1]
    centroids = []
    k_remaining, m_remaining = k, m
    for set_idx in range(m):
        count = k_remaining // m_remaining
        k_remaining -= count
        m_remaining -= 1
        idx = np.random.choice(len(P), count)
        centroids.append(P[idx, set_idx])
    return np.concatenate(centroids, axis=0)

if __name__ == "__main__":
    n = 500
    m=2
    k = 4
    P = generate_colored_points_sets(n, m)
    cost, centroids = kmedians(P, k,
                               draw_f=draw_colored_point_sets,
                               draw_centroids_f=draw_colored_point_set,
                               dist_f=dist_colored_points_min_set_to_point,
                               enumerate_f=enumerate_set_of_sets,
                               dist_to_set_f=dist_colored_points_min_p_to_set,
                               centroids_init_f=centroids_set_init,
                               #centroids_init=np.array(
                               #    [[0, 10,0], [0,-10,1], [100,10,0], [100,-10,1]])
                               )
    # np.array([[0,0], [100,0]]))
    