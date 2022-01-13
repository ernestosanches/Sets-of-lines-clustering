#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 16:14:41 2021

@author: ernesto
"""

import numpy as np
from collections import Counter
from collections.abc import Iterable
from median import (
    median_exhaustive, robust_median, recursive_robust_median,
    closest_lines, closest_points, dist_lines, dist_points,
    closest_point_sets, pack_lines, unpack_lines, dist_points_min_set_to_set,
    pack_colored_points, dist_colored_points, dist_lines_min_set_to_set,
    closest_colored_points, unpack_colored_points,
    closest_colored_point_sets, dist_colored_points_min_set_to_set,
    dist_colored_points_min_set_to_point, dist_colored_points_min_p_to_set,
    enumerate_set_of_sets, closest_colored_point_sets_to_points)

from median_sets_of_sets import (CS_dense, Grouped_sensitivity, LS_dense, 
                                 coreset, coreset_sample, intersect_spheres,
                                 project_line, project_point)

from matplotlib import pyplot as plt

def generate_points(n):
    mu1, mu2 = np.random.normal(0, 5, 2), np.random.normal(0, 5, 2)
    P = np.random.multivariate_normal(mu1 - np.array([150, 50]),
                                      np.eye(2) * 2, int(n * 1/20))
    Q = np.random.multivariate_normal(mu2 + np.array([130, 0]), 
                                      np.eye(2) * 2, int(n * 1/20))
    R = np.random.multivariate_normal([0,0], np.eye(2) * 5, 
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

def generate_colored_points_sets(n, m):
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
    return stack_point_sets(colored_points)

def generate_set_of_lines(n, offset=np.zeros(2)):
    c = generate_points(n) + offset.reshape((1,2))
    d = generate_points(n)
    np.random.shuffle(d)
    d = d / np.linalg.norm(d, axis=-1)[:, np.newaxis]
    return pack_lines(c, d)

def generate_set_of_sets_of_lines(n, m):
    L_set = np.concatenate([
        np.expand_dims(generate_set_of_lines(
            n, offset=np.random.normal(size=(2)) * 5), axis=1)
        for i in range(m)], axis=1)
    return L_set

def draw_set_of_sets_of_lines(L_set, s=2):
    n, m, d2 = L_set.shape
    colors = get_color_list()
    for i in range(m):
        L = L_set[:, i, :]
        draw_line_set(L, color=colors[i % len(colors)], s=s)
        
def draw_point_set(P, label=None, color=None, s=4):
    plt.scatter(P[:, 0], P[:, 1], label=label, c=color, s=s, zorder=10)

def get_color_list():
    return np.array(["blue", "green", "red", "cyan"])

def draw_colored_point_set(P, label=None, s=4):
    colors = get_color_list()
    p, pcolor = unpack_colored_points(P)
    pcolor = np.asarray(pcolor, dtype=int).flatten()
    draw_point_set(p, color=colors[pcolor], label=label, s=s)

def draw_colored_point_sets(P_set, label=None, s=2, linecolor="blue", 
                            linealpha=0.2):
    n, m, d = P_set.shape
    # lines
    for i in range(n):
        P = P_set[i, :, :]
        x = P[:, 0]
        y = P[:, 1]
        plt.plot(x, y, linewidth=s, c=linecolor, alpha=linealpha)
    # points
    for i in range(m):
        P = P_set[:, i, :]
        p, pcolor = unpack_colored_points(P)
        draw_colored_point_set(P, label=label if i==0 else None, s=s*5)
        
 
def draw_colored_point_sets_all(P, Q, C, title):
    plt.figure()
    draw_colored_point_sets(P, "P", linecolor="blue")
    draw_colored_point_sets(Q, "Q", s=5, linealpha=0.4, linecolor="green")
    draw_colored_point_sets(C, "C", s=2, linealpha=0.4, linecolor="orange")
    plt.title(title)
    plt.legend()    
        
def draw_points(P, Q, C, title):
    plt.figure()
    draw_point_set(P, "P")
    draw_point_set(Q, "Q", s=100)
    draw_point_set(C, "C")
    plt.title(title)
    plt.legend()    
    
def draw_colored_points(P, Q, C, title):   
    plt.figure()
    draw_colored_point_set(P, "P")
    draw_colored_point_set(Q, "Q", s=100)
    draw_colored_point_set(C, "C", s=10)
    plt.legend()    


def draw_lines_from_points(points1, points2, color=None, s=2, alpha=1):
    # TODO: check new 3D shape format
    #points1, points2 = (np.asarray([item[0] for item in P]),
    #                    np.asarray([item[1] for item in P]))
    s = np.asarray(s)
    grid = np.dstack((points1, points2))
    if s.ndim == 0:
        plt.plot(grid[:,0,:].T, grid[:,1,:].T, c=color, linewidth=s,
                 alpha=alpha)
    else:
        for i in range(len(grid)):
            plt.plot(grid[i,0,:].T, grid[i,1,:].T, c=color, linewidth=s[i],
                     alpha=alpha)

def draw_line_set(lines, color, s=2):
    def draw(c, d, s, idx, set_alpha=None):
        def alpha(a):
            return a if set_alpha is None else set_alpha * a
        c, d, s = c[idx], d[idx], s[idx]
        draw_lines_from_points(c - 2 * d, c + 2 * d, color, s, 
                               alpha=alpha(0.05))
        draw_lines_from_points(c - 5 * d, c + 5 * d, color, s, 
                               alpha=alpha(0.02))
        draw_lines_from_points(c - 10 * d, c + 10 * d, color, s,
                               alpha=alpha(0.01))
    #print(lines.shape, color)
    c, d = unpack_lines(lines)
    if not isinstance(s, Iterable):
        s = np.full(len(lines), s)
    #print(Counter(s))
    idx_normal = s <= 2
    idx_emph = s > 2
    draw(c, d, s, idx_normal, None)
    draw(c, d, s, idx_emph, 10)

def draw_lines(L, Q, C, title):
    #print(L.shape, Q.shape, C.shape)
    plt.figure()
    draw_line_set(L, color="blue")
    draw_point_set(Q, color="orange", s=100)
    draw_line_set(C, color="green")
    plt.title(title)

def draw_lines_set_of_sets(L, widths=None):
    if widths is None:
        widths = 2
    colors = get_color_list()
    n, m, d = L.shape
    for i in range(m):
        draw_line_set(L[:, i, :], colors[i], s=widths)

def test_points_median(n=2000):
    P = generate_points(n)
    gamma = 1/5.0
    #C, cost = closest_points(P, Q, gamma)
    #draw_points(P, Q, C, "C = closest(P,Q)")
    k = 10
    do_recursive=False
    if do_recursive:
        center, cost_med = recursive_robust_median(P, k, tau=1/10, delta=1/10,
                                                   dist_f=dist_points)
        Q = center 
    else:
        center, cost_med = robust_median(P, k, delta=1/10,
                                         dist_f=dist_points)
        Q = np.array(center)[np.newaxis, :]
    C, cost = closest_points(P, Q, gamma)
    draw_points(P, Q, C, "Q = Median(P); C = Closest(P, Q)")
    
    

def test_colored_points_median(n=2000):
    P_points = generate_points(n)
    P_color = np.abs(np.asarray(P_points.sum(
        axis=1, keepdims=True), dtype=int)) % 3
    P = pack_colored_points(P_points, P_color)
    #plt.figure()
    #plt.scatter(P[:, 0], P[:, 1], c=P_color)
    
    gamma = 1/5.0
    k = 10
    do_recursive=True
    if do_recursive:
        center, cost_med = recursive_robust_median(P, k, tau=1/10, delta=1/10,
                                                   dist_f=dist_colored_points)
        Q = center 
    else:
        center, cost_med = robust_median(P, k, delta=1/10,
                                         dist_f=dist_colored_points)
        Q = np.array(center)[np.newaxis, :]
    C, cost = closest_colored_points(P, Q, gamma)
    draw_colored_points(P, Q, C, "Q = Median(P); C = Closest(P, Q)")
    return P, Q, C

def test_colored_point_sets_to_sets_median(n=50):
    P = generate_colored_points_sets(n)
    #draw_colored_point_sets(P)
   
    gamma = 1/5.0
    k = 10
    do_recursive=False
    if do_recursive:
        center, cost_med = recursive_robust_median(
            P, k, tau=1/10, delta=1/10, 
            dist_f=dist_colored_points_min_set_to_set)
        Q = center 
    else:
        center, cost_med = robust_median(
            P, k, delta=1/10,
            dist_f=dist_colored_points_min_set_to_set)
        Q = np.array(center)[np.newaxis, :]
    C, cost = closest_colored_point_sets(P, Q, gamma)
    draw_colored_point_sets_all(P, Q, C, "Q = Median(P); C = Closest(P, Q)")
    return P, Q, C

def test_colored_point_sets_to_points_median(n=20):
    P = generate_colored_points_sets(n)
    #draw_colored_point_sets(P)
   
    gamma = 1/5.0
    k = 10
    do_recursive=False
    if do_recursive:
        center, cost_med = recursive_robust_median(
            P, k, tau=1/10, delta=1/10, 
            dist_f=dist_colored_points_min_set_to_point,
            enumerate_f=enumerate_set_of_sets)
    else:
        center, cost_med = robust_median(
            P, k, delta=1/10,
            dist_f=dist_colored_points_min_set_to_point,
            enumerate_f=enumerate_set_of_sets,
            dist_to_set_f=dist_colored_points_min_p_to_set)
    Q = np.expand_dims(center, axis=[0]) # set of points  from single point
    C, cost = closest_colored_point_sets_to_points(P, Q, gamma)
    Q = np.expand_dims(Q, axis=[0]) # set of sets from set of points
    draw_colored_point_sets_all(P, Q, C, "Q = Median(P); C = Closest(P, Q)")
    return P, Q, C



def test_lines_median(n=100):
    L = generate_set_of_lines(n)
    gamma = 1/5.0
    
    Q = np.array([[-1,-1], [1,1]])
    C, cost = closest_lines(L, Q, gamma)
    draw_lines(L, Q, C, "Q = [[-1,-1], [1,1]]; C = closest(L, Q)")
    '''
    k = 10
    do_recursive=False
    if do_recursive:
        center, cost_med = recursive_robust_median(L, k, tau=1/10, delta=1/10,
                                                   dist_f=dist_lines)
        Q = center 
    else:
        center, cost_med = robust_median(L, k, delta=1/10,
                                     dist_f=dist_lines)
        Q = np.array(center)[np.newaxis, :]
    C, cost = closest_lines(L, Q, gamma)
    draw_lines(L, Q, C, "Q = Median(P); C = Closest(P, Q)")
    '''
def test_point_sets():
    n=100
    P = generate_points_sets(n)
    m = len(P[0])
    plt.figure()

    #draw_lines_from_points(P, color="blue")
    #k=10
    #res = CS_dense(P, k)
    
    # dist_min_set_to_set
    gamma = 1/5.0

    Q = np.asarray([(np.array([-1,-1]), np.array([1,1]))])
    C, cost = closest_point_sets(P, Q, gamma)
    #draw_lines(L, Q, C, "Q = [[-1,-1], [1,1]]; C = closest(L, Q)")
    draw_lines_from_points(P, color="blue")
    draw_lines_from_points(C, color="orange")
    draw_lines_from_points(Q, color="green", s=5)
    return P, Q, C

def test_main():
    n=100
    k=10
    P = generate_points(n)
    Q = np.array([[-1,-1], [1,1]])
    Cp = project_point(P, Q)
    #plt.figure()
    #draw_lines_from_points(P, color="blue")
    #res = CS_dense(P, k)
    

def test_cs_dense():
    n = 2000
    k = 2
    #gamma = 1/5.0
    P = generate_colored_points_sets(n)
    #draw_colored_point_sets(P)
    Cd, Qd = CS_dense(P, k)
    #Cd, _ = closest_colored_point_sets_to_points(Pd, Qd, gamma)
    Qd = np.expand_dims(Qd, axis=0) # single set to set of sets for drawing
    draw_colored_point_sets_all(P, Qd, Cd, "C, Q = CS_dense(P, k)")
    return Cd, Qd

def generate_group_sensitivity_input(n, m, dim=2):
    def get_random_d():
        d = np.random.normal(size=(dim))
        return d / np.linalg.norm(d)
    P = generate_points(m)
    L = []
    for i in range(n):
        Li = []
        for p in P:
            Li.append(pack_lines(p, get_random_d()))
        L.append(Li)
    return np.asarray(L), P

def get_grouped_sensitivity():
    n = 2000
    k = 2
    m = 2
    #gamma = 1/5.0
    L, P = generate_group_sensitivity_input(n, m)
    #draw_lines_set_of_sets(L)
    #draw_point_set(P, s=400)
    s_lines, P_L, s = Grouped_sensitivity(L, P, k)
    return L, P, s_lines, P_L, s

def test_grouped_sensitivity(L, P, s_lines, P_L, s):
    plt.figure()
    plt.gca().set_aspect('equal')
    draw_lines_set_of_sets(L, widths=1 / (4 * np.asarray(s_lines)))
    draw_point_set(P, s=100, color="black")

def get_ls_dense():
    k = 2
    n = 1000
    m = 2
    L = generate_set_of_sets_of_lines(n, m)
    plt.figure()
    draw_set_of_sets_of_lines(L,s=1)
    Lm, Bm = LS_dense(L, k)
    return Lm, Bm

def test_ls_dense(Lm, Bm):
    draw_set_of_sets_of_lines(np.asarray(list(Lm)), s=10)
    draw_point_set(Bm, s=400, color="black")

def get_coreset_lines(n, m, k):
    L = generate_set_of_sets_of_lines(n, m)
    lines, sensitivities = coreset(L, k)
    return L, lines, sensitivities

def get_coreset_points(n, m, k):
    P = generate_colored_points_sets(n, m)
    lines, sensitivities = coreset(P, k, f_dense=CS_dense)
    return P, lines, sensitivities


def test_coreset(L, lines, sensitivities, title, do_draw=False):
    if do_draw:
        is_lines = L[0][0].shape[-1] == 4
        if is_lines:
            f_draw=draw_set_of_sets_of_lines
        else:
            f_draw=draw_set_of_sets_of_points
    result = []
    if do_draw:
        plt.figure()
        plt.title("Original data")
        f_draw(L,s=1)

    for size in [10, 20, 40, 80, 160, 320, 640, 1000, len(L) - 1]:
        print("{}: Sampling size: {}".format(title, size))
        Lm, Wm = coreset_sample(lines, sensitivities, size=size)
        if do_draw:
            plt.figure()
            plt.title("{}: size {}".format(title, size))
            f_draw(L,s=1/2)
            f_draw(np.asarray(list(Lm)), s=Wm/2)#10 * Wm / np.max(Wm))
        result.append((size, Lm, Wm))
    return L, result
    
# TODO:
# just check the closests over n,m sets
# check CS - dense -- should result in spheres?
# check spheres, Grouped - sens.
# check LS - dense
    
#reduces (1/k)**mk
#m=4 k=3
#min_n = 1/3**12
# n - 10000
# m - 2
# k - 2

def cost_set_to_points(L_set_of_sets, Lw, P, dist_f=dist_lines_min_set_to_set):
    result = 0
    for L_set, L_set_w in zip(L_set_of_sets, Lw):
        result += L_set_w * dist_f(L_set, P)
    return result



def evaluate_lines(L, Lm, Wm, k):
    n_trials = 30
    all_lines = np.asarray([l for l_set in L for l in l_set])
    all_p, all_d = unpack_lines(all_lines)
    p_min = np.min(all_p - 10 * all_d, axis=0)
    p_max = np.max(all_p + 10 * all_d, axis=0)
    (x_min, y_min), (x_max, y_max) = p_min, p_max
    epsilons = []
    for i in range(n_trials):
        x_rnd = np.random.uniform(x_min, x_max, (k, 1))
        y_rnd = np.random.uniform(y_min, y_max, (k, 1))
        p_rnd = np.hstack((x_rnd, y_rnd))
        cost = cost_set_to_points(L, np.ones_like(L), p_rnd)
        cost_coreset = cost_set_to_points(Lm, Wm, p_rnd)
        epsilon = np.abs(cost - cost_coreset) / cost
        epsilons.append(epsilon)
    return 1 + np.mean(epsilons), np.std(epsilons)


def evaluate_points(L, Lm, Wm, k):
    n_trials = 30
    all_points = np.asarray([p for p_set in L for p in p_set])
    all_p, all_colors = unpack_colored_points(all_points)
    p_min = np.min(all_p, axis=0)

def test_coreset(L, lines, sensitivities, title, do_draw=False):
    if do_draw:
        is_lines = L[0][0].shape[-1] == 4
        if is_lines:
            f_draw=draw_set_of_sets_of_lines
        else:
            f_draw=draw_set_of_sets_of_points
    result = []
    if do_draw:
        plt.figure()
        plt.title("Original data")
        f_draw(L,s=1)

    for size in [10, 20, 40, 80, 160, 320, 640, 1000, len(L) - 1]:
        print("{}: Sampling size: {}".format(title, size))
        Lm, Wm = coreset_sample(lines, sensitivities, size=size)
        if do_draw:
            plt.figure()
            plt.title("{}: size {}".format(title, size))
            f_draw(L,s=1/2)
            f_draw(np.asarray(list(Lm)), s=Wm/2)#10 * Wm / np.max(Wm))
        result.append((size, Lm, Wm))
    return L, result
    p_max = np.max(all_p, axis=0)
    (x_min, y_min), (x_max, y_max) = p_min, p_max
    color_min, color_max = np.min(all_colors), np.max(all_colors)
    epsilons = []
    for i in range(n_trials):
        x_rnd = np.random.uniform(x_min, x_max, (k, 1))
        y_rnd = np.random.uniform(y_min, y_max, (k, 1))
        color_rnd = np.random.uniform(color_min, color_max, (k, 1))
        p_rnd = np.hstack((x_rnd, y_rnd, color_rnd))
        cost = cost_set_to_points(L, np.ones_like(L), p_rnd,
                                  dist_f=dist_colored_points_min_set_to_set)
        cost_coreset = cost_set_to_points(Lm, Wm, p_rnd,
                                  dist_f=dist_colored_points_min_set_to_set)
        epsilon = np.abs(cost - cost_coreset) / cost
        epsilons.append(epsilon)
    return 1 + np.mean(epsilons), np.std(epsilons)



    
def plot_mu_sigma(x, mus, sigmas, color, label):
    plt.plot(x, mus, label=label, marker="s", color=color, linewidth=2)
    plt.fill_between(x, mus - sigmas, mus + sigmas, alpha = 0.1, color=color)

def plot_graphs(epsilons, size):
    (sizes, epsilon_mus, epsilon_sigmas,
     epsilon_random_mus, epsilon_random_sigmas) = map(
         np.asarray, zip(*epsilons))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.figure()
    plot_mu_sigma(sizes, epsilon_mus, epsilon_sigmas, 
                  colors[0], "Coreset")
    plot_mu_sigma(sizes, epsilon_random_mus, epsilon_random_sigmas, 
                  colors[1], "Random sampling")
    plt.plot(sizes, np.ones_like(sizes), label="Full data",
             marker="s", color=colors[2], linewidth=2)
    plt.title("Approximation comparison (n = {})".format(size))
    plt.ylabel("Error ratio (1 + epsilon)")
    plt.xlabel("Sample size")
    plt.xscale("log")
    plt.legend()

# total sensitivity: log(n) * (2k)**(mk)

import pickle

if __name__ == "__main__":
    #test_points_median()
    #test_lines_median()  # update format
    # P, Q, C = test_point_sets() # update drawing similar to colored sets
    #P, Q, C = test_colored_points_median()
    #P, Q, C = test_colored_point_sets_to_sets_median()
    #P, Q, C = test_colored_point_sets_to_points_median()
    #Cd, Qd = test_cs_dense()
    
    #L, P, s_lines, P_L, s = get_grouped_sensitivity()
    #test_grouped_sensitivity(L, P, s_lines, P_L, s)
    
    ###
    '''
    Lm, Bm = get_ls_dense()
    test_ls_dense(Lm, Bm)   
    '''
    
    DO_COMPUTE = False
    DO_LINES = True
    DO_DRAW = True
    n = 2000
    m = 2
    k = 2
    
    print("Get coreset"))
    if DO_LINES:
        L, lines, sensitivities = get_coreset_lines(n, m, k)
        evaluate = evaluate_lines
    else:
        L, lines, sensitivities = get_coreset_points(n, m, k)
        evaluate = evaluate_points
    
    pickle.dump((L, lines, sensitivities), open(
        "results/coreset_{}_{}_{}_{}.p".format(n, m, k, int(DO_LINES)), "wb"))
        
    print("Test coreset")
    _, result = test_coreset(L, lines, sensitivities, "Coreset", DO_DRAW)
    _, result_random = test_coreset(
        L, lines, np.ones_like(sensitivities), "Random", DO_DRAW)

    
    pickle.dump(result, open(
        "results/result_{}_{}_{}.p".format(n, k, int(DO_LINES)), "wb"))
    pickle.dump(result_random, open(
        "results/result_random_{}_{}_{}_{}.p".format(n, m, k, int(DO_LINES)), "wb"))
     
    print("Evaluate coreset")
    epsilons = []
    epsilons_random = []
    for i in range(len(result)):
        size, Lm, Wm = result[i]
        print("Evaluating coreset of size:", size)
        epsilon_mu, epsilon_sigma = evaluate(L, Lm, Wm, k)
        _, Lm_random, Wm_random = result_random[i]
        epsilon_random_mu, epsilon_random_sigma = evaluate(
            L, Lm_random, Wm_random, k)
        epsilons.append((size, epsilon_mu, epsilon_sigma,
                         epsilon_random_mu, epsilon_random_sigma))
    pickle.dump(epsilons, open(
        "results/epsilons_{}_{}_{}_{}.p".format(n, m, k, int(DO_LINES)), "wb"))

    plot_graphs(epsilons, len(L))
    