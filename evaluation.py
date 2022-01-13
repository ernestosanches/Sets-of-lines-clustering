''' Coreset evaluation and comparison with random sampling.
    Also allows testing of intermediate algorithms. '''


import numpy as np
from collections.abc import Iterable
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from median import (
    robust_median, recursive_robust_median,
    closest_lines, closest_points, dist_points,
    closest_point_sets, pack_lines, unpack_lines,
    pack_colored_points, dist_colored_points, dist_lines_min_set_to_set,
    closest_colored_points, unpack_colored_points,
    closest_colored_point_sets, dist_colored_points_min_set_to_set,
    dist_colored_points_min_set_to_point, dist_colored_points_min_p_to_set,
    enumerate_set_of_sets, closest_colored_point_sets_to_points)

from generation import (
    generate_points, generate_colored_points, generate_colored_points_sets,
    generate_points_sets, generate_set_of_lines, generate_set_of_sets_of_lines)
from drawing import (
    draw_points, draw_colored_point_set, draw_colored_point_sets,
    draw_colored_point_sets_all, draw_colored_points, draw_lines,
    draw_line_set, draw_lines_from_points, draw_lines_set_of_sets,
    draw_set_of_sets_of_lines, draw_point_set)

from coresets import (CS_dense, Grouped_sensitivity, LS_dense, 
                      coreset, coreset_sample)



''' Testing functions for intermediate algorithms '''

def test_points_median(n=2000):
    P = generate_points(n)
    gamma = 1/5.0
    #C, cost = closest_points(P, Q, gamma)
    #draw_points(P, Q, C, "C = closest(P,Q)")
    k = 10
    do_recursive=True
    if do_recursive:
        Q, center, cost_med = recursive_robust_median(P, k, tau=1/10, delta=1/10,
                                                   dist_f=dist_points)
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
        Q, center, cost_med = recursive_robust_median(P, k, tau=1/10, delta=1/10,
                                                   dist_f=dist_colored_points)
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
        Q, center, cost_med = recursive_robust_median(
            P, k, tau=1/10, delta=1/10, 
            dist_f=dist_colored_points_min_set_to_set)
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
        center, q, cost_med = recursive_robust_median(
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
        Q, center, cost_med = recursive_robust_median(L, k, tau=1/10, delta=1/10,
                                                   dist_f=dist_lines)
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

''' Testing functions for main algorithms from the paper ''' 

def test_cs_dense():
    n = 5000
    m = k = 2
    #gamma = 1/5.0
    P = generate_colored_points_sets(n, m)
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
    sensitivities = coreset(L, k)
    return L, sensitivities

from text import load_text_data

def get_coreset_points(n, m, k, USE_TEXT):
    #USE_TEXT = False #(m==3)
    if USE_TEXT:
        #assert(m==3) # for text data
        if m == 3:
            print("Loading data (m = 3)...")
            P = load_text_data(n)
            print("Data loaded.")
        elif m == 1:
            print("Loading data (m = 1)...")
            P = load_text_data(n // 3)
            P = P.reshape((-1, 1, P.shape[-1]))
            P[:, :, -1] = 0
            print("Data loaded.")
        else:
            assert False, "Incorrect m: {}".format(m)
    else:
        P = generate_colored_points_sets(n, m)
    sensitivities = coreset(P, k, f_dense=CS_dense)
    return P, sensitivities

def test_coreset(L, sensitivities, title, do_draw, sizes):
    if do_draw:
        is_lines = L[0][0].shape[-1] == 4
        if is_lines:
            f_draw = draw_set_of_sets_of_lines
        else:
            f_draw = draw_colored_point_sets
    result = []
    if do_draw:
        plt.figure()
        plt.title("Original data sensitivities")
        f_draw(L, s=50 * sensitivities)

    for size in sizes:
        print("{}: Sampling size: {}".format(title, size))
        Lm, Wm = coreset_sample(L, sensitivities, size=size)
        if do_draw:
            plt.figure()
            plt.title("{}: size {}".format(title, size))
            f_draw(L, s=1/4, marker="+")
            f_draw(np.asarray(list(Lm)), s=Wm/2)#10 * Wm / np.max(Wm))
        result.append((size, Lm, Wm))
    return L, result
    

''' Evaluation functions for comparison against random sampling ''' 

def cost_set_to_points(L_set_of_sets, Lw, P, dist_f=dist_lines_min_set_to_set):
    result = 0
    for L_set, L_set_w in zip(L_set_of_sets, Lw):
        result += L_set_w * dist_f(L_set, P)
    return result


from kmedians import kmedians, centroids_set_init
P_queries = []

def evaluate_lines(L, sensitivities, size, k, n_samples, sample_f):
    DO_K_MEDIANS = False
    if DO_K_MEDIANS:
        pass
    else:
        all_lines = np.asarray([l for l_set in L for l in l_set])
        all_p, all_d = unpack_lines(all_lines)
        p_min = np.min(all_p - 100 * all_d, axis=0)
        p_max = np.max(all_p + 100 * all_d, axis=0)
        (x_min, y_min), (x_max, y_max) = p_min, p_max
        epsilons = []
        smallest = 1e-7
    
    def evaluate_sample():
        Lm, Wm = sample_f(L, sensitivities, size=size)
        x_rnd = np.random.uniform(x_min, x_max, (k, 1))
        y_rnd = np.random.uniform(y_min, y_max, (k, 1))
        p_rnd = np.hstack((x_rnd, y_rnd))
        cost = cost_set_to_points(L, np.ones(len(L)), p_rnd)
        cost_coreset = cost_set_to_points(Lm, Wm, p_rnd)
        if cost > smallest:
            epsilon = np.abs(cost - cost_coreset) / cost
        else:
            epsilon = 0   
        return epsilon
    
    epsilons = Parallel()([
        delayed(evaluate_sample)() for i in range(n_samples)])
    #epsilons = [
    #    evaluate_sample() for i in range(n_samples)]
    '''
    block_size = 10
    epsilons = [np.max(epsilons[i:i+block_size]) 
                for i in range(0, len(epsilons), block_size)]
    '''
    return epsilons, np.mean(epsilons), np.std(epsilons)


from kmedians import kmedians, centroids_set_init
P_queries = []
def evaluate_colored_points(L, sensitivities, size, k, n_samples, sample_f):
    global P_queries
    DO_K_MEDIANS = False
    if DO_K_MEDIANS:
        DO_COMPLETE_RANDOM = False
        #n_samples = 5
        if len(P_queries) == 0:
            #plt.figure()
            print("Building set of optimal queries")
            def do_work(i):
                print("K-medians query:", i)
                cost, centroids = kmedians(
                    L, k,
                    draw_f=draw_colored_point_sets,
                    draw_centroids_f=draw_colored_point_set,
                    dist_f=dist_colored_points_min_set_to_point,
                    enumerate_f=enumerate_set_of_sets,
                    dist_to_set_f=dist_colored_points_min_p_to_set,
                    centroids_init_f=centroids_set_init,
                    #centroids_init=np.array(
                    #    [[0, 10,0], [0,-10,1], [100,10,0], [100,-10,1]])
                    )
                return centroids
            P_queries = Parallel()(
                [delayed(do_work)(i) for i in range(n_samples)])
            print("Finished building set of optimal queries")
        pass
    else:
        DO_COMPLETE_RANDOM = False
        if DO_COMPLETE_RANDOM:
            all_points = np.asarray([p for p_set in L for p in p_set])
            all_p, all_colors = unpack_colored_points(all_points)
            p_min = np.min(all_p, axis=0)
            p_max = np.max(all_p, axis=0)
            #(x_min, y_min), (x_max, y_max) = p_min, p_max
            color_min, color_max = np.min(all_colors), np.max(all_colors)
        else:
            n, m = len(L), len(L[0])
            P_queries = L# generate_colored_points_sets(n, m)
    
    epsilons = []
    smallest = 1e-7
    for i in range(n_samples):
        Lm, Wm = sample_f(L, sensitivities, size=size)
        if DO_COMPLETE_RANDOM:
            coordinates_rnd = np.random.uniform(p_min, p_max, size=(k,len(p_min)))
            #x_rnd = np.random.uniform(x_min, x_max, (k, 1))
            #y_rnd = np.random.uniform(y_min, y_max, (k, 1))
            color_rnd = np.random.randint(color_min, color_max + 1, (k, 1))
            p_rnd = np.hstack((coordinates_rnd, color_rnd))
            #p_rnd = np.hstack((x_rnd, y_rnd, color_rnd))
        else:
            if n_samples < len(P_queries):
                idx = np.random.randint(len(P_queries))
                p_rnd = np.asarray(P_queries[idx])
            else:
                p_rnd = P_queries
        cost = cost_set_to_points(L, np.ones(len(L)), p_rnd,
                                  dist_f=dist_colored_points_min_set_to_set)
        cost_coreset = cost_set_to_points(Lm, Wm, p_rnd,
                                  dist_f=dist_colored_points_min_set_to_set)

        if cost > smallest:
            epsilon = np.abs(cost - cost_coreset) / cost
        else:
            epsilon = 0

        epsilons.append(epsilon)
    '''
    block_size = 20
    epsilons = [np.max(epsilons[i:i+block_size]) 
                for i in range(0, len(epsilons), block_size)]
    '''
    return epsilons, np.mean(epsilons), np.std(epsilons)
    #return np.max(epsilons), 0# np.std(epsilons)#np.mean(epsilons), 0 #np.std(epsilons)
    #return np.mean(epsilons), np.std(epsilons)

def evaluate_points(L, sensitivities, size, k, n_samples):
    # TODO: generalize by creating a random generator
    pass
    
def plot_mu_sigma(x, mus, sigmas, color, label, n_samples):
    plt.plot(x, mus, label=label, marker="s", color=color, linewidth=2)
    sqrt_n = 1 #np.sqrt(n_samples)
    plt.fill_between(x, 
                     mus - sigmas / sqrt_n, 
                     mus + sigmas / sqrt_n, 
                     alpha = 0.1, color=color)

def plot_graphs(epsilons, n, m, k, n_samples, do_lines, use_text):
    (sizes, epsilon_mus, epsilon_sigmas,
     epsilon_mus_biased, epsilon_sigmas_biased,
     epsilon_random_mus, epsilon_random_sigmas) = map(
         np.asarray, zip(*epsilons))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.figure()
    plot_mu_sigma(sizes, epsilon_mus_biased, epsilon_sigmas_biased, 
                  colors[3], "Coreset (biased)", n_samples)
    plot_mu_sigma(sizes, epsilon_mus, epsilon_sigmas, 
                  colors[0], "Coreset (theoretical)", n_samples)
    plot_mu_sigma(sizes, epsilon_random_mus, epsilon_random_sigmas, 
                  colors[1], "Uniform sampling", n_samples)
    plt.plot(sizes, np.zeros_like(sizes), label="Full data",
             marker="s", color=colors[2], linewidth=2)
    plt.title("Approximation comparison (n = {}, m = {}, k = {}, lines = {}, text = {})".format(
        n, m, k, do_lines, use_text))
    plt.ylabel("Error ratio (epsilon)")
    plt.xlabel("Sample size")
    plt.xscale("log")
    #plt.yscale("log")
    plt.legend()