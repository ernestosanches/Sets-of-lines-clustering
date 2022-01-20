import pickle
import numpy as np
from time import ctime
from os import mkdir, path
from matplotlib import pyplot as plt

from evaluation import (
    test_points_median, test_lines_median, test_point_sets,
    test_colored_points_median, test_colored_point_sets_to_sets_median,
    test_colored_point_sets_to_points_median, test_cs_dense,
    get_grouped_sensitivity, test_grouped_sensitivity, get_ls_dense, 
    test_ls_dense, get_coreset_lines, get_coreset_points, evaluate_lines,
    evaluate_colored_points, test_coreset)
from drawing import plot_graphs

from coresets import coreset_sample


def test_basic():
    ''' Testing basic algorithms '''
    test_points_median()
    test_lines_median()  
    P, Q, C = test_point_sets() 
    P, Q, C = test_colored_points_median()
    P, Q, C = test_colored_point_sets_to_sets_median()
    P, Q, C = test_colored_point_sets_to_points_median()
    
def test_intermediate():
    ''' Testing main paper intermediate algorithms '''
    Cd, Qd = test_cs_dense()
    L, P, s_lines, P_L, s = get_grouped_sensitivity()
    test_grouped_sensitivity(L, P, s_lines, P_L, s)
    Lm, Bm = get_ls_dense()
    test_ls_dense(Lm, Bm)  
    
def get_coreset(n, m, k, do_lines, use_text, r=1, is_colored=True):
    print("Getting coreset")
    if do_lines:
        L, sensitivities = get_coreset_lines(n, m, k)
        evaluate_f = evaluate_lines
    else:
        L, sensitivities = get_coreset_points(n, m, k, use_text, r, is_colored)
        evaluate_f = evaluate_colored_points
    pickle.dump((L, sensitivities), open(
        "results/coreset_{}_{}_{}_{}_{}_{}.p".format(n, m, k, int(do_lines),
                                                     r, int(is_colored)), "wb"))
    return L, sensitivities, evaluate_f

def do_test_coreset(L, sensitivities, do_lines, sizes):
    print("Testing coreset")
    _, result = test_coreset(L, sensitivities, "Coreset", do_draw=True, 
                             sizes=sizes)
    #_, result_random = test_coreset(
    #    L, np.ones_like(sensitivities), "Random", do_draw=False, sizes=sizes)
    #pickle.dump(result, open(
    #    "results/result_{}_{}_{}.p".format(n, k, int(do_lines)), "wb"))
    #pickle.dump(result_random, open(
    #    "results/result_random_{}_{}_{}_{}.p".format(n, m, k, int(do_lines)), "wb"))
    result_random = None
    return result, result_random

def evaluate_coreset(L, k, sensitivities, evaluate_f, sizes, 
                     do_lines, use_text, r, is_colored):
    global epsilons
    print("Evaluating coreset")
    n, m = len(L), len(L[0])
    epsilons = []
    epsilons_all = []
    n_samples = 30#15**2
    for size in sizes:
        print("Evaluating coreset of size:", size)
        epsilon_all, epsilon_mu, epsilon_sigma = evaluate_f(
            L, sensitivities, size, k,
            n_samples, coreset_sample)
        #epsilon_all_biased, epsilon_mu_biased, epsilon_sigma_biased = evaluate_f(
        #    L, sensitivities, size, k,
        #    n_samples, coreset_sample_biased)
        epsilon_all_random, epsilon_random_mu, epsilon_random_sigma = evaluate_f(
            L, np.ones_like(sensitivities), size, k, n_samples, coreset_sample)
        epsilons.append((size, epsilon_mu, epsilon_sigma,
        #                 epsilon_mu_biased, epsilon_sigma_biased,
                         epsilon_random_mu, epsilon_random_sigma))
        epsilons_all.append((size, epsilon_all, #epsilon_all_biased,
                             epsilon_all_random))
    pickle.dump(epsilons, open(
        "results/epsilons_{}_{}_{}_{}_{}_{}.p".format(
            n, m, k, int(do_lines), use_text, ctime()), "wb"))
    plot_graphs(epsilons, n, m, k, n_samples, do_lines, use_text, r, is_colored)

def run_coreset_set_of_sets(n, m, k, sizes, do_lines, use_text, r=1, is_colored=True):
    global L, sensitivities, evaluate_f
    ''' Coreset on sets of colored points or lines '''
    # calculating the coreset
    L, sensitivities, evaluate_f = get_coreset(n, m, k, do_lines, use_text,
                                               r, is_colored)
    # uncomment for coreset visualization
    #result, result_random = do_test_coreset(L, sensitivities, do_lines, sizes)
    # evaluating the coreset and plotting comparison graph
    evaluate_coreset(L, k, sensitivities, evaluate_f, sizes, do_lines, use_text,
                     r, is_colored)
    return L, sensitivities