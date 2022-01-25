import pickle
import numpy as np
from time import ctime
from drawing import plot_graphs
from parameters import Datasets
from coresets import coreset_sample
from visualization import visualize_coreset
from evaluation import (
    test_points_median, test_lines_median, test_lines_closest,
    test_colored_points_median, test_colored_point_sets_to_sets_median,
    test_colored_point_sets_to_points_median, test_cs_dense,
    get_grouped_sensitivity, test_grouped_sensitivity, get_ls_dense, 
    test_ls_dense, get_coreset_lines, get_coreset_points, evaluate_lines,
    evaluate_colored_points, test_coreset)


def test_basic():
    ''' Testing basic algorithms '''
    test_points_median()
    test_lines_median()  
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
    
def get_coreset(n, m, k, data_type):
    print("Getting coreset")
    do_lines = data_type in Datasets.DATASETS_LINES 
    if do_lines:
        L, sensitivities = get_coreset_lines(n, m, k, data_type)
        evaluate_f = evaluate_lines
    else:
        L, sensitivities = get_coreset_points(n, m, k, data_type)
        evaluate_f = evaluate_colored_points
    pickle.dump((L, sensitivities), open(
        "results/coreset_{}_{}_{}_{}_{}.p".format(
            n, m, k, data_type, ctime()), "wb"))
    return L, sensitivities, evaluate_f

def do_test_coreset(L, sensitivities, sizes):
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

def evaluate_coreset(L, k, sensitivities, evaluate_f, sizes, data_type, n_samples):
    global epsilons
    print("Evaluating coreset")
    n, m = len(L), len(L[0])
    epsilons = []
    epsilons_all = []
    P_queries = []
    for size in sizes:
        print("Evaluating coreset of size:", size)
        epsilon_all, epsilon_mu, epsilon_sigma = evaluate_f(
            L, sensitivities, size, k, n_samples, coreset_sample, P_queries)
        epsilon_all_random, epsilon_random_mu, epsilon_random_sigma = evaluate_f(
            L, np.ones_like(sensitivities), size, k, n_samples, coreset_sample,
            P_queries)
        epsilons.append((size, epsilon_mu, epsilon_sigma,
                         epsilon_random_mu, epsilon_random_sigma))
        epsilons_all.append((size, epsilon_all,
                             epsilon_all_random))
    pickle.dump(epsilons, open(
        "results/epsilons_{}_{}_{}_{}_{}.p".format(
            n, m, k, data_type, ctime()), "wb"))
    plot_graphs(epsilons, n, m, k, n_samples, data_type)

# TODO: remove global variables
def run_coreset_set_of_sets(n, m, k, sizes, data_type, n_samples):
    global L, sensitivities, evaluate_f
    ''' Coreset on sets of colored points or lines '''
    # calculating the coreset
    L, sensitivities, evaluate_f = get_coreset(n, m, k, data_type)

    # uncomment for additional coreset visualization
    #result, result_random = do_test_coreset(L, sensitivities, do_lines, sizes)
    # evaluating the coreset and plotting comparison graph
    
    visualize_coreset(L, sensitivities, k, data_type)
    
    evaluate_coreset(L, k, sensitivities, evaluate_f, sizes, data_type, 
                     n_samples)
    return L, sensitivities

if __name__ == "__main__":
    pass
    #test_points_median()
    #test_lines_closest()  
    #test_lines_median()  
    #P, Q, C = test_colored_points_median()
    #P, Q, C = test_colored_point_sets_to_sets_median()
    #P, Q, C = test_colored_point_sets_to_points_median()
    