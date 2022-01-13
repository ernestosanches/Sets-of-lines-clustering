''' Main function that tests the coreset and plots a graph of 
    error comparison with random sampling '''

import pickle
import numpy as np
from os import mkdir, path
from matplotlib import pyplot as plt

from evaluation import (
    test_points_median, test_lines_median, test_point_sets,
    test_colored_points_median, test_colored_point_sets_to_sets_median,
    test_colored_point_sets_to_points_median, test_cs_dense,
    get_grouped_sensitivity, test_grouped_sensitivity, get_ls_dense, 
    test_ls_dense, get_coreset_lines, get_coreset_points, evaluate_lines,
    evaluate_points, evaluate_colored_points, plot_graphs, test_coreset)

from coresets import coreset_sample, coreset_sample_biased

def test_basic():
    ''' Testing basic algorithms '''
    pass
    '''
    test_points_median()
    test_lines_median()  
    P, Q, C = test_point_sets() 
    P, Q, C = test_colored_points_median()
    P, Q, C = test_colored_point_sets_to_sets_median()
    P, Q, C = test_colored_point_sets_to_points_median()
    '''
    
def test_intermediate():
    ''' Testing main paper intermediate algorithms '''
    pass
    '''
    Cd, Qd = test_cs_dense()
    L, P, s_lines, P_L, s = get_grouped_sensitivity()
    test_grouped_sensitivity(L, P, s_lines, P_L, s)
    Lm, Bm = get_ls_dense()
    test_ls_dense(Lm, Bm)  
    '''
    
def get_coreset(n, m, k, do_lines, use_text):
    print("Getting coreset")
    if do_lines:
        L, sensitivities = get_coreset_lines(n, m, k)
        evaluate_f = evaluate_lines
    else:
        L, sensitivities = get_coreset_points(n, m, k, use_text)
        evaluate_f = evaluate_colored_points
    pickle.dump((L, sensitivities), open(
        "results/coreset_{}_{}_{}_{}.p".format(n, m, k, int(do_lines)), "wb"))
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
                     do_lines, use_text):
    global epsilons
    print("Evaluating coreset")
    n, m = len(L), len(L[0])
    epsilons = []
    epsilons_all = []
    n_samples = 100#15**2#15**2
    for size in sizes:
        print("Evaluating coreset of size:", size)
        epsilon_all, epsilon_mu, epsilon_sigma = evaluate_f(
            L, sensitivities, size, k,
            n_samples, coreset_sample)
        epsilon_all_biased, epsilon_mu_biased, epsilon_sigma_biased = evaluate_f(
            L, sensitivities, size, k,
            n_samples, coreset_sample_biased)
        epsilon_all_random, epsilon_random_mu, epsilon_random_sigma = evaluate_f(
            L, np.ones_like(sensitivities), size, k, n_samples, coreset_sample)
        epsilons.append((size, epsilon_mu, epsilon_sigma,
                         epsilon_mu_biased, epsilon_sigma_biased,
                         epsilon_random_mu, epsilon_random_sigma))
        epsilons_all.append((size, epsilon_all, epsilon_all_biased,
                             epsilon_all_random))
    pickle.dump(epsilons, open(
        "results/epsilons_{}_{}_{}_{}_{}_{}.p".format(
            n, m, k, int(do_lines), use_text, ctime()), "wb"))
    plot_graphs(epsilons, n, m, k, n_samples, do_lines, use_text)


def run_coreset_points(n, m, k, sizes):
    ''' Coreset on simple points (not sets). 
        Reproduces a recursive robust median coreset '''
    pass


def run_coreset_set_of_sets(n, m, k, sizes, do_lines, use_text):
    global L, sensitivities, evaluate_f
    ''' Coreset on sets of colored points or lines '''
    # calculating the coreset
    L, sensitivities, evaluate_f = get_coreset(n, m, k, do_lines, use_text)
    '''
    import pickle
    L, sensitivities = pickle.load(open("data_1.p", "rb"))
    evaluate_f = evaluate_colored_points
    '''
    # uncomment for coreset visualization
    #result, result_random = do_test_coreset(L, sensitivities, do_lines, sizes)
    # evaluating the coreset and plotting comparison graph
    evaluate_coreset(L, k, sensitivities, evaluate_f, sizes, do_lines, use_text)

from time import ctime

'''
if 0:
    import pickle
    from time import ctime
    d = (L, sensitivities)
    pickle.dump(d, open("data_counter_text_n10000_m3_k3.p".format(ctime()), "wb"))
'''

if __name__ == "__main__":
    if not path.exists("results"):
        mkdir("results")

    # parameters
    n = 1000
    
    m = 1
    k = 1
    do_lines = False
    use_text = False
    
    print("n = {}, m = {}, k = {}, lines = {}, text = {}".format(
        n, m, k, do_lines, use_text))
    '''
    sizes = [x for x in [2, 5, 10, 20, 40, 80, 160,
                         320,
                         640, 
                         1280,
                         2560,
                         5120,
                         #1000,
                         #3000,
                         #n - 1
                         ]   if x < n]
    '''
    sizes = [int(2**i) for i in np.arange(5, int(np.log2(n)) + 1, 0.5)]
    ''' Basic and intermediate algorithms testing 
        Commented out because of focusing on final coreset recently '''
    #test_basic()
    #test_intermediate()
    
    ''' coreset testing ''' 
    #run_coreset_points(n, m, k, sizes)
    
    #run_coreset_set_of_sets(n, m, k, sizes, do_lines=False) # colored points
    run_coreset_set_of_sets(n, m, k, sizes, do_lines, use_text)  # lines 


'''
offs=0
MUL = 10
points = L[:,0,:]
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
p = ax.scatter(points[:,0+offs], points[:,1+offs], points[:,2+offs],
               c=np.minimum(sensitivities*MUL, 1), s=4)
cbar = fig.colorbar(p)
cbar.ax.set_ylabel("Sensitivity")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D model and sensitivities * {}".format(MUL))
fig.show()
'''

'''
# SENSITIVITY POINTS
P=L
MUL = 10
plt.figure(); 
plt.axes().set_aspect('equal')
plt.title("Sensitivities * {}, n = {}, m = {}, k = {}, lines = {}, text = {}".format(
    MUL, n, m, k, do_lines, use_text))
for i in range(P.shape[1]):
    plt.scatter(P[:,i,0], P[:,i,1], 
               c=np.minimum(sensitivities * MUL, 1), s=4)    
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")
# SENSITIVITY POINTS
plt.figure(); 
plt.axes().set_aspect('equal')
plt.title("Log sensitivities, n = {}, m = {}, k = {}, lines = {}, text = {}".format(
    n, m, k, do_lines, use_text))
for i in range(P.shape[1]):
    plt.scatter(P[:,i,0], P[:,i,1], 
               c=np.log(sensitivities), s=4)    
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")
# COLORED POINTS
plt.figure(); 
plt.axes().set_aspect('equal')
plt.title("Colored points, n = {}, m = {}, k = {}, lines = {}, text = {}".format(
    n, m, k, do_lines, use_text))
for i in range(P.shape[1]):
    plt.scatter(P[:,i,0], P[:,i,1], 
               c=P[:,i,-1], s=4, vmin=0, vmax=2, alpha=0.2)    
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")
'''


'''
import pandas as pd
USE_OURS = True
Q = 0.92

fpath = "/home/ernesto/projects/tracxpoint/sfm_postprocessing/"
fname = fpath + "points3DWithDescriptors_front (4).txt"
points = P[:,0,:]
colors = pd.read_csv(
    fname,
    delimiter="|", usecols=range(4,4+3), 
    header=0, names=["r", "g", "b"]).values / 255.0

if USE_OURS:
    sss = np.minimum(sensitivities, 1) 
    sss_threshold = np.quantile(sss, Q)
    idx = sss <= sss_threshold
else:    
    idx = np.ones(len(points), dtype=bool)
print("Total: {}, filtered: {}".format(len(points), sum(idx)))

###
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
p = ax.scatter(points[idx,0], points[idx,1], points[idx,2],
               c=colors[idx], s=4)
cbar = fig.colorbar(p)
cbar.ax.set_ylabel("Sensitivity")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D model and sensitivities. Quantile={}".format(Q))
fig.show()
'''