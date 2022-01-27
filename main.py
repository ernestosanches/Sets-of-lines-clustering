''' Main function that tests the coreset and plots a graph of 
    error comparison with random sampling '''

import numpy as np
from os import mkdir, path
from tests import run_coreset_set_of_sets
from parameters import Datasets

if __name__ == "__main__":
    if not path.exists("results"):
        mkdir("results")
    # parameters
    n = 100 # total data size
    m = 1    # size of each set in the data
    k = 2 # k centers
    n_samples = 50 # how many times experiment for each graph point is repeated
    do_lines = True
    
    if do_lines:
        data_types = [Datasets.LINES_RANDOM,
                      #Datasets.LINES_PERPENDICULAR,
                      #Datasets.LINES_KDDCUP,
                      #Datasets.LINES_COVTYPE,
                      ]
    else:
        data_types = [
                      Datasets.POINTS_RANDOM,
                      #Datasets.POINTS_REUTERS,
                      #Datasets.POINTS_COVTYPE,
                      #Datasets.POINTS_CLOUD,
                     ]
    
    for data_type in data_types:
        print("\nExperiment on {}: n = {}, m = {}, k = {}".format(
            data_type, n, m, k))
        sizes = np.logspace(1, 4, 10, dtype=int)
        sizes = sizes[sizes < n]
        
        ''' coreset testing ''' 
        L, sensitivities = run_coreset_set_of_sets(n, m, k, sizes, data_type,
                                                   n_samples)
