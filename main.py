''' Main function that tests the coreset and plots a graph of 
    error comparison with random sampling. '''

import numpy as np
from time import time
from os import mkdir, path
from median_coreset.tests import run_coreset_set_of_sets
from median_coreset.parameters import Datasets

if __name__ == "__main__":
    if not path.exists("results"):
        mkdir("results")
    # parameters
    n = 150 # total data size
    m = 2 # size of each set in the data
    k = 2   # k centers
    n_samples = 300 # how many times experiment for each graph point is repeated
    do_lines = True
    
    if do_lines:
        data_types = [Datasets.LINES_RANDOM,
                      #Datasets.LINES_PERPENDICULAR,
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
        print("\n\n***\n\nExperiment on {}: n = {}, m = {}, k = {}, n_samples = {}\n\n***\n\n".format(
            data_type, n, m, k, n_samples))
        t_start = time()
        sizes = np.logspace(1, 3, 10, dtype=int)
        sizes = sizes[sizes < n]
        
        ''' coreset testing ''' 
        L, sensitivities = run_coreset_set_of_sets(n, m, k, sizes, data_type,
                                                   n_samples)
        print("\n***\nFinished [time={:.3f}]: {}: n = {}, m = {}, k = {}, n_samples = {}\n***\n".format(
            time() - t_start, data_type, n, m, k, n_samples))
