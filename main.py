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
    n = 2000
    m = 1
    k = 1
    n_samples = 30
    do_lines = True
    if do_lines:
        data_types = [Datasets.LINES_KDDCUP] 
    else: 
        data_types = [#Datasets.POINTS_CLOUD,
                      #Datasets.POINTS_REUTERS,
                      #Datasets.POINTS_SYNTHETIC
                     ]
    
    for data_type in data_types:
        print("\n{}: n = {}, m = {}, k = {}".format(
            data_type, n, m, k))
        sizes = np.logspace(1, 4, 10, dtype=int)
        sizes = sizes[sizes < n]
        
        ''' coreset testing ''' 
        L, sensitivities = run_coreset_set_of_sets(n, m, k, sizes, data_type,
                                                   n_samples)
