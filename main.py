''' Main function that tests the coreset and plots a graph of 
    error comparison with random sampling '''

import numpy as np
from os import mkdir, path
from tests import run_coreset_set_of_sets

if __name__ == "__main__":
    if not path.exists("results"):
        mkdir("results")

    # parameters
    n = 100
    comment = ""#"r = 1e6, no colors"
    m = 3
    k = 1
    r = 1e6
    is_colored = True
    do_lines = False
    use_text = True    
    print("n = {}, m = {}, k = {}, lines = {}, text = {}, r = {}, is_colored = {}".format(
        n, m, k, do_lines, use_text, r, is_colored))
    print(comment)
    sizes = np.logspace(1, 4, 10, dtype=int)
    sizes = sizes[sizes < n]
    
    ''' coreset testing ''' 
    L, sensitivities = run_coreset_set_of_sets(n, m, k, sizes, do_lines, use_text, r, is_colored)  # lines 
