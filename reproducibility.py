''' Module for reproducing the experimental results from pickle files. '''

import pickle
from os import path
from matplotlib import pyplot as plt
from median_coreset.drawing import plot_graphs

# TODO: plotting of the graphs from pickle
def results_from_pickle(data_path, name):
    n, m, k, data_type = name.split('_')[1:5]
    epsilons = pickle.load(open(data_path, "rb"))
    plot_graphs(epsilons, n, m, k, None, data_type, do_save=False)
    return epsilons
    
if __name__ == '__main__':
    load_path = 'results/results_f' #'results_g/california 2 2'
    save_path = load_path #"results_f"
    
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(load_path) if isfile(join(load_path, f))]
    for data_name in onlyfiles:
        fname, extension = path.splitext(data_name)
        if extension == '.p':
            data = results_from_pickle(
                path.join(load_path, data_name), data_name)
            plt.savefig(path.join(
                save_path, fname + "_upd.png"), dpi=300)
