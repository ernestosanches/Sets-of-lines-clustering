import pickle
from drawing import plot_graphs

# TODO: plotting of the graphs from pickle
def results_from_pickle(data_path):
    epsilons = pickle.load(open(data_path, "rb"))
    plot_graphs(epsilons, n, m, k, n_samples, data_type)
    return epsilons
    
if __name__ == '__main__':
    data = results_from_pickle(
        "results/coreset_500_1_1_Random lines_Thu Jan 27 00:28:57 2022.p")
