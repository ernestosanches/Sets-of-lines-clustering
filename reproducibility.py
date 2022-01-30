import pickle
from os import path
from drawing import plot_graphs
from matplotlib import pyplot as plt

# TODO: plotting of the graphs from pickle
def results_from_pickle(data_path, name):
    n, m, k, data_type = name.split('_')[1:5]
    epsilons = pickle.load(open(data_path, "rb"))
    plot_graphs(epsilons, n, m, k, None, data_type, do_save=False)
    return epsilons
    
if __name__ == '__main__':
    load_path = 'results_z' #'results_g/california 2 2'
    save_path = load_path #"results_f"
    
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(load_path) if isfile(join(load_path, f))]
    for data_name in onlyfiles:
        #data_name = 'epsilons_1000_3_2_California housing (lines)_Fri Jan 28 10-57-18 2022.p'
        data = results_from_pickle(path.join(load_path, data_name), data_name)
        plt.savefig(path.join(
            save_path, path.splitext(data_name)[0] + "_upd.png"), dpi=300)
