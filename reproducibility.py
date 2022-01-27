import pickle

# TODO: plotting of the graphs from pickle
def results_from_pickle(data_path):
    data = pickle.load(open(data_path, "rb"))
        
    
if __name__ == '__main__':
    data = results_from_pickle(
        "results/coreset_500_1_1_Random lines_Thu Jan 27 00:28:57 2022.p")
