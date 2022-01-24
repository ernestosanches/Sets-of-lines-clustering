import numpy as np
from generation import (generate_colored_points_sets_3d_cloud,
                        read_colors_3d_cloud, get_3d_cloud_path)
from coresets import coreset, CS_dense
from visualization import visualize_coreset_points_3d

def remove_outliers(data_path, quantile, k=1, do_visualize=False):
    P = generate_colored_points_sets_3d_cloud(None, 1, data_path)
    colors = read_colors_3d_cloud()
    sensitivities = coreset(P, k, f_dense=CS_dense)
    sensitivities = np.minimum(sensitivities, 1) 
    threshold = np.quantile(sensitivities, quantile)
    idx_keep = sensitivities <= threshold
    P_filtered = P[idx_keep]
    print("Outliers removal. Total points: {}, filtered: {}".format(
        len(P), len(P_filtered)))
    if do_visualize:
        visualize_coreset_points_3d(P, sensitivities, colors=None,
                                    threshold=threshold, mul=100)
        visualize_coreset_points_3d(P, sensitivities, colors=colors,
                                    threshold=threshold)
    return P_filtered

    
if __name__ == "__main__":
    data_path = get_3d_cloud_path()
    P_filtered = remove_outliers(data_path, quantile=0.95, do_visualize=True)
