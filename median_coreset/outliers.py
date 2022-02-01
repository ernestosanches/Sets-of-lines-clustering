''' Using the coreset for outliers removal from 3d points cloud '''

import numpy as np
from median_coreset.generation import (generate_colored_points_sets_3d_cloud,
                        read_colors_3d_cloud, get_3d_cloud_path)
from median_coreset.coresets import coreset, CS_dense
from median_coreset.visualization import visualize_coreset_points_3d

def convert_points(P):
    ''' Adds a single color and converts points to sets of size m = 1 '''
    if P.ndim == 2:
        n = len(P)
        P = np.hstack((P, np.zeros((n, 1))))        
        P = np.expand_dims(P, axis=1)
    return P

def get_inlier_indices(P, quantile, k=1):
    ''' Gets indices of the inliers in the dataset P.
        quantile - fraction of the inliers from total number of points.
        k - precision parameter, k = 1 ... 3 usually are good values. '''
    assert P.ndim == 3
    sensitivities = coreset(P, k, f_dense=CS_dense)
    sensitivities = np.minimum(sensitivities, 1) 
    threshold = np.quantile(sensitivities, quantile)
    idx_keep = np.arange(len(P))[sensitivities <= threshold]
    return idx_keep, sensitivities, threshold

def visualize_inliers_outliers(P, sensitivities, threshold, k=1, colors=None):
    ''' Draws 3D points cloud with inliers and outliers marked by colors ''' 
    assert P.ndim == 3
    visualize_coreset_points_3d(P, k, sensitivities, colors=None,
                                threshold=threshold, mul=100, 
                                save_figure=False)
    if colors is not None:
        visualize_coreset_points_3d(P, k, sensitivities, colors=colors,
                                    threshold=threshold, save_figure=False)
    
        
if __name__ == "__main__":
    k = 1
    quantile = 0.95
    do_visualize = True
    data_path = get_3d_cloud_path()
    P = generate_colored_points_sets_3d_cloud(data_path)
    colors = read_colors_3d_cloud()
    idx_keep, sensitivities, threshold = get_inlier_indices(P, quantile, k)
    P_filtered = P[idx_keep]
    print("Outliers removal. Total points: {}, filtered: {}".format(
        len(P), len(P_filtered)))
    if do_visualize:
        visualize_inliers_outliers(P, sensitivities, threshold, k, colors)
