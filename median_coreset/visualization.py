import numpy as np
import pandas as pd
from time import ctime
from matplotlib import pyplot as plt
from matplotlib import cm
from .drawing import draw_line_set
from .parameters import Datasets


def _get_colormap():
    #from matplotlib.colors import LinearSegmentedColormap
    '''return LinearSegmentedColormap.from_list(
        "custom",
        [(0.0, '#0000ff'), 
         (1.0, '#cc0000')])
    '''
    return cm.jet

def _draw_sensitivities(P, sensitivities, data_type, offset):
    cmap = _get_colormap()
    def draw_points(P_set, s):
        #plt.axes().set_aspect('equal')
        for i in range(P_set.shape[1]):
            plt.scatter(P_set[:, i, 0 + offset], 
                        P_set[:, i, 1 + offset], c=s, s=4, cmap=cmap)            
    def draw_lines(L_set, s):
        n, m, d2 = L_set.shape
        #plt.axes().set_aspect('equal')
        for i in range(n):
            L = L_set[i, :, :]
            draw_line_set(L, color=cmap(s[i])[:3])
    if data_type in Datasets.DATASETS_POINTS:
        draw_points(P, sensitivities)
    elif data_type in Datasets.DATASETS_LINES:
        draw_lines(P, sensitivities)
    else:
        assert False, "Incorrect data type: {}".format(data_type)

def visualize_coreset_sensitivities(P, sensitivities, k, data_type, 
                                    apply_log=False, feature_offset=0,
                                    mul=None):
    if mul is None:
        mul = 1 #if data_type in Datasets.DATASETS_POINTS else 10
    if apply_log:
        sensitivities = np.log(sensitivities)
    else:
        sensitivities = np.minimum(mul * sensitivities, 1)
        
    n, m, d = P.shape
    plt.figure(); 
    plt.title("{}: {}, n = {}, m = {}, k = {}".format( 
        data_type,
        "Log sensitivities" if apply_log 
            else "Sensitivities{}".format(" * {}".format(mul) if mul != 1
                                          else ""), 
        n, m, k))
    _draw_sensitivities(P, sensitivities, data_type, feature_offset)
    sm = plt.cm.ScalarMappable(cmap=cm.jet, 
                               norm=plt.Normalize(vmin=sensitivities.min(), 
                                                  vmax=sensitivities.max()))
    plt.colorbar(sm)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("results/sensitivities_coreset_n{}_m{}_k{}_{}_L{}_mul{}, {}.png".format(
        n, m, k, data_type, int(apply_log), mul,
        ctime().replace(':', '-')), dpi=300)

    

def visualize_points_colors(P, k, data_type, feature_offset=0):
    n, m, d = P.shape
    plt.figure(); 
    #plt.axes().set_aspect('equal')
    plt.title("{}: Colored points, n = {}, m = {}, k = {}".format(
        data_type, n, m, k))
    cmin = P[:,:,-1].min()
    cmax = P[:,:,-1].max()
    for i in range(P.shape[1]):
        plt.scatter(P[:, i, 0 + feature_offset], P[:, i, 1 + feature_offset], 
                   c=P[:,i,-1], s=4, vmin=cmin, vmax=cmax, alpha=0.2)    
    plt.colorbar()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("results/colors_coreset_n{}_m{}_k{}_{}_{}.png".format(
        n, m, k, data_type, ctime().replace(':', '-')), dpi=300)

def visualize_coreset(P, sensitivities, k, data_type, feature_offset=0, 
                      mul=None):
    for apply_log in (False, True):
        muls = [1] if apply_log else [1,5,10,25] 
        for mul in muls:
            visualize_coreset_sensitivities(
                P, sensitivities, k, data_type, apply_log, feature_offset, mul)
    if data_type in Datasets.DATASETS_POINTS:
        visualize_points_colors(P, k, data_type, feature_offset)
        d = P.shape[-1]
        if d >= 3+1:
            visualize_coreset_points_3d(P, k, sensitivities, data_type=data_type, 
                                        feature_offset=feature_offset)   
    plt.show()
    plt.pause(0.001)

##################################
# Outliers removal visualization #
##################################

def visualize_coreset_points_3d(P, k, sensitivities, threshold=1, 
                                data_type=None, feature_offset=0, 
                                colors=None, mul=1):
    def draw_3d(ax, points, idx, offset, sensitivities, 
                colors, s, alpha, label):
        c = (np.minimum(sensitivities[idx] * mul, 1) if colors is None 
             else colors[idx])
        ax.scatter(
            points[idx, 0 + offset], points[idx, 1 + offset], 
            points[idx, 2 + offset], c=c, s=s, alpha=alpha, label=label,
            vmin=0, vmax=1#vmin=sensitivities.min(), vmax=sensitivities.max()
            )        
    idx_inliers = sensitivities <= threshold
    idx_outliers = sensitivities > threshold
    n, m, d = P.shape
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in range(m):
        points = P[:,i,:]
        draw_3d(ax, points, idx_inliers, feature_offset, sensitivities, 
                colors=colors, s=1, alpha=None, label="Inliers")
        if threshold < 1:
            draw_3d(
                ax, points, idx_outliers, feature_offset, sensitivities, 
                colors=colors, s=25, alpha=0.3, label="Outliers")
            ax.legend()
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, 
                               norm=plt.Normalize(vmin=sensitivities.min(), 
                                                  vmax=sensitivities.max()))
    fig.colorbar(sm)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("{}3D model{}{}".format(
        "{}: ".format(data_type) if data_type is not None else "",
        " and sensitivities" if colors is None else "",
        " * {}".format(mul) if (mul != 1 and colors is None) else ""))
    plt.savefig("results/sensitivities_3d_n{}_m{}_k{}_{}_{}.png".format(
        n, m, k, data_type, ctime().replace(':', '-')), dpi=300)


def load_colors_rgb(fname):
    colors = pd.read_csv(
        fname, delimiter="|", usecols=range(4,4+3), 
        header=0, names=["r", "g", "b"]).values / 255.0
    return colors
