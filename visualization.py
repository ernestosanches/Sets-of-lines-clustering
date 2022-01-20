import numpy as np
import pandas as pd
from os import path
from matplotlib import pyplot as plt

def visualize_coreset_points_3d(P, sensitivities):
    offs=0
    MUL = 10
    n, m, d = P.shape
    fig = plt.figure()
    for i in range(m):
        points = P[:,i,:]
        ax = fig.add_subplot(projection="3d")
        p = ax.scatter(points[:,0+offs], points[:,1+offs], points[:,2+offs],
                       c=np.minimum(sensitivities*MUL, 1), s=4)
    cbar = fig.colorbar(p) # TODO: pass an array of p's for all i's
    cbar.ax.set_ylabel("Sensitivity")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D model and sensitivities * {}".format(MUL))
    fig.show()


def visualize_coreset_lines(L, sensitivities, k):
    from matplotlib import cm
    from drawing import draw_line_set
    def draw_set_of_sets_of_lines_colored(L_set, s):
        n, m, d2 = L_set.shape
        for i in range(n):
            L = L_set[i, :, :]
            draw_line_set(L, color=cm.viridis(s[i]))
    n, m, d = L.shape
    plt.figure()
    plt.title("Sensitivities, n = {}, m = {}, k = {}".format(n,m,k))
    draw_set_of_sets_of_lines_colored(L, sensitivities)
    #plt.colorbar()
    plt.xlabel("X")
    plt.ylabel("Y")

def visualize_coreset_points(P, sensitivities, k, do_lines, use_text,
                             apply_log=False):
    if apply_log:
        sensitivities = np.log(sensitivities)
    else:
        MUL = 10
        sensitivities = np.minimum(MUL * sensitivities, 1)
    n, m, d = P.shape
    plt.figure(); 
    plt.axes().set_aspect('equal')
    plt.title("{}, n = {}, m = {}, k = {}, lines = {}, text = {}".format(
        "Log sensitivities" if apply_log 
            else "Sensitivities * {}".format(MUL), 
        n, m, k, do_lines, use_text))
    for i in range(P.shape[1]):
        plt.scatter(P[:,i,0], P[:,i,1], c=sensitivities, s=4)    
    plt.colorbar()
    plt.xlabel("X")
    plt.ylabel("Y")
    
    

# COLORED POINTS
def visualize_points_colors(P, k, do_lines, use_text):
    n, m, d = P.shape
    plt.figure(); 
    plt.axes().set_aspect('equal')
    plt.title("Colored points, n = {}, m = {}, k = {}, lines = {}, text = {}".format(
        n, m, k, do_lines, use_text))
    cmin = P[:,:,-1].min()
    cmax = P[:,:,-1].max()
    for i in range(P.shape[1]):
        plt.scatter(P[:,i,0], P[:,i,1], 
                   c=P[:,i,-1], s=4, vmin=cmin, vmax=cmax, alpha=0.2)    
    plt.colorbar()
    plt.xlabel("X")
    plt.ylabel("Y")


def load_colors_rgb(fname=None):
    if fname is None:
        fpath = "/home/ernesto/projects/tracxpoint/sfm_postprocessing/"
        fname = path.join(fpath, "points3DWithDescriptors_front (4).txt")
    colors = pd.read_csv(
        fname, delimiter="|", usecols=range(4,4+3), 
        header=0, names=["r", "g", "b"]).values / 255.0
    return colors


def visualize_3d_outlier_removal(P, sensitivities, colors, quantile=0.92):
    points = P[:,0,:]
    sss = np.minimum(sensitivities, 1) 
    sss_threshold = np.quantile(sss, quantile)
    idx = sss <= sss_threshold
    print("Total: {}, filtered: {}".format(len(points), sum(idx)))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    p = ax.scatter(points[idx,0], points[idx,1], points[idx,2],
                   c=colors[idx], s=4)
    cbar = fig.colorbar(p)
    cbar.ax.set_ylabel("Sensitivity")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D model and sensitivities. Quantile={}".format(quantile))
    fig.show()
