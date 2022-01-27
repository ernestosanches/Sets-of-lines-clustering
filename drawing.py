import numpy as np
from time import ctime
from collections.abc import Iterable
from matplotlib import pyplot as plt
from utils import unpack_colored_points, unpack_lines


def draw_set_of_sets_of_lines(L_set, s=2, marker='o'):
    n, m, d2 = L_set.shape
    colors = get_color_list()
    for i in range(m):
        L = L_set[:, i, :]
        draw_line_set(L, color=colors[i % len(colors)], s=s)
        
def draw_point_set(P, label=None, color=None, s=4, marker="o"):
    plt.scatter(P[:, 0], P[:, 1], label=label, c=color, s=s, zorder=10,
                marker=marker)

def get_color_list():
    return np.array(["blue", "green", "red", "cyan"])

def draw_colored_point_set(P, label=None, s=4, marker="o"):
    colors = get_color_list()
    p, pcolor = unpack_colored_points(P)
    pcolor = np.asarray(pcolor, dtype=int).flatten()
    draw_point_set(p, color=colors[pcolor], label=label, s=s, marker=marker)

def draw_colored_point_sets(P_set, label=None, s=2, linecolor="blue", 
                            linealpha=0.2, marker="o"):
    n, m, d = P_set.shape
    s = np.array(s)
    if s.ndim == 0:
        s = np.repeat(s, n)
    # points
    for i in range(m):
        P = P_set[:, i, :]
        p, pcolor = unpack_colored_points(P)
        draw_colored_point_set(P, label=label if i==0 else None, s=s*5, 
                               marker=marker)
         
def draw_colored_point_sets_all(P, Q, C, title):
    plt.figure()
    draw_colored_point_sets(P, "P", s=1, linecolor="blue")
    draw_colored_point_sets(C, "C", s=10, linealpha=0.4, linecolor="orange")
    draw_colored_point_sets(Q, "Q", s=100, linealpha=0.4, linecolor="green")
    plt.title(title)
    plt.legend()    
        
def draw_points(P, Q, C, title):
    plt.figure()
    draw_point_set(P, "P", s=2)
    draw_point_set(C, "C", s=4)
    draw_point_set(Q, "Q", s=100)
    plt.title(title)
    plt.legend()    
    
def draw_colored_points(P, Q, C, title):   
    plt.figure()
    draw_colored_point_set(P, "P")
    draw_colored_point_set(C, "C", s=20)
    draw_colored_point_set(Q, "Q", s=100)
    plt.title(title)
    plt.legend()    

def draw_lines_from_points(points1, points2, color=None, s=2, alpha=1):
    s = np.asarray(s)
    grid = np.dstack((points1, points2))
    if s.ndim == 0:
        plt.plot(grid[:,0,:].T, grid[:,1,:].T, c=color, linewidth=s,
                 alpha=alpha)
    else:
        for i in range(len(grid)):
            plt.plot(grid[i,0,:].T, grid[i,1,:].T, c=color, linewidth=s[i],
                     alpha=alpha)

def draw_line_set(lines, color, s=2):
    def draw(c, d, s, idx, set_alpha=None):
        def alpha(a):
            return a if set_alpha is None else set_alpha * a
        c, d, s = c[idx], d[idx], s[idx]
        draw_lines_from_points(c - 1 * d, c + 1 * d, color, s, 
                               alpha=alpha(0.2))
        draw_lines_from_points(c - 5 * d, c + 5 * d, color, s, 
                               alpha=alpha(0.1))
        draw_lines_from_points(c - 10 * d, c + 10 * d, color, s,
                               alpha=alpha(0.05))
    c, d = unpack_lines(lines)
    if not isinstance(s, Iterable):
        s = np.full(len(lines), s)
    idx_normal = s <= 2
    idx_emph = s > 2
    draw(c, d, s, idx_normal, None)
    draw(c, d, s, idx_emph, 10)

def draw_lines(L, Q, C, title):
    if Q.ndim == 1:
        Q = np.expand_dims(Q, axis=0)
    plt.figure()
    draw_line_set(L, color="blue")
    draw_point_set(Q, color="orange", s=100)
    draw_line_set(C, color="green", s=4)
    plt.title(title)

def draw_lines_set_of_sets(L, widths=None):
    if widths is None:
        widths = 2
    colors = get_color_list()
    n, m, d = L.shape
    for i in range(m):
        draw_line_set(L[:, i, :], colors[i], s=widths)


#####################

def plot_mu_sigma(x, mus, sigmas, color, label, n_samples):
    plt.plot(x, mus, label=label, marker="s", color=color, linewidth=2)
    sqrt_n = 1 #np.sqrt(n_samples)
    plt.fill_between(x, 
                     mus - sigmas / sqrt_n, 
                     mus + sigmas / sqrt_n, 
                     alpha = 0.1, color=color)

def plot_graphs(epsilons, n, m, k, n_samples, data_type):
    (sizes, epsilon_mus, epsilon_sigmas,
     epsilon_random_mus, epsilon_random_sigmas) = map(
         np.asarray, zip(*epsilons))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.figure()
    plot_mu_sigma(sizes, epsilon_mus, epsilon_sigmas, 
                  colors[0], "Coreset", n_samples)
    plot_mu_sigma(sizes, epsilon_random_mus, epsilon_random_sigmas, 
                  colors[1], "Uniform sampling", n_samples)
    plt.plot(sizes, np.zeros_like(sizes), label="Full data",
             marker="s", color=colors[2], linewidth=2)
    plt.title("Approximation comparison ({}: n = {}, m = {}, k = {})".format(
        data_type, n, m, k))
    plt.ylabel("Error ratio (epsilon)")
    plt.xlabel("Sample size")
    plt.xscale("log")
    #plt.yscale("log")
    plt.ylim(0, 1.1 * max(epsilon_mus.max(), epsilon_random_mus.max()))
    plt.legend()
    plt.show()
    plt.pause(0.001)    
    plt.savefig("results/graph_coreset_n{}_m{}_k{}_{}_{}.png".format(
        n, m, k, data_type, ctime().replace(':', '-')), dpi=300)

# TODO:  REAL DATA LINES ---   USE ONLY discrete feat. with small num of vals.




