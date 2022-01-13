import numpy as np
from collections.abc import Iterable
from matplotlib import pyplot as plt
from median import (
    robust_median, recursive_robust_median,
    closest_lines, closest_points, dist_points,
    closest_point_sets, pack_lines, unpack_lines,
    pack_colored_points, dist_colored_points, dist_lines_min_set_to_set,
    closest_colored_points, unpack_colored_points,
    closest_colored_point_sets, dist_colored_points_min_set_to_set,
    dist_colored_points_min_set_to_point, dist_colored_points_min_p_to_set,
    enumerate_set_of_sets, closest_colored_point_sets_to_points)


def draw_set_of_sets_of_lines(L_set, s=2, marker='o'):
    n, m, d2 = L_set.shape
    colors = get_color_list()
    for i in range(m):
        L = L_set[:, i, :]
        draw_line_set(L, color=colors[i % len(colors)], s=s)
        
''' Drawing functions '''        

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
    '''
    # lines
    for i in range(n):
        P = P_set[i, :, :]
        x = P[:, 0]
        y = P[:, 1]
        plt.plot(x, y, linewidth=s[i], c=linecolor, alpha=linealpha)
    '''
    # points
    for i in range(m):
        P = P_set[:, i, :]
        p, pcolor = unpack_colored_points(P)
        draw_colored_point_set(P, label=label if i==0 else None, s=s*5, 
                               marker=marker)
        
 
def draw_colored_point_sets_all(P, Q, C, title):
    plt.figure()
    draw_colored_point_sets(P, "P", linecolor="blue")
    draw_colored_point_sets(Q, "Q", s=5, linealpha=0.4, linecolor="green")
    draw_colored_point_sets(C, "C", s=2, linealpha=0.4, linecolor="orange")
    plt.title(title)
    plt.legend()    
        
def draw_points(P, Q, C, title):
    plt.figure()
    draw_point_set(P, "P")
    draw_point_set(Q, "Q", s=100)
    draw_point_set(C, "C")
    plt.title(title)
    plt.legend()    
    
def draw_colored_points(P, Q, C, title):   
    plt.figure()
    draw_colored_point_set(P, "P")
    draw_colored_point_set(Q, "Q", s=100)
    draw_colored_point_set(C, "C", s=10)
    plt.legend()    

def draw_lines_from_points(points1, points2, color=None, s=2, alpha=1):
    #points1, points2 = (np.asarray([item[0] for item in P]),
    #                    np.asarray([item[1] for item in P]))
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
        draw_lines_from_points(c - 2 * d, c + 2 * d, color, s, 
                               alpha=alpha(0.1))
        draw_lines_from_points(c - 5 * d, c + 5 * d, color, s, 
                               alpha=alpha(0.05))
        draw_lines_from_points(c - 10 * d, c + 10 * d, color, s,
                               alpha=alpha(0.02))
    #print(lines.shape, color)
    c, d = unpack_lines(lines)
    if not isinstance(s, Iterable):
        s = np.full(len(lines), s)
    #print(Counter(s))
    idx_normal = s <= 2
    idx_emph = s > 2
    draw(c, d, s, idx_normal, None)
    draw(c, d, s, idx_emph, 10)

def draw_lines(L, Q, C, title):
    #print(L.shape, Q.shape, C.shape)
    plt.figure()
    draw_line_set(L, color="blue")
    draw_point_set(Q, color="orange", s=100)
    draw_line_set(C, color="green")
    plt.title(title)

def draw_lines_set_of_sets(L, widths=None):
    if widths is None:
        widths = 2
    colors = get_color_list()
    n, m, d = L.shape
    for i in range(m):
        draw_line_set(L[:, i, :], colors[i], s=widths)
