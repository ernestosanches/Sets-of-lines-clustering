import numpy as np

''' Helper functions '''
def pack_pairs(c, d):
    return np.hstack((c, d))
def unpack_pairs(L, separator_idx_f):
    separator_idx = separator_idx_f(L)
    if L.ndim == 2:
        # list of lines
        return L[:, :separator_idx], L[:, separator_idx:]
    elif L.ndim == 1:
        # signle line
        return L[:separator_idx], L[separator_idx:]

pack_lines = pack_colored_points = pack_pairs
def unpack_lines(L):
    return unpack_pairs(L, lambda x: x.shape[-1] // 2)
def unpack_colored_points(P):
    p, pcolor = unpack_pairs(P, lambda x: -1)
    return p, pcolor
