import numpy as np

''' Helper functions '''
def pack_pairs(c, d):
    if c.ndim == 1:
        return np.hstack((c,d))
    else:
        return np.concatenate((c, d), axis=-1)
def unpack_pairs(L, separator_idx_f):
    separator_idx = separator_idx_f(L)
    return L[..., :separator_idx], L[..., separator_idx:]

pack_lines = pack_colored_points = pack_pairs

def unpack_lines(L):
    p, d = unpack_pairs(L, lambda x: x.shape[-1] // 2)
    return p, d
def unpack_colored_points(P):
    p, pcolor = unpack_pairs(P, lambda x: -1)
    return p, pcolor
