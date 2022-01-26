import numpy as np
import pandas as pd
from functools import partial
from utils import unpack_lines, pack_lines, pack_colored_points
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_california_housing
from sklearn.preprocessing import scale

from text import load_text_data

''' Helper dataset generation functions '''

def generate_points(n, variant=1):
    mu = np.random.normal(0, 5, 2)
    if variant == 1:
        p_offset = [100, 10]
    elif variant == 2:
        p_offset = [10000, 10]
    elif variant == 3:
        p_offset = [-30, 100]
    P = np.random.multivariate_normal(mu + np.array(p_offset), 
                                      np.eye(2) * 2, int(n * 1/100))
    Q = np.random.multivariate_normal([0,0], np.array([[5,0],[0,1]]) * 5, 
                                      (n - len(P)) // 2)
    R = np.random.multivariate_normal([0,0], np.array([[1,0],[0,5]]) * 5, 
                                      n - len(P) - len(Q))
    P = np.vstack([P, Q, R])
    #np.random.shuffle(P)
    return P

def color_the_points(P_points, color=None):
    if color is None:
        P_color = np.abs(np.asarray(P_points.sum(
            axis=1, keepdims=True), dtype=int)) % 3
    else:
        P_color = np.full((len(P_points), 1), color)        
    return pack_colored_points(P_points, P_color)
    
def generate_colored_points(n):
    P = generate_points(n)
    return color_the_points(P)

def color_set_of_sets_points(P):
    result = []
    for p_set in P:
        result.append([])
        for color, p in enumerate(p_set):
            result[-1].append(color_the_points(p, color))
    return result

def stack_point_sets(point_sets):
    m = len(point_sets)
    if m == 0:
        return None
    n, d = point_sets[0].shape
    return np.concatenate([np.expand_dims(ps, axis=1) for ps in point_sets],
                          axis=1)

def generate_points_sets(n):
    P = np.random.multivariate_normal([0,0], np.eye(2) * 5, n)
    Q = np.random.multivariate_normal([5,1], np.eye(2) * 2, n)
    n, d = P.shape    
    return stack_point_sets((P, Q)) #list(zip(P, Q))

def generate_set_of_lines(n, offset=np.zeros((1, 2)), variant=2):
    if variant == 1:
        size = n//50
        c = np.repeat(offset, size, axis=0)
        d = np.hstack([- np.ones((size, 1)), 3 * np.random.normal(0, 1, size=(size, 1))])
        d = scale(d)
        print("c,d:", c.shape, d.shape)
        P = pack_lines(c, d)    
    
        size = n - n//50
        c = np.repeat(offset, size, axis=0)
        d = np.hstack([np.random.normal(0, 0.5, size=(size, 1)), 3 * np.ones((size, 1))])
        d = scale(d)
        Q = pack_lines(c, d)    
        return np.vstack((P, Q))
    else:
        c = generate_points(n, variant=2) + offset
        d = generate_points(n, variant=3)
        d = scale(d)
        return pack_lines(c, d)
        
''' 3D Point cloud processing functions'''

def get_3d_cloud_path():
    fpath = "/home/ernesto/projects/tracxpoint/sfm_postprocessing/"
    fname = fpath + "points3DWithDescriptors_front (4).txt"
    return fname

def generate_colored_points_sets_3d_cloud(n, m, fname=None):
    assert(m == 1)
    if fname is None:
        fname = get_3d_cloud_path()
    data = pd.read_csv(
        fname, delimiter="|", usecols=range(1,1+3), 
        header=0, names=["x", "y", "z"])#, "r", "g", "b"])  
    P = data.values
    n = len(P)
    P = np.hstack((P, np.zeros((n, 1))))        
    return np.expand_dims(P, axis=1)

def read_colors_3d_cloud(fname=None):
    if fname is None:
        fpath = "/home/ernesto/projects/tracxpoint/sfm_postprocessing/"
        fname = fpath + "points3DWithDescriptors_front (4).txt"
    data = pd.read_csv(
        fname, delimiter="|", usecols=range(4,4+3), 
        header=0, names=["r", "g", "b"])  
    return data.values / 255.0


''' Main dataset generation functions '''

def generate_colored_points_sets_synthetic_flower(n, m, r=1, is_colored=True):
    from SetsClustering.Utils import createFlowerDataset, to_array    
    set_P = createFlowerDataset(r=r, n = n - 90, m=m)
    #set_P_indiced = [(P, idx) for (idx, P) in enumerate(set_P)] 
    P, w = to_array(set_P)
    n, m, d = P.shape
    colors = np.zeros((n, m, 1))
    for i in range(m):
        colors[:, i, 0] = i if is_colored else 0
    P = np.concatenate((P, colors), axis=-1)
    return np.unique(np.around(P, 5), axis=0)
    
def generate_colored_points_sets_reuters(n, m):
    print("Loading text data (m = {})...".format(m))
    if m == 3:
        P = load_text_data(n)
    elif m == 1:
        P = load_text_data(n // 3)
        P = P.reshape((-1, 1, P.shape[-1]))
        P[:, :, -1] = 0
    else:
        assert False, "Incorrect m for text data: {}".format(m)
    print("Data loaded.")
    return P

def generate_colored_points_sets_counterexample(n, m):
    P = np.random.uniform(0, 1, size = (n, m, 3))
    P[:, :, -1] = np.random.randint(0, m, size=(n, 1))
    return P

def generate_colored_points_sets_synthetic_random(n, m, variant=3):
    if variant == 1:
        P1 = np.random.multivariate_normal([0,0], np.eye(2) * 0.2, n)
        P2 = np.random.multivariate_normal([3,2], np.eye(2) * 0.2, n)
        P3 = np.random.multivariate_normal([5,-2], np.eye(2) * 0.2, n)
        P4 = np.random.multivariate_normal([7,0], np.eye(2) * 0.2, n)
    elif variant == 2:
        P1 = np.random.multivariate_normal([0,0], np.array([[0.02,1],[1,0.02]]), n)
        P2 = np.random.multivariate_normal([3,2], np.array([[0.02,-1],[-1,0.02]]), n)
        P3 = np.random.multivariate_normal([5,-2], np.array([[0.02,-1],[-1,0.02]]), n)
        P4 = np.random.multivariate_normal([7,0], np.array([[0.02,1],[1,0.02]]), n)
        colored_points = [color_the_points(P, color) 
                          for color, P in enumerate([P1, P2, P3, P4][:m])]
    else:
        point_groups = [generate_points(n) + np.random.normal(0, 10, (1,2))
                           for i in range(m)]
        colored_points = [color_the_points(P, color) 
                          for color, P in enumerate(point_groups)]
    return stack_point_sets(colored_points)

def generate_colored_points_from_data(n, m, fetch_data_f):
    assert(m==1)
    X, Y = fetch_data_f(return_X_y=True)
    X = X[np.random.choice(len(X), n, replace=False)]
    X = np.asarray(X, dtype=float)
    X = scale(X)
    X = pack_colored_points(X, np.zeros((n, 1)))
    return np.expand_dims(X, axis=1)
    
def generate_set_of_sets_of_lines_reconstruction(
        n, m, fetch_data_f, continuous_features, discrete_features,
        project_2d=False):
    assert(m==1)
    X, Y = fetch_data_f(return_X_y=True)
    X = X[np.random.choice(len(X), n, replace=False)]
    if project_2d:
        X = X[:, 9:11] # TODO: remove; use all dimensions
    X = np.asarray(X, dtype=float)
    X = scale(X)
    
    n, d = X.shape
    
    p_set = np.expand_dims(X, axis=1)
    d_set = np.zeros((n, 1, d))
    
    direction_coordinates = np.random.randint(0, d, size=n)
    d_set[np.arange(n), 0, direction_coordinates] = 1
    L = np.concatenate((p_set, d_set), axis=-1)    
    return L


def generate_set_of_sets_of_lines_reconstruction_old(
        n, m, fetch_data_f, continuous_features, discrete_features,
        project_2d=True):        
    X, Y = fetch_data_f(return_X_y=True)
    X = X[np.random.choice(len(X), n, replace=False)]
    if project_2d:
        X = X[:, 9:11] # TODO: remove; use all dimensions
    X = np.asarray(X, dtype=float)
    X = scale(X)
    
    n, d = X.shape
    
    p_set = np.repeat(np.expand_dims(X, axis=1), m, axis=1)
    d_set = np.zeros((n, m, d))
    
    if d > 2:
        continuous_idxs = np.random.choice(continuous_features, n)
        discrete_idxs = np.random.choice(discrete_features, n)
    else:
        continuous_idxs = np.random.choice([0, 1], n)
        discrete_idxs = 1 - continuous_idxs
        continuous_features = [0, 1]
        discrete_features = [0, 1]
    continuous_values = np.zeros((n, m))
    discrete_values = np.zeros((n, m))
    for i in continuous_features:
        idx = (continuous_idxs == i)
        continuous_values[idx] = np.random.choice(X[:, i], (np.sum(idx),m))
    for i in discrete_features:
        idx = (discrete_idxs == i)
        discrete_values[idx] = np.random.choice(X[:, i], (np.sum(idx),m))
    
    p_set[np.arange(n), :, continuous_idxs] = continuous_values
    p_set[np.arange(n), :, discrete_idxs] = discrete_values
    d_set[np.arange(n), :, continuous_idxs] = 1
    L = np.concatenate((p_set, d_set), axis=-1)
    return L


def generate_set_of_sets_of_lines_synthetic_random(n, m, variant=1):
    L_set = np.concatenate([
        np.expand_dims(generate_set_of_lines(
            n, offset=np.random.normal(0, 100, size=(1, 2)) * np.array([1, 0]),
            variant=2), axis=1)
        for i in range(m)], axis=1)    
    
    if variant == 2:
        a = np.pi/12
        sina = np.sin(a)
        cosa = np.cos(a)
        rotation = np.array([[cosa, -sina],[sina, cosa]])
        L_set[:, 1, 2:4] = np.dot(L_set[:, 0, 2:4], rotation.T)
    return L_set

def generate_set_of_sets_of_lines_synthetic_perpendicular(n, m):
    p_set = generate_colored_points_sets_synthetic_random(n, m)[:, :, :-1]
    d_set = np.zeros_like(p_set)
    d_set[:, :, 0] = np.random.randint(0, 2, size=(n, m))
    d_set[:, :, 1] = 1 - d_set[:, :, 0]
    L_set = np.concatenate((p_set, d_set), axis=-1)
    return L_set

def normalize_lines(L):
    ''' Sets direction vectors to be of unit length '''
    p, d = unpack_lines(L)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return pack_lines(p, d)

############################################################
### Wrapper function to generate a given dataset by name ###
############################################################
from parameters import Datasets

def generate_data_set_of_sets(n, m, data_type):
    real_data = fetch_covtype #fetch_california_housing
    generate_fs = {
        Datasets.POINTS_RANDOM : generate_colored_points_sets_synthetic_random,
        Datasets.POINTS_FLOWER : generate_colored_points_sets_synthetic_flower,
        Datasets.POINTS_REUTERS : generate_colored_points_sets_reuters,
        Datasets.POINTS_CLOUD : generate_colored_points_sets_3d_cloud,
        Datasets.POINTS_COVTYPE : partial(
            generate_colored_points_from_data,
            fetch_data_f=real_data),
        Datasets.LINES_RANDOM : generate_set_of_sets_of_lines_synthetic_random,
        Datasets.LINES_PERPENDICULAR : generate_set_of_sets_of_lines_synthetic_perpendicular,
        Datasets.LINES_COVTYPE : partial(
            generate_set_of_sets_of_lines_reconstruction,
            fetch_data_f=real_data, 
            continuous_features=list(range(0,10)), 
            discrete_features=list(range(10,54))),
        }
        
    '''
        Datasets.LINES_KDDCUP : partial(
            generate_set_of_sets_of_lines_reconstruction,
            fetch_data_f=fetch_kddcup99, 
            continuous_features=[6,7], discrete_features=[8,9]),
        #Datasets.LINES_TRIANGULATION : None,        
    '''
        
    generate_f = generate_fs[data_type]
    data = generate_f(n, m)
    if data_type in Datasets.DATASETS_LINES:
        data = normalize_lines(data)
    return data