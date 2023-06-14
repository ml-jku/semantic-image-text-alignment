import numpy as np
from numpy.linalg import svd

def calc_cosine_sim(src, target, proj=None):
    if proj is not None:
        src = src @ proj
    src = src / np.linalg.norm(src, ord=2, axis=-1, keepdims=True)
    target = target / np.linalg.norm(target, ord=2, axis=-1, keepdims=True)
    return src @ target.T


def orthogonal_procrustes(A, B):
    u, w, vt = svd(B.T.dot(A).T, full_matrices=False)
    R = u.dot(vt)
    return R