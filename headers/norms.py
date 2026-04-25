import numpy as np

def L1(A,B):
    return np.sum(abs(A-B))

def L2(A,B):
    return np.sqrt(np.average((A-B)**2))

def Linf(A,B):
    return np.max(abs(A-B))
