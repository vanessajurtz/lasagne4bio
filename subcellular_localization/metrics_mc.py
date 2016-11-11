from __future__ import division
from itertools import product
import math
import numpy as np


def gcsq(z):
    k, n = kn(z)
    x = np.sum(z, axis=1)
    y = np.sum(z, axis=0)
    r = np.zeros([k, k])
    for i in range(k):
        for j in range(k):
            r[i,j] = x[i] * y[j] / n
    rd = np.copy(r)
    rd[rd == 0] = 1
    return np.sum((z - r) ** 2 / rd) / (n * (k - 1))

def gorodkin(z):
    k, n = kn(z)
    t2 = sum(np.dot(z[i,:], z[:,j]) for i, j in it(k))
    t3 = sum(np.dot(z[i,:], z.T[:,j]) for i, j in it(k))
    t4 = sum(np.dot(z.T[i,:], z[:,j]) for i, j in it(k))
    return (n * np.trace(z) - t2) / (math.sqrt(n**2 - t3) * math.sqrt(n**2 - t4))

def kappa(z):
    N = np.sum(z)
    obs  = np.sum(z, axis=1)
    pred = np.sum(z, axis=0)
    Ncorrect = np.sum(z.diagonal())
    Pr_a = Ncorrect / N
    Pr_e = np.sum(obs / N * pred / N)
    return (Pr_a - Pr_e) / (1 - Pr_e)

def IC(z):
    N = np.sum(z)
    obs  = np.sum(z, axis=1)
    pred = np.sum(z, axis=0)
    H_obs  = np.sum(-xlogx(obs/N))
    H_pred = np.sum(-xlogx(pred/N))
    H_count = np.sum(-xlogx(z/N))
    Itotal = H_obs + H_pred - H_count
    return Itotal / H_obs

def xlogx(x):
    y = np.copy(x)
    y[x>0] = y[x>0] * np.log(y[x>0])/math.log(2)
    y[x<=0] = 0
    return y

def kn(z):
    return (z.shape[0], np.sum(z))

def it(k):
    return product(range(k), range(k))

