from __future__ import print_function
from topologylayer.nn import AlphaLayer
import matplotlib.pyplot as plt

import torch
import time
import numpy as np


def sum_finite(d):
    diff = d[:,0] - d[:,1]
    inds = diff < np.inf
    return torch.sum(diff[inds])

# apparently there is some overhead the first time backward is called.
# we'll just get it over with now.
n = 20
y = torch.rand(n, 2, dtype=torch.float).requires_grad_(True)
layer = AlphaLayer(maxdim=0)
dgm, issublevel = layer(y)
p = sum_finite(dgm[0])
p.backward()

algs = ['hom', 'hom2', 'cohom', 'union_find', 'union_find2']

tcs = {}
tfs = {}
tbs = {}
for alg in algs:
    tcs[alg] = []
    tfs[alg] = []
    tbs[alg] = []


ns = [10, 50, 100, 200, 500, 1000, 2000]

for alg in algs:
    for n in ns:
        y = torch.rand(n, 2, dtype=torch.float).requires_grad_(True)

        t0 = time.time()
        layer = AlphaLayer(maxdim=0, alg=alg)
        ta = time.time() - t0
        tcs[alg].append(ta)

        t0 = time.time()
        dgm, issublevel = layer(y)
        ta = time.time() - t0
        tfs[alg].append(ta)

        p = sum_finite(dgm[0])
        t0 = time.time()
        p.backward()
        ta = time.time() - t0
        tbs[alg].append(ta)

for alg in algs:
    plt.loglog(ns, tfs[alg], label=alg)
plt.legend()
plt.xlabel("n")
plt.ylabel("forward time")
plt.savefig("alg_time_forward_alpha0.png")
