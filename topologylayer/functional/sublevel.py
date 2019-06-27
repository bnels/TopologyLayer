from __future__ import print_function

import torch
import numpy as np

from torch.autograd import Variable, Function
from .persistence import SimplicialComplex, persistenceForwardCohom, persistenceBackward, persistenceForwardHom
from .persistence import persistenceForwardUF, persistenceForwardUF2, critEdges, critEdges2

class SubLevelSetDiagram(Function):
    """
    Compute sub-level set persistence on a space
    forward inputs:
        X - simplicial complex
        f - torch.float tensor of function values on vertices of X
        maxdim - maximum homology dimension
        alg - algorithm
            'hom' = homology (default)
            'hom2' = nz suppressing homology variant
            'cohom' = cohomology
    """
    @staticmethod
    def forward(ctx, X, f, maxdim, alg='hom'):
        ctx.retshape = f.shape
        f = f.view(-1)
        X.extendFloat(f)
        if alg == 'hom':
            ret = persistenceForwardHom(X, maxdim, 0)
        elif alg == 'hom2':
            ret = persistenceForwardHom(X, maxdim, 1)
        elif alg == 'cohom':
            ret = persistenceForwardCohom(X, maxdim)
        elif alg == 'union_find':
            assert maxdim == 0
            ret = [persistenceForwardUF(X)]
        elif alg == 'union_find2':
            assert maxdim == 0
            ret = [persistenceForwardUF2(X)]
        ctx.X = X
        return tuple(ret)

    @staticmethod
    def backward(ctx, *grad_dgms):
        # print(grad_dgms)
        X = ctx.X
        retshape = ctx.retshape
        grad_ret = list(grad_dgms)
        grad_f = persistenceBackward(X, grad_ret)
        return None, grad_f.view(retshape), None, None


def CriticalEdges(X, y):
    """
    return critical edges of a levelset complex
    """
    X.extendFloat(y)
    edges = np.array(critEdges2(X))
    return edges.reshape(-1, 2)
