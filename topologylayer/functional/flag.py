from __future__ import print_function

import numpy as np

from torch.autograd import Variable, Function
from .persistence import SimplicialComplex, persistenceForwardCohom, persistenceBackwardFlag, persistenceForwardHom
from .persistence import persistenceForwardUF, persistenceForwardUF2, critEdges, graphCritEdges

class FlagDiagram(Function):
    """
    Compute Flag complex persistence using point coordinates

    forward inputs:
        X - simplicial complex
        y - N x D torch.float tensor of coordinates
        maxdim - maximum homology dimension
        alg - algorithm
            'hom' = homology (default)
            'hom2' = nz suppressing homology variant
            'cohom' = cohomology
    """
    @staticmethod
    def forward(ctx, X, y, maxdim, alg='hom'):
        X.extendFlag(y)
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
        ctx.save_for_backward(y)
        return tuple(ret)

    @staticmethod
    def backward(ctx, *grad_dgms):
        # print(grad_dgms)
        X = ctx.X
        y, = ctx.saved_tensors
        grad_ret = list(grad_dgms)
        grad_y = persistenceBackwardFlag(X, y, grad_ret)
        return None, grad_y, None, None


def CriticalEdges(X, y):
    """
    return critical edges for PH0 of a Flag complex
    """
    X.extendFlag(y)
    edges = np.array(critEdges(X))
    return edges.reshape(-1, 2)


def GraphCriticalEdges(X, y):
    """
    return critical edges for PH0 and PH1 of the 1-skeleton
    """
    X.extendFlag(y)
    e0, e1 = graphCritEdges(X)
    return np.array(e0).reshape(-1, 2), np.array(e1).reshape(-1, 2)
