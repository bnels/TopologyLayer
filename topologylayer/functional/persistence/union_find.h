#include <torch/extension.h>
#include <vector>

#include "complex.h"

// 0-dimensional persistence using union-find algorithm
torch::Tensor persistence_forward_uf(SimplicialComplex &X);
