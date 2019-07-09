#include <torch/extension.h>
#include <vector>

#include "complex.h"

// 0-dimensional persistence using union-find algorithm
torch::Tensor persistence_forward_uf(SimplicialComplex &X);

// uses vectors then dumps into tensor
torch::Tensor persistence_forward_uf2(SimplicialComplex &X);


// returns critical 1-cells in ascending order
std::vector<int> crit_edges_uf(SimplicialComplex &X);

// returns critical 1-cells for H0 and H1 on graph
std::pair<std::vector<int>, std::vector<int>> graph_crit_edges(SimplicialComplex &X);
