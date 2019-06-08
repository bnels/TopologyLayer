#ifndef _BACKWARD_H
#define _BACKWARD_H

#include <torch/extension.h>
#include <vector>

#include "complex.h"

// backward function for lower-star
torch::Tensor persistence_backward(
    SimplicialComplex &X, std::vector<torch::Tensor> &grad_res);

// backward function for flag complexes
torch::Tensor persistence_backward_flag(
     SimplicialComplex &X,
     torch::Tensor &y,
     std::vector<torch::Tensor> &grad_res);

#endif
