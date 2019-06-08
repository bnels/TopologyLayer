#ifndef _COHOM_H
#define _COHOM_H

#include <torch/extension.h>
#include <vector>

#include "cocycle.h"
// #include "interval.h"
#include "complex.h"

// cohomology reduction algorithm
// return barcode

// typedef std::map<int,Interval> Barcode;

// forward function for any filtration
std::vector<torch::Tensor> persistence_forward(
    SimplicialComplex &X, size_t MAXDIM);

#endif
