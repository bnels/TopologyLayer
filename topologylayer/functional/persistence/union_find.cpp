#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <map>
#include <cstddef>
#include "sparsevec.h"

#include "union_find.h"

// find root node of tree containing 0-cell i
int find_parent(std::vector<int> parent, int i) {
  while (i != parent[i]) {
    i = parent[i];
  }
  return i;
}

/*
  compute 0-dimensional persistence barcode using union-find algorithm
  INPUTS:
		X - simplicial complex
			IMPORTANT: assumes that X has been initialized, and filtration has been extended
	OUTPUT: tensor - t
	 t is a float32 tensor with barcode for dimension 0
*/
torch::Tensor persistence_forward_uf(SimplicialComplex &X) {

   // produce sort permutation on X
   X.sortedOrder();

   // initialize reutrn diagram
	 int N = X.numPairs(0);
	 torch::Tensor diagram = torch::empty({N,2},
		 torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).requires_grad(true)); // return array
	 X.backprop_lookup[0].resize(N);

	 // initialize parent vector
	 std::vector<int> parent(N);
	 std::iota(parent.begin(), parent.end(), 0);

	 int nfinite = 0; // number of finite bars
	 for (size_t k : X.filtration_perm ) {
		 // loop over cells in permutation order
		 if (X.dim(k) == 0) {
			 	// if a 0-cell, set:
				// birth index in X.backprop_lookup
				// birth time in diagram

				// first, get simplex
				int i = X.cells[k][0];
				// set backprop_lookup - default to infinite bar
				X.backprop_lookup[0][i] = {(int) k, -1};
				// set birth time in diagram
				(diagram[i].data<float>())[0] = (float) X.full_function[k].first;
				// set default infinite death in diagram
				(diagram[i].data<float>())[1] = std::numeric_limits<float>::infinity();


		 } else if (X.dim(k) == 1) {
			 // if a 1-cell, set:
			 // death index in X.backprop_lookup
			 // death time in diagram

			 // first, get simplices on boundary
			 int i = X.cells[k][0];
			 int j = X.cells[k][1];

			 // find parents
			 int pi = find_parent(parent, i);
			 int pj = find_parent(parent, j);

			 // if parents are same, nothing to do
			 if (pi == pj) { continue; }
			 // get birth times in filtration order
			 float bi = (diagram[pi].data<float>())[0];
			 float bj = (diagram[pj].data<float>())[0];
			 if (bi < bj) {
				 // component with j merges into component with i
				 // kill bar at pj
				 X.backprop_lookup[0][pj][1] = (int) k;
				 (diagram[pj].data<float>())[1] = (float) X.full_function[k].first;

				 // merge components
				 parent[pj] = pi;
			 } else {
				 // component with i merges into component with j
				 // kill bar at pi
				 X.backprop_lookup[0][pi][1] = (int) k;
				 (diagram[pi].data<float>())[1] = (float) X.full_function[k].first;

				 // merge components
				 parent[pi] = pj;
			 }
			 nfinite++;
			 // if we've found spanning tree, then break
			 if (nfinite == N - 1) { break; }
		 }
		 // else continue
	}

	return diagram;
}
