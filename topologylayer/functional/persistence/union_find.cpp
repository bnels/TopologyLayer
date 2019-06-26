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
	 float *dgmdata = diagram.data<float>();
	 for (size_t k : X.filtration_perm ) {
		 // loop over cells in permutation order
		 if (X.dim(k) == 0) {
			 	// if a 0-cell, set:
				// birth index in X.backprop_lookup
				// birth time in diagram

				int i = X.cells[k][0];
				// first, get simplex
				// set backprop_lookup - default to infinite bar
				X.backprop_lookup[0][i] = {(int) k, -1};
				// set birth time in diagram
				*(dgmdata + 2*i) = (float) X.full_function[k].first;
				// set default infinite death in diagram
				*(dgmdata + 2*i + 1) = std::numeric_limits<float>::infinity();


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
			 float bi = *(dgmdata + 2*pi);
			 float bj = *(dgmdata + 2*pj);
			 if (bi < bj) {
				 // component with j merges into component with i
				 // kill bar at pj
				 X.backprop_lookup[0][pj][1] = (int) k;
				 *(dgmdata + 2*pj + 1) = (float) X.full_function[k].first;

				 // merge components
				 parent[pj] = pi;
				 parent[j] = pi;
				 parent[i] = pi;
			 } else {
				 // component with i merges into component with j
				 // kill bar at pi
				 X.backprop_lookup[0][pi][1] = (int) k;
				 *(dgmdata + 2*pi + 1) = (float) X.full_function[k].first;

				 // merge components
				 parent[pi] = pj;
				 parent[i] = pj;
				 parent[j] = pj;
			 }
			 nfinite++;
			 // if we've found spanning tree, then break
			 if (nfinite == N - 1) { break; }
		 }
		 // else continue
	}

	return diagram;
}


// find root node of tree containing 0-cell i
// also find depth of tree
// *pi will be set to parent node
// *depthi with be set to depth of tree
void find_parent_depth(std::vector<int> parent, int i, int *pi, int *depthi) {
	*depthi = 0;
	*pi = i;
  while (*pi != parent[*pi]) {
		(*depthi)++;
    *pi = parent[*pi];
  }
  return;
}

// // merge components
// // try to minimize depth of tree
// // merge component with i in it to component to j in it
// // i.e. at the end everything should have pj as root.
// void merge_components(std::vector<int> parent, int i, int j, int pi, int pj, int depthi, int depthj) {
//
// }

// TODO: second implementation that initializes birth and death arrays
// also need to remember critical cells in X.backprop_lookup

/*
  compute 0-dimensional persistence barcode using union-find algorithm
  INPUTS:
		X - simplicial complex
			IMPORTANT: assumes that X has been initialized, and filtration has been extended
	OUTPUT: tensor - t
	 t is a float32 tensor with barcode for dimension 0
*/
torch::Tensor persistence_forward_uf2(SimplicialComplex &X) {

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
	 std::vector<float> birth(N);
	 // std::fill
	 std::vector<float> death(N);
	 std::fill(death.begin(), death.end(), std::numeric_limits<float>::infinity());

	 int pi, pj, depthi, depthj;

	 int nfinite = 0; // number of finite bars
	 for (size_t k : X.filtration_perm ) {
		 // loop over cells in permutation order
		 if (X.dim(k) == 0) {
			 	// if a 0-cell, set:
				// birth index in X.backprop_lookup
				// birth time in diagram

				int i = X.cells[k][0];
				// first, get simplex
				// set backprop_lookup - default to infinite bar
				X.backprop_lookup[0][i] = {(int) k, -1};
				// set birth time in diagram
				birth[i] = (float) X.full_function[k].first;

		 } else if (X.dim(k) == 1) {
			 // if a 1-cell, set:
			 // death index in X.backprop_lookup
			 // death time in diagram

			 // first, get simplices on boundary
			 int i = X.cells[k][0];
			 int j = X.cells[k][1];

			 // find parents
			 find_parent_depth(parent, i, &pi, &depthi);
			 find_parent_depth(parent, j, &pj, &depthj);

			 // if parents are same, nothing to do
			 if (pi == pj) { continue; }
			 // get birth times in filtration order
			 float bi = birth[pi];
			 float bj = birth[pj];
			 if (bi < bj) {
				 // component with j merges into component with i
				 // kill bar at pj
				 X.backprop_lookup[0][pj][1] = (int) k;
				 death[pj] = (float) X.full_function[k].first;

				 // merge components
				 parent[pj] = pi;
				 parent[j] = pi;
				 parent[i] = pi;
			 } else {
				 // component with i merges into component with j
				 // kill bar at pi
				 X.backprop_lookup[0][pi][1] = (int) k;
				 death[pi] = (float) X.full_function[k].first;

				 // merge components
				 parent[pi] = pj;
				 parent[i] = pj;
				 parent[j] = pj;
			 }
			 nfinite++;
			 // if we've found spanning tree, then break
			 if (nfinite == N - 1) { break; }
		 }
		 // else continue
	}

	// finally dump births and deaths into diagram
	float *dgmdata = diagram.data<float>();
	for (int i = 0; i < N; i++) {
		*(dgmdata + 2*i)     = birth[i];
		*(dgmdata + 2*i + 1) = death[i];
	}

	return diagram;
}


/*
  compute critical edges for 0-dimensional persistence barcode using union-find algorithm
  INPUTS:
		X - simplicial complex
			IMPORTANT: assumes that X has been initialized, and filtration has been extended
	OUTPUT: edges = std::vector<int>
	(i,j) = (edges[2*k], edges[2*k+1]) is kth critical edge
	WARNING:
	Does not return diagram, or set X.backprop_lookup
*/
// returns critical 1-cells in ascending order
std::vector<int> crit_edges_uf(SimplicialComplex &X) {

   // produce sort permutation on X
   X.sortedOrder();

   // initialize edge vector
	 int N = X.numPairs(0);
	 std::vector<int> edges;
	 edges.reserve(2*N - 2); // maximum number of critical edges

	 // initialize parent vector
	 std::vector<int> parent(N);
	 std::iota(parent.begin(), parent.end(), 0);
	 std::vector<float> birth(N); // don't initialize births

	 int nfinite = 0; // number of finite bars
	 for (size_t k : X.filtration_perm ) {
		 // loop over cells in permutation order
		 if (X.dim(k) == 0) {
			 	// if a 0-cell, set:
				// birth index in X.backprop_lookup
				// birth time in diagram

				// first, get simplex
				int i = X.cells[k][0];

				// set birth time in diagram
				birth[i] = (float) X.full_function[k].first;

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
			 // else the edge is critical
			 edges.push_back(i);
			 edges.push_back(j);

			 // get birth times in filtration order
			 float bi = birth[pi];
			 float bj = birth[pj];
			 if (bi < bj) {
				 // component with j merges into component with i

				 // merge components
				 parent[pj] = pi;
				 parent[j] = pi;
				 parent[i] = pi;
			 } else {
				 // component with i merges into component with j

				 // merge components
				 parent[pi] = pj;
				 parent[i] = pj;
				 parent[j] = pj;
			 }
			 nfinite++;
			 // if we've found spanning tree, then break
			 if (nfinite == N - 1) { break; }
		 }
		 // else continue
	}

	return edges;
}
