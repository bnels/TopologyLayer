
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <limits>

#include "cohom.h"
#include "cocycle.h"

void reduction_step(SimplicialComplex &X,\
		 const size_t i,\
     std::vector<Cocycle> &Z,\
     std::vector<torch::Tensor> &diagram,\
		 std::vector<int> &nbars,
	 	 const size_t MAXDIM) {

	 // get cocycle
	 Cocycle c = X.bdr[i];

   // perform single reduction step with cocycle x
   bool flag = false;
   auto pivot = Z.rbegin();
   for(auto x  = Z.rbegin(); x != Z.rend();  ++x){
     // see if inner product is non-zero
     if(x->dot(c) > 0){
       if(flag==false){
         // save as column that will be used for schur complement
         pivot = x;
         flag=true;
       } else {
         // schur complement
         x->add(*pivot);
       }
     }
   }

   // cocycle was closed
   if (flag) {
     // add persistence pair to diagram.

		 // get birth and death indices
		 size_t bindx = pivot->index;
		 size_t dindx = c.index;
		 // get birth dimension
		 size_t hdim = X.dim(bindx);
		 //py::print("bindx: ", bindx, " dindx: ", dindx, " hdim: ", hdim);

		 // delete reduced column from active cocycles
		 // stupid translation from reverse to iterator
		 Z.erase(std::next(pivot).base());

		 // check if we want this bar
		 if (hdim > MAXDIM) { return; }

		 // get location in diagram
		 size_t j = nbars[hdim]++;

		 // put births and deaths in diagram.
		 (diagram[hdim][j].data<float>())[0] = (float) X.full_function[bindx].first;
		 (diagram[hdim][j].data<float>())[1] = (float) X.full_function[dindx].first;

		 // put birth/death indices of bar in X.backprop_lookup
		 X.backprop_lookup[hdim][j] = {(int) bindx, (int) dindx};
   } else {
     //  cocycle opened
		 size_t bindx = c.index;
		 // add active cocycle
     Z.emplace_back(Cocycle(bindx));
   }
 }


/*
	INPUTS:
		X - simplicial complex
			IMPORTANT: assumes that X has been initialized, and filtration has been extended
		MAXDIM - maximum homology dimension
	OUTPUTS: vector of tensors - t
	 t[k] is float32 tensor with barcode for dimension k
*/
std::vector<torch::Tensor> persistence_forward(SimplicialComplex &X, size_t MAXDIM) {

   // produce sort permutation on X
   X.sortedOrder();

   // empty vector of active cocycles
   std::vector<Cocycle> Z;

	 // initialize reutrn diagram
	 std::vector<torch::Tensor> diagram(MAXDIM+1); // return array
	 for (size_t k = 0; k < MAXDIM+1; k++) {
		 int Nk = X.numPairs(k); // number of bars in dimension k
		 // allocate return tensor
		 diagram[k] = torch::empty({Nk,2},
			 torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).requires_grad(true));
		 // allocate critical indices
		 // TODO: do this in intialization since number of pairs is deterministic
		 X.backprop_lookup[k].resize(Nk);
	 }
	 // keep track of how many pairs we've put in diagram
	 std::vector<int> nbars(MAXDIM+1);
	 for (size_t k = 0; k < MAXDIM+1; k++) {
		 nbars[k] = 0;
	 }

	 // go through reduction algorithm
   for (size_t i : X.filtration_perm ) {
     reduction_step(X, i, Z, diagram, nbars, MAXDIM);
   }

	 // add infinite bars using removing columns in Z
	 // backprop_lookup death index = -1,
	 // death time is std::numeric_limits<float>::infinity()
	 // while (!(Z.empty())){
	 for (auto pivot  = Z.begin(); pivot != Z.end();  ++pivot) {
		 // Cocycle pivot = Z.pop_back();
		 // get birth index
		 size_t bindx = pivot->index;
		 // get birth dimension
		 size_t hdim = X.bdr[bindx].dim();
		 if (hdim > MAXDIM) { continue; }

		 // get location in diagram
		 size_t j = nbars[hdim]++;

		 // put births and deaths in diagram.
		 (diagram[hdim][j].data<float>())[0] = (float) X.full_function[bindx].first;
		 (diagram[hdim][j].data<float>())[1] = (float) std::numeric_limits<float>::infinity();

		 // put birth/death indices of bar in X.backprop_lookup
		 X.backprop_lookup[hdim][j] = {(int) bindx, -1};
	 }


   return diagram;
 }
