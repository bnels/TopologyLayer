#ifndef _COCYCLE_H
#define _COCYCLE_H

#include <vector>
#include <cstddef>
#include "sparsevec.h"

#include <torch/extension.h>
namespace py = pybind11;


class Cocycle{
	public:
		// birth index
		size_t index;

		// non-zero entries
		// IMPORTANT: this is assumed to always be sorted!
		SparseF2Vec<int> cochain;

		// we should never have this
		Cocycle() : index(-1){}

		// initializations
		Cocycle(size_t x) : index(x) , cochain((int) x) {}
		Cocycle(size_t x, std::vector<int> y) :  index(x) , cochain(y) {}

		// for debug purposes
		inline void insert(int x) {
			cochain.insert(x);
		}

		// add two cocycles over Z_2
		inline void add(const Cocycle &x) {
			cochain.add(x.cochain);
			return;
		}

		// dot product of two cocycles
		inline int dot(const Cocycle &x) const {
			return cochain.dot(x.cochain);
		}

		// dimension - number of nonzero entries -1
		int dim() const {
			return (cochain.nzinds.size()==0) ? 0 : cochain.nzinds.size()-1;
		}

		// debug function
		void print() {
			py::print(index, " : ");
			cochain.print();
		}

};



#endif
