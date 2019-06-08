#ifndef _SPARSEVEC_H
#define _SPARSEVEC_H

#include <vector>
#include <cstddef>
#include <limits>

#include <torch/extension.h>
namespace py = pybind11;

/*
Sparse vector definition
header-only file.
*/

template <typename T>
class SparseF2Vec{
	public:
		// non-zero entries
		// IMPORTANT: this is assumed to always be sorted!
		std::vector<T> nzinds;

		// initialize with empty nzinds
		SparseF2Vec() {}

		// non-trivial initialization
		SparseF2Vec(std::vector<T> x) : nzinds(x) {}
		SparseF2Vec(T x) {nzinds.push_back(x);}

		// insert index to nzinds
		void insert(T x) {
			nzinds.push_back(x);
			// TODO: sort indices?
		}


		// add two vectors over F2
		// IMPORTANT: assumes that nzinds is sorted
		void add(const SparseF2Vec<T> &x) {
			// quick check to see if there is anything to do
			if (x.nzinds.size() == 0) {return;}
			if (nzinds.size() == 0) {nzinds = x.nzinds; return;}

			// now we know there is something non-trivial to do
			std::vector<T> tmp;
			auto i1 = nzinds.cbegin();
			auto i2 = x.nzinds.cbegin();
			do {
				if (*i1 == *i2) {
					++i1;
					++i2;
				} else if (*i1 < *i2) {
					tmp.push_back(*i1);
					++i1;
				} else {
					tmp.push_back(*i2);
					++i2;
				}
			} while (i1 < nzinds.cend() && i2 < x.nzinds.cend());
			// run through rest of entries and dump in
			// at most one of the loops does anything
			while (i1 < nzinds.cend()) {
				tmp.push_back(*i1);
				++i1;
			}
			while (i2 < x.nzinds.cend()) {
				tmp.push_back(*i2);
				++i2;
			}
			nzinds = tmp;
			return;
		}


		// dot product of two vectors
		// IMPORTANT: this function assumes that nzinds are sorted!
		int dot(const SparseF2Vec<T> &x) const {
			// inner product
			// quick check to see if anything to be done
			if (nzinds.size() == 0 || x.nzinds.size() == 0) return 0;
			// loop over indices to compute size of intersection
			auto i1 = nzinds.cbegin();
			auto i2 = x.nzinds.cbegin();
			size_t intersection = 0;
			do {
				if (*i1 == *i2) {
					++i1;
					++i2;
					++intersection;
				} else if (*i1 < *i2) {
					++i1;
				} else { // v2 < v1
					++i2;
				}
			} while (i1 < nzinds.cend() && i2 < x.nzinds.cend());
			// mod-2
			return intersection % 2;
		}

		// debug function
		void print() {
			py::print(nzinds);
		}

		// return number of non-zeros
		size_t nnz() {
			return nzinds.size();
		}

		// return last nonzero index
		T last() {
			return nzinds.back();
		}

		// return offset element from last
		T from_end(size_t offset) {
			return nzinds[nzinds.size() - 1 - offset];
		}

};



#endif
