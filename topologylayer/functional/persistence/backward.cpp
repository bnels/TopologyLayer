#include "backward.h"

/*
INPUTS:
		X - simplicial complex
		IMPORTANT: assumes that X has been initialized
	grad_res - vector of vectors of tensors
	same as input format:
	grad_res[k] is float32 tensor of gradient of births/deaths in dimension k
OUTPUT:
	grad_f - gradient w.r.t. original function
*/
torch::Tensor persistence_backward(
 SimplicialComplex &X, std::vector<torch::Tensor> &grad_res) {

	 int N = X.ncells[0]; // number of cells in X
	 torch::Tensor grad_f = torch::zeros({N},
		 torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided));
	// pointer to data
	float *grad_f_data = grad_f.data<float>();

	int NDIMS = grad_res.size();

	// loop over homology dimensions
	for (int k = 0; k < NDIMS; k++) {

		// number of bars in dimension k
		int Nk = grad_res[k].size(0);


		// loop over bars in dim k barcode
		for (int j = 0; j < Nk; j++) {
			// get pointers to filtration indices and pointers
			// int *filtind = grad_res[k][j].data<int>();
			float *grad = grad_res[k][j].data<float>();

			int bi = X.backprop_lookup[k][j][0];
			// check for non-infinite bar
			if (bi != -1) {
				// get birth cell
				// auto ci = X.filtration_perm[bi];
				// find critical vertex
				auto i = X.function_map[bi][0];
				// add gradient to critical vertex.
				grad_f_data[i] += grad[0];
			}

			int di = X.backprop_lookup[k][j][1];
			// check for non-infinite bar
			if (di != -1) {
				// get death cell
				// auto ci = X.filtration_perm[di];
				// find critical vertex
				auto i = X.function_map[di][0];
				// add gradient to critical vertex.
				grad_f_data[i] += grad[1];
			}
		}
	}

	 return grad_f;
}


/*
INPUTS:
	X - simplicial complex
		IMPORTANT: assumes that X has been initialized
	y - coordinate positions
	grad_res - vector of vectors of tensors
		same as input format:
		grad_res[k] is float32 tensor of gradient of births/deaths in dimension k
OUTPUT:
	grad_y - gradient of coordinate positions y
*/
torch::Tensor persistence_backward_flag(
 SimplicialComplex &X,
 torch::Tensor &y,
 std::vector<torch::Tensor> &grad_res) {

	 int N = y.size(0); // number of points
	 int D = y.size(1);
	 torch::Tensor grad_y = torch::zeros({N, D},
		 torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided));
	// pointer to data
	//float *grad_y_data = grad_y.data<float>();

	int NDIMS = grad_res.size();

	// loop over homology dimensions
	for (int k = 0; k < NDIMS; k++) {

		// number of bars in dimension k
		int Nk = grad_res[k].size(0);


		// loop over bars in dim k barcode
		for (int j = 0; j < Nk; j++) {
			// get pointers to filtration indices and pointers
			// int *filtind = grad_res[k][j].data<int>();
			float *grad = grad_res[k][j].data<float>();

			// get index of birth
			int bi = X.backprop_lookup[k][j][0];
			// check for non-infinite bar
			if (bi != -1) {
				// check that birth dim is > 0
				if (X.full_function[bi].second > 0) {
					// get birth cell
					// find critical edge
					auto edge = X.function_map[bi] ;
					// produce unit vector along edge
					torch::Tensor dy = y[edge[0]] - y[edge[1]];
					dy /= torch::norm(dy);
					// add gradient to critical vertex.
					grad_y[edge[0]] += grad[0] * dy;
					grad_y[edge[1]] -= grad[0] * dy;
				}
			}

			int di = X.backprop_lookup[k][j][1];
			// check for non-infinite bar
			if (di != -1) {
				// get death cell
				// auto ci = X.filtration_perm[di];
				// find critical vertex
				auto edge = X.function_map[di];
				// produce unit vector along edge
				torch::Tensor dy = y[edge[0]] - y[edge[1]];
				// TODO: check for zero norm.
				dy /= torch::norm(dy);
				// add gradient to critical vertex.
				grad_y[edge[0]] += grad[1] * dy;
				grad_y[edge[1]] -= grad[1] * dy;
			}
		}
	}

	 return grad_y;
}
