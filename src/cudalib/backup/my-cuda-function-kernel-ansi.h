#ifndef MY_CUDA_FUNTION_KERNEL_ANSI
#define MY_CUDA_FUNTION_KERNEL_ANSI
#include "cudamatrix/cu-matrixdim.h"

#if HAVE_CUDA == 1

extern "C"
{
	void _F_my_cuda_compute_fft(float *data, int dim);
	void _D_my_cuda_compute_fft(double *data, int dim);
	void _F_my_cuda_gmm_select(int32_cuda Gr, int32_cuda Bl, float *data, MatrixDim d, int32_cuda num_gselect, int32_cuda *gmm_out);
	void _D_my_cuda_gmm_select(int32_cuda Gr, int32_cuda Bl, double *data, MatrixDim d, int32_cuda num_gselect, int32_cuda *gmm_out);
}

#endif //HAVE_CUDA

#endif
