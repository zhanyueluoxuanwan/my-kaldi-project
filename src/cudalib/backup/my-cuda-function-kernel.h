#ifndef MY_CUDA_FUNCTION_KERNEL
#define MY_CUDA_FUNCTION_KERNEL

#if HAVE_CUDA == 1

#include "base/kaldi-error.h"
#include "base/kaldi-common.h"
#include "cudalib/my-cuda-function-kernel-ansi.h"
#include "cudamatrix/cu-common.h"
#include "matrix/matrix-common.h"
#include "matrix/sp-matrix.h"
#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-packed-matrix.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "cudalib/my-cuda-data-struct.h"



inline void my_cuda_compute_fft(float *data, int32_cuda dim) { _F_my_cuda_compute_fft(data, dim); }
inline void my_cuda_compute_fft(double *data, int32_cuda dim) { _D_my_cuda_compute_fft(data, dim); }

inline void _my_cuda_gmm_select(int32_cuda Gr, int32_cuda Bl, double *data, MatrixDim d, int32_cuda num_gselect, int32_cuda *gmm_out)
{
	_D_my_cuda_gmm_select(Gr, Bl, data, d, num_gselect, gmm_out);
}

inline void _my_cuda_gmm_select(int32_cuda Gr, int32_cuda Bl, float *data, MatrixDim d, int32_cuda num_gselect, int32_cuda *gmm_out)
{
	_F_my_cuda_gmm_select(Gr, Bl, data, d, num_gselect, gmm_out);
}

template<typename Real>
inline void my_cuda_gmm_select(CuMatrixBase<Real> &loglikesmat, CuMatrixInt &gmm_out, int32_cuda num_gselect)
{
	int32_cuda dimBlock(CU1DBLOCK);
	int32_cuda dimGrid = (loglikesmat.NumRows() + dimBlock - 1) / dimBlock;
	_my_cuda_gmm_select(dimGrid, dimBlock, loglikesmat.Data(), loglikesmat.Dim(), num_gselect, gmm_out.Data());
}

template<typename Real>
Real CudaTraceSpSpLower(const CuSpMatrix<Real> &A, const CuSpMatrix<Real> &B)
{
	KALDI_ASSERT(A.NumRows() == B.NumRows());
	if (A.NumRows() == 0) return 0.0;
	MatrixIndexT nr = A.NumRows(), size = nr * (nr + 1) / 2;
	CuSubVector<Real> Aall(A.Data(), size);
	CuSubVector<Real> Ball(B.Data(), size);
	return VecVec(Aall, Ball);
}

#endif

#endif
