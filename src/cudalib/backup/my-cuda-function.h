#ifndef MY_CUDA_FUNCTION
#define MY_CUDA_FUNCTION

#include "util/common-utils.h"
#include "cudalib/my-cuda-function-kernel.h"


using namespace kaldi;
using kaldi::int32;
using kaldi::int64;

template<typename Real>
class CudaFFT {
	public:
		CudaFFT(int32 dim): dim_(dim) {}
		~CudaFFT() {}

		void compute(Real *data);
	private:
		int32 dim_;
};

template<typename Real>
void CudaFFT<Real>::compute(Real *data)
{
	my_cuda_compute_fft(data, dim_);
}

#endif
