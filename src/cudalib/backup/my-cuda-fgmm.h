#ifndef MY_CUDA_FGMM
#define MY_CUDA_FGMM
#include <cstdlib>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-packed-matrix.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "matrix/matrix-lib.h"
#include "cudalib/my-cuda-data-struct.h"
#include "hmm/posterior.h"
#include <vector>

namespace kaldi
{


class CudaFGMM
{
	public:
		CudaFGMM() {}
		~CudaFGMM() {}
		void Read(std::istream &is, bool binary);
		void LogLikelihoodsPreselect(const CuVectorBase<BaseFloat> &data,
									 const std::vector<int32> &indices,
									 Vector<BaseFloat> *loglikes) const;
		int32 Dim() const { return means_invcovars_.NumCols(); }
		int32 NumGauss() const { return means_invcovars_.NumRows(); }

	private:
		void ResizeInvCovars(int32 nmix, int32 dim);
		CuVector<BaseFloat> gconsts_;
		CuVector<BaseFloat> weights_;
		std::vector<CuSpMatrix<BaseFloat> > inv_covars_;
		CuMatrix<BaseFloat> means_invcovars_;
};

}
#endif
