#ifndef MY_CUDA_SRE
#define MY_CUDA_SRE
#include <cstdlib>
#include "mylib/sre.cc"
#include "cudalib/my-cuda-fgmm.h"
#include "cudalib/my-cuda-gmm.h"
#include "cudalib/my-cuda-ie.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-sp-matrix.h"
#include "cudamatrix/cu-tp-matrix.h"
#include "cudamatrix/cu-packed-matrix.h"
#include "util/common-utils.h"
#include "base/kaldi-common.h"
#include "cudamatrix/cu-matrix-lib.h"
#include "mylib/gettime.h"
#include "cudalib/my-option.h"
#include "cudalib/conf.h"
#include "cudalib/my-cuda-data-struct.h"
#include "cudalib/my-cuda-tool.h"
#include "cudalib/my-cuda-init.h"

namespace kaldi
{


template <class DataType>
class CudaSRE: public SRE<DataType>
{
	public:
		CudaSRE(): SRE<DataType>() {}
		~CudaSRE() {}

		bool setVoiceFileName(std::string filename);
		CuMatrix<BaseFloat> get_cuda_feature();
		int32 NumFrames() const { return cu_feature_.NumRows(); }
		Vector<double> get_ivector();

		bool cuda_compute_mfcc();
		bool gmm_select(CudaGMM &cu_gmm_);
		bool cuda_compute_posterior(FullGmm &fgmm_);
		bool cuda_compute_ivector(CudaIE &ie);

	private:
		my_option opts;
		CuMatrix<BaseFloat> cu_feature_;
		CuMatrixInt cu_gselect_;
		Posterior cu_post_;
		CudaIEStats cu_stats_;
		CuVector<double> ivector_;
		Vector<double> ivector_result_;
};


}
#endif

