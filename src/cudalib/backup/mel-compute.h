#ifndef CUDA_MEL_COMPUTE
#define CUDA_MEL_COMPUTE

#include "cudalib/OptionStruct.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-matrix-lib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>
#include <complex>
#include <vector>
#include <utility>

using namespace kaldi;
using kaldi::int32;
using kaldi::int64;
using std::vector;

struct CudaFrameExtractionOptions;
struct CudaMelBanksOptions;

class CudaMelBanks
{
	public:
		static inline BaseFloat InverseMelScale(BaseFloat mel_freq)
		{
			return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
		}

		static inline BaseFloat MelScale(BaseFloat freq)
		{
			return 1127.0f * logf(1.0f + freq / 700.0f);
		}

		static BaseFloat VtlnWarpFreq(BaseFloat vtln_low_cutoff,
									  BaseFloat vtln_high_cutoff,
									  BaseFloat low_freq,
									  BaseFloat high_freq,
									  BaseFloat vtln_warp_factor,
									  BaseFloat freq);

		static BaseFloat VtlnWarpMelFreq(BaseFloat vtln_low_cutoff,
									  BaseFloat vtln_high_cutoff,
									  BaseFloat low_freq,
									  BaseFloat high_freq,
									  BaseFloat vtln_warp_factor,
									  BaseFloat freq);

		void Compute(const CuVectorBase<BaseFloat> &fft_energies,
					 CuVector<BaseFloat> *mel_energies_out) const;

		CudaMelBanks(const CudaMelBanksOptions &opts,
					 const CudaFrameExtractionOptions &frame_opts,
					 BaseFloat vtln_warp_factor);

		int32 NumBins() const { return bins_.size(); }

		const CuVector<BaseFloat> &GetCenterFreqs() const { return center_freqs_;}

	private:
		CuVector<BaseFloat> center_freqs_;
		vector<std::pair<int32, CuVector<BaseFloat> > > bins_;
		bool debug_;
		bool htk_mode_;
		KALDI_DISALLOW_COPY_AND_ASSIGN(CudaMelBanks);

};

void CudaComputeLifterCoeffs(BaseFloat Q, CuVectorBase<BaseFloat> *coeffs);

BaseFloat CudaDurbin(int32 n, const BaseFloat *pAC, BaseFloat *pLP, BaseFloat *pTmp);

void CudaLpc2Cepstrum(int32 n, const BaseFloat *pLPC, BaseFloat *pCepst);

#endif
