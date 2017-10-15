#include "cudalib/my-cuda-fgmm.h"
#include "cudalib/my-cuda-function-kernel.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <string>
#include <queue>
#include <utility>
using std::pair;
#include <vector>
using std::vector;

void CudaFGMM::ResizeInvCovars(int32 nmix, int32 dim)
{
	KALDI_ASSERT(nmix > 0 && dim > 0);
	if (inv_covars_.size() != static_cast<size_t>(nmix))
		inv_covars_.resize(nmix);
	for (int32 i = 0; i < nmix; i++)
	{
		inv_covars_[i].Resize(dim);
		inv_covars_[i].SetUnit();
	}
}

void CudaFGMM::Read(std::istream &is, bool binary)
{
#if HAVE_CUDA == 1
	std::string token;
	ReadToken(is, binary, &token);
	if (token != "<FullGMMBegin>" && token != "<FullGMM>")
	{
		KALDI_ERR << "Expected <FullGMM>, got" << token;
		exit(-1);
	}
	ReadToken(is, binary, &token);
	if (token == "<GCONSTS>")
	{
		gconsts_.Read(is, binary);
		ExpectToken(is, binary, "<WEIGHTS>");
	}
	else
		if (token != "<WEIGHTS>")
		{
			KALDI_ERR << "FullGMM::Read, expected <WEIGHTS> or <GCONSTS>, got"
					  << token;
			exit(-1);
		}
	weights_.Read(is, binary);
	ExpectToken(is, binary, "<MEANS_INVCOVARS>");
	means_invcovars_.Read(is, binary);
	ExpectToken(is, binary, "<INV_COVARS>");
	int32 ncomp = weights_.Dim(), dim = means_invcovars_.NumCols();
	ResizeInvCovars(ncomp, dim);
	for (int32 i = 0; i < ncomp; i++)
		inv_covars_[i].Read(is, binary);
	ReadToken(is, binary, &token);
	if (token != "<FullGMMEnd>" && token != "</FullGMM>")
	{
		KALDI_ERR << "Expected </FullGMM>, got" << token;
		exit(-1);
	}
#else
	KALDI_ERR << "No Cuda";
	exit(-1);
#endif
}

void CudaFGMM::LogLikelihoodsPreselect(const CuVectorBase<BaseFloat> &data,
							 const vector<int32> &indices,
							 Vector<BaseFloat> *loglikes) const
{
	int32 dim = Dim();
	KALDI_ASSERT(dim == data.Dim());
	int32 num_indices = static_cast<int32>(indices.size());
	CuSpMatrix<BaseFloat> data_sq(dim);
	data_sq.AddVec2(1.0, data);
	data_sq.ScaleDiag(0.5);
	CuVector<BaseFloat> loglikes_(num_indices, kUndefined);
	for (int32 i = 0; i < num_indices; i++)
	{
		int32 idx = indices[i];
		loglikes_(i) = gconsts_(idx) + VecVec(means_invcovars_.Row(idx), data) - CudaTraceSpSpLower(data_sq, inv_covars_[idx]);
	}
	loglikes->Resize(num_indices, kUndefined);
	loglikes_.CopyToVec(loglikes);
}

