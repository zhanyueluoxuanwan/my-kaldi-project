#include "cudalib/my-cuda-gmm.h"

void CudaGMM::Read(std::istream &is, bool binary)
{
#if HAVE_CUDA == 1
	std::string token;
	ReadToken(is, binary, &token);
	if (token != "<DiagGMMBegin>" && token != "<DiagGMM>")
	{
		KALDI_ERR << "Expected <DiagGMM>, got" << token;
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
			KALDI_ERR << "DiagGMM::Read, expected <WEIGHTS> or <GCONSTS>, got"
					  << token;
			exit(-1);
		}
	weights_.Read(is, binary);
	ExpectToken(is, binary, "<MEANS_INVVARS>");
	means_invvars_.Read(is, binary);
	ExpectToken(is, binary, "<INV_VARS>");
	inv_vars_.Read(is, binary);
	ReadToken(is, binary, &token);
	if (token != "<DiagGMMEnd>" && token != "</DiagGMM>")
	{
		KALDI_ERR << "Expected </DiagGMM>, got" << token;
		exit(-1);
	}
#else
	KALDI_ERR << "No Cuda";
	exit(-1);
#endif
}

void CudaGMM::LoglikeLihoods(const CuMatrixBase<BaseFloat> &data,
                    CuMatrix<BaseFloat> *loglikes) const
{
	long long int start, end;
	start = getSystemTime();
	KALDI_ASSERT(data.NumRows() != 0);
	loglikes->Resize(data.NumRows(), gconsts_.Dim(), kUndefined);
	loglikes->CopyRowsFromVec(gconsts_);
	if (data.NumCols() != Dim())
	{
		KALDI_ERR << "dim is not match";
		exit(-1);
	}
	CuMatrix<BaseFloat> data_sq(data);
	data_sq.ApplyPow(2.0);
	loglikes->AddMatMat(1.0, data, kNoTrans, means_invvars_, kTrans, 1.0);
	loglikes->AddMatMat(-0.5, data_sq, kNoTrans, inv_vars_, kTrans, 1.0);
	end = getSystemTime();
	KALDI_LOG << "compute loglikelihoods time: " << end - start << "ms";
}

void CudaGMM::GaussianSelection(const CuMatrixBase<BaseFloat> &data,
					   int32 num_gselect,
					   CuMatrixInt &output) const
{
	my_time t;
	t.start();
	int32 num_frames = data.NumRows(), num_gauss = NumGauss();
	KALDI_ASSERT(num_frames != 0);
	CuMatrix<BaseFloat> loglikes_mat(num_frames, num_gauss, kUndefined);
	this->LoglikeLihoods(data, &loglikes_mat);
	my_time t1;
	t1.start();
	output.Resize(num_frames, num_gselect);
	my_cuda_gmm_select(loglikes_mat, output, num_gselect);
	t1.end();
	KALDI_LOG << "gmm select time: " << t1.used_time() << "ms";
	t.end();
	KALDI_LOG << "Gaussian select time: " << t.used_time() << "ms";
}




