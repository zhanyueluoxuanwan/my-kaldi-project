#include "cudalib/my-cuda-sre.h"

template <class DataType>
CuMatrix<BaseFloat> CudaSRE<DataType>::get_cuda_feature()
{
	return cu_feature_;
}

template <class DataType>
Vector<double> CudaSRE<DataType>::get_ivector()
{
	my_time t;
	t.start();
	KALDI_ASSERT(ivector_.Dim() > 0);
	if(ivector_result_.Dim() == 0 )
	{
		ivector_result_.Resize(ivector_.Dim(), kUndefined);
		ivector_.CopyToVec(&ivector_result_);
	}
	t.end();
	KALDI_LOG << "get ivector result time:" << t.used_time() << "ms";
	return ivector_result_;
}

template <class DataType>
bool CudaSRE<DataType>::setVoiceFileName(std::string filename)
{
	if (filename.empty())
		return false;
	else
		this->set_filename(filename);
	return true;
}


template <class DataType>
bool CudaSRE<DataType>::cuda_compute_mfcc()
{
	if (!this->data_read(-1, opts.mfcc_opts.frame_opts.samp_freq))
	{
		KALDI_ERR << "read pcm file error";
		return false;
	}
	if (!this->compute_mfcc(1.0, opts.mfcc_opts))
	{
		KALDI_ERR << "compute mfcc error";
		return false;
	}
	if (!this->compute_vad(opts.vad_opts))
	{
		KALDI_ERR << "compute vad";
		return false;
	}
	if (!this->add_feats(opts.slid_opts, opts.delta_opts))
	{
		KALDI_ERR << "add deltas to feature";
		return false;
	}
	my_time t;
	t.start();
	cu_feature_.Resize(this->get_feature().NumRows(), this->get_feature().NumCols());
	cu_feature_.CopyFromMat(this->get_feature());
	t.end();
	KALDI_LOG << "copy feats to GPU time: " << t.used_time() << "ms";
#if FEATURE_TEST == 1
	Matrix<BaseFloat> out_mat(cu_feature_.NumRows(), cu_feature_.NumCols());
	cu_feature_.CopyToMat(&out_mat);
	std::string filename("feature_result.txt");
	out_to_file(out_mat, filename);
	KALDI_LOG << "write feature result to file succesfully";
#endif
	return true;
}


template <class DataType>
bool CudaSRE<DataType>::gmm_select(CudaGMM &cu_gmm_)
{
	int32 num_gselect = numGselect;
	cu_gmm_.GaussianSelection(cu_feature_, num_gselect, cu_gselect_);
	KALDI_LOG << "cu_gselect_ rows:" << cu_gselect_.NumRows() << "\t cols:" << cu_gselect_.NumCols();
#if GSELECT_TEST == 1
	std::string filename("gmmselect_result.txt");
	out_to_file(cu_gselect_, filename);
	KALDI_LOG << "write gmm select result to file succesfully";
#endif
	return true;
}

template <class DataType>
bool CudaSRE<DataType>::cuda_compute_posterior(FullGmm &fgmm_)
{
	my_time t;
	t.start();
	Matrix<BaseFloat> frames(this->get_feature());
	int32 num_frames = NumFrames();
	Posterior post(num_frames);
	for (int32 t = 0; t < num_frames; t++)
	{
		SubVector<BaseFloat> frame(frames, t);
		std::vector<int32> this_gselect(cu_gselect_.Data() + t * cu_gselect_.NumCols(), cu_gselect_.Data() + t * cu_gselect_.NumCols() + cu_gselect_.NumCols());
		KALDI_ASSERT(this_gselect.size() == cu_gselect_.NumCols());
		Vector<BaseFloat> loglikes;
		fgmm_.LogLikelihoodsPreselect(frame, this_gselect, &loglikes);
		loglikes.ApplySoftMax();
		if (fabs(loglikes.Sum() - 1.0) > 0.01)
			return false;
		else
		{
			if (minPosterior != 0.0)
			{
				int32 max_index = 0;
				loglikes.Max(&max_index);
				for (int32 i = 0; i < loglikes.Dim(); i++)
					if (loglikes(i) < minPosterior)
						loglikes(i) = 0;
				BaseFloat sum = loglikes.Sum();
				if (sum == 0.0)
					loglikes(max_index) = 1.0;
				else
					loglikes.Scale(1.0 / sum);
			}
			for (int32 i = 0; i < loglikes.Dim(); i++)
				if (loglikes(i) != 0.0)
					post[t].push_back(std::make_pair(this_gselect[i], loglikes(i)));
			KALDI_ASSERT(!post[t].empty());
		}
	}
	ScalePosterior(PosteriorScale, &post);
	cu_post_ = post;
	t.end();
	KALDI_LOG << "compute posterior time: " << t.used_time() << "ms";
#if POST_TEST == 1
	PosteriorHolder post_holder_;
	Output out_post("post_result.txt", false, false);
	post_holder_.Write(out_post.Stream(), false, cu_post_);
	KALDI_LOG << "write posterior result to file succesfully";
#endif
	return true;
}


template <class DataType>
bool CudaSRE<DataType>::cuda_compute_ivector(CudaIE &ie)
{
	my_time t;
	t.start();
	ScalePosterior(1.0, &cu_post_);
	cu_stats_.Resize(ie.NumGauss(), ie.FeatDim());
	cu_stats_.AccStats(this->get_feature(), cu_post_);
	t.end();
	KALDI_LOG << "compute stats time: " << t.used_time() << "ms";
	t.start();
	ivector_.Resize(ie.IvectorDim());
	ivector_(0) = ie.PriorOffset();
	ie.GetIvectorDistribution(cu_stats_, &ivector_);
	ivector_(0) = ivector_(0) - ie.PriorOffset();
	cuda_ivector_normalize_length(ivector_);
	cuda_ivector_mean(ivector_);
	cuda_ivector_normalize_length(ivector_);
	t.end();
	KALDI_LOG << "compute ivector distribution time: " << t.used_time() << "ms";
	return true;
}



