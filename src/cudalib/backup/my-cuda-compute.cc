#include "cudalib/my-cuda-compute.h"

bool SRECompute::cuda_sre_compute(CudaInitUBM *ubm, std::string voice_file, std::string ivector_path, int valid_frames)
{
	my_time t;
	t.start();
	CudaSRE<PCMGetData> sre;
	sre.setVoiceFileName(voice_file);
	sre.cuda_compute_mfcc();
	if (sre.get_cuda_feature().NumRows() < static_cast<int32>(valid_frames))
		return false;
	sre.gmm_select(ubm->gmm_);
	sre.cuda_compute_posterior(ubm->fgmm_);
	sre.cuda_compute_ivector(ubm->ie_);
	if(!out_vec_to_file(sre.get_ivector(), ivector_path))
		return false;
	t.end();
	KALDI_LOG << "total computation time: " << t.used_time() << "ms";
	return true;
}
