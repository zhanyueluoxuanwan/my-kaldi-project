#include "mylib/cpu-init.h"

bool InitSRE::ReadUBMFile(std::string final_ubm, std::string final_ie, std::string gmm_ubm)
{
	my_time t;
	t.start();
	if (!ubm.ReadUBMFile(final_ubm, final_ie, gmm_ubm))
		return true;
	t.end();
	KALDI_LOG << "Init time: " << t.used_time() << "ms";
	return true;
}
