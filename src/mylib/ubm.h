#ifndef SRE_UBM
#define SRE_UBM

#include "base/kaldi-common.h"
#include "mylib/my-ivector-extractor.h"
#include "gmm/full-gmm.h"
#include "hmm/posterior.h"
#include "gmm/mle-full-gmm.h"
#include "util/common-utils.h"
#include "gmm/diag-gmm.h"
#include "gmm/am-diag-gmm.h"
#include "util/kaldi-io.h"

using namespace kaldi;

class SREUBM
{
	public:
		SREUBM() {}
		~SREUBM() {}

		bool ReadUBMFile(std::string final_ubm, std::string final_ie, std::string gmm_ubm);

		DiagGmm gmm_;
		FullGmm fgmm_;
		MyIvectorExtractor ie_;
};


#endif
