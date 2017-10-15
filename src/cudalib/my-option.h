#ifndef MY_OPTION
#define MY_OPTION

#include "feat/feature-mfcc.h"
#include "feat/feature-functions.h"
#include "ivector/voice-activity-detection.h"
#include "mylib/conf.h"

using namespace kaldi;

struct my_option
{
	MfccOptions mfcc_opts;
	VadEnergyOptions vad_opts;
	SlidingWindowCmnOptions slid_opts;
	DeltaFeaturesOptions delta_opts;

	my_option(): mfcc_opts(), vad_opts(), slid_opts(), delta_opts()
	{
		vad_opts.vad_energy_threshold = vadEnergyThreshold;
		vad_opts.vad_energy_mean_scale = vadEnergyMeanScale;
		mfcc_opts.frame_opts.samp_freq = sampFreq;
		mfcc_opts.frame_opts.frame_length_ms = frameLength;
		mfcc_opts.mel_opts.high_freq = highFreq;
		mfcc_opts.mel_opts.low_freq = lowFreq;
		mfcc_opts.num_ceps = numCeps;
		slid_opts.normalize_variance = normVars;
		slid_opts.center = cmnCenter;
		slid_opts.cmn_window = cmnWindow;
		delta_opts.order = deltaOrder;
		delta_opts.window = deltaWindow;
	}
};

#endif
