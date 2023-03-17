#include "ocos.h"
#include "negpos.hpp"
#ifdef ENABLE_DLIB
#include "dlib/inverse.hpp"
#include "dlib/stft_norm.hpp"
#endif
#include "segment_extraction.hpp"
#include "segment_sum.hpp"


FxLoadCustomOpFactory LoadCustomOpClasses_Math = 
    LoadCustomOpClasses<CustomOpClassBegin, 
                        CustomOpNegPos,
#ifdef ENABLE_DLIB
                        CustomOpInverse,
                        CustomOpStft,
                        CustomOpStftNorm,
#endif
                        CustomOpSegmentExtraction,
                        CustomOpSegmentSum>;
