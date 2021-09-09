#include "ocos.h"
#include "negpos.hpp"
#ifdef ENABLE_INVERSE
#include "inverse/inverse.hpp"
#endif
#include "segment_extraction.hpp"
#include "segment_sum.hpp"


FxLoadCustomOpFactory LoadCustomOpClasses_Math = 
    LoadCustomOpClasses<CustomOpClassBegin, 
                        CustomOpNegPos,
#ifdef ENABLE_INVERSE
                        CustomOpInverse,
#endif
                        CustomOpSegmentExtraction,
                        CustomOpSegmentSum>;
