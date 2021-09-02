#include "ocos.h"
#include "negpos.hpp"
#include "inverse.hpp"
#include "segment_extraction.hpp"
#include "segment_sum.hpp"


FxLoadCustomOpFactory LoadCustomOpClasses_Math = 
    LoadCustomOpClasses<CustomOpClassBegin, 
                        CustomOpNegPos,
                        CustomOpInverse,
                        CustomOpSegmentExtraction,
                        CustomOpSegmentSum>;
