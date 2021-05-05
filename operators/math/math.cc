#include "ocos.h"
#include "negpos.hpp"
#include "inverse.hpp"

template const OrtCustomOp** LoadCustomOpClasses<CustomOpNegPos, CustomOpInverse>();

FxLoadCustomOpFactory LoadCustomOpClasses_Math = &LoadCustomOpClasses<CustomOpNegPos, CustomOpInverse>;
