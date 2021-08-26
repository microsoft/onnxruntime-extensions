#include "ocos.h"
#include "negpos.hpp"
#include "inverse.hpp"


FxLoadCustomOpFactory LoadCustomOpClasses_Math = LoadCustomOpClasses<CustomOpNegPos, CustomOpInverse>;
