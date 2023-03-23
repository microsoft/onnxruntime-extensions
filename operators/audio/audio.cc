#include "ocos.h"
#ifdef ENABLE_DR_LIBS
#include "audio_decoder.hpp"
#endif  // ENABLE_DR_LIBS


FxLoadCustomOpFactory LoadCustomOpClasses_Audio = 
    LoadCustomOpClasses<CustomOpClassBegin, 
#ifdef ENABLE_DR_LIBS
                        CustomOpAudioDecoder
#endif
                        >;
