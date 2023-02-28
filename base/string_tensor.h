// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include "ocos.h"


// Retrieves a vector of strings if the input type is std::string.
// It is a copy of the input data and can be modified to compute the output.
void GetTensorMutableDataString(const OrtApi& api, OrtW::CustomOpApi& ort, OrtKernelContext* context,
                                const OrtValue* value, std::vector<std::string>& output);
void FillTensorDataString(const OrtApi& api, OrtW::CustomOpApi& ort, OrtKernelContext* context,
                          const std::vector<std::string>& value, OrtValue* output);
