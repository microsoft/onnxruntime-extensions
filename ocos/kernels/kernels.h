// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

typedef OrtCustomOp const* CPTR_OrtCustomOp;
typedef CPTR_OrtCustomOp (*FxGetSchemaInstance)();

FxGetSchemaInstance const * GetCustomOpSchemaList();
