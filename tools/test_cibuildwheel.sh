#!/bin/bash

if [[ "$OCOS_ENABLE_AZURE" == "1" ]]
then
    python ./test/test_azure_ops.sh
fi
