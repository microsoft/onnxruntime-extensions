# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

class PyCustomOpDef:
    ...

def enable_custom_op(enabled: bool) -> bool:
    ...

def add_custom_op(opdef: PyCustomOpDef) -> None:
    ...

def hash_64(s: str, num_buckets: int, fast: int) -> int:
    ...

def default_opset_domain() -> str:
    ...
