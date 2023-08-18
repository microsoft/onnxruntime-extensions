#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib

# save latest from https://curl.se/docs/caextract.html.
certs_file = "cacert-2023-05-30.pem"
sha_file = certs_file + ".sha256"

with open(certs_file, "rb") as f:
    digest = hashlib.file_digest(f, "sha256")

with open(sha_file, "r") as f:
    expected_hash = f.readline()
    if not expected_hash.startswith(digest.hexdigest()):
        raise ValueError(f"Hash of {certs_file} does not match expected value from {sha_file}.")

# strip out everything except the certs as per https://curl.se/libcurl/c/cacertinmem.html example.
# we create a .inc file that defines a char array with the contents to #include in c++ code
with (open("cacert.pem.inc", "w") as out):
    out.write("static const char curl_pem[] = \n")
    in_cert = False
    num_certs = 0
    with (open(certs_file, "r") as pem_input):
        for line in pem_input.readlines():
            if not in_cert:
                in_cert = "-----BEGIN CERTIFICATE-----" in line
                if in_cert:
                    num_certs += 1

            if in_cert:
                # write line with quoted text + \n
                # indent each line by 2
                out.write(f'  "{line.strip()}\\n"\n')
                in_cert = "-----END CERTIFICATE-----" not in line

    out.write(";")

    assert(num_certs > 0)
    assert(not in_cert)  # mismatched begin/end if not false
    print(f"Processed {num_certs} certificates")

