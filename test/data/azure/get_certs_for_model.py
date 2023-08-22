#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Curl + openssl has issues reading the system certificates on Android.
# Pending a better solution we create an in-memory certificate store from certificates included in the model.
#
# The certificates must be added to the first Azure operator in the model in an attribute called 'x509_certificates'.
# The user must determine the correct certificates for their scenario, and add them to the model.
# The PEM file from https://curl.se/docs/caextract.html may be used.
#
# Include this file in the python script that is creating your model with Azure custom operators.
# Get the value to use in the 'x509_certificates' attribute from eith a file (call get_certs_from_file) or 
# a url (call get_certs_from_url)
# 
# See create_openai_whisper_transcriptions.py for example usage.
#
# Notes:
#
# - Supposedly if openssl uses md5 hashing for the certificates in /system/etc/security/cacerts it should work, but
# a patched version of openssl with this change still failed.
# - The 'better' solution might be to use boringssl instead of openssl as it handles the certificate format in
# /system/etc/security/cacerts, although even that is potentially problematic as there's no versioning of boringssl.

import pathlib
import tempfile


def _get_certs_from_input(input_data):
    certs = None

    # strip out everything except the certs as per https://curl.se/libcurl/c/cacertinmem.html example.
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as out:
        in_cert = False
        num_certs = 0
        for line in input_data.readlines():
            if not in_cert:
                in_cert = "-----BEGIN CERTIFICATE-----" in line
                if in_cert:
                    num_certs += 1

            if in_cert:
                out.write(line)
                in_cert = "-----END CERTIFICATE-----" not in line

        assert num_certs > 0
        assert not in_cert  # mismatched begin/end if not false
        print(f"Processed {num_certs} certificates")

        # rewind and return as UTF-8 string
        out.seek(0)
        certs = out.read()

    return certs


def get_certs_from_url(url: str):
    """
    Read the contents of a url that returns a PEM file, and return the certificates as a UTF-8 string
    for inclusion as a node attribute of an Azure custom operator.
    e.g. https://curl.se/ca/cacert.pem
    :param url: URL that returns the PEM file contents
    :return: UTF-8 string containing the certificates
    """
    import urllib.request
    certs = None

    with tempfile.TemporaryFile(mode="w+", encoding='utf-8') as tmpfile, urllib.request.urlopen(url) as url_input:
        data = url_input.read()
        tmpfile.write(data.decode('utf-8'))
        tmpfile.seek(0)

        certs = _get_certs_from_input(tmpfile)

    return certs


def get_certs_from_file(pem_filename: pathlib.Path):
    """
    Read the contents of a PEM file and return the certificates as a UTF-8 string for inclusion as a node attribute
    of an Azure custom operator.
    :param pem_filename: path to the PEM file
    :return: UTF-8 string containing the certificates
    """

    certs = None
    pem_filename = pem_filename.resolve(strict=True)
    with open(pem_filename) as input_data:
        certs = _get_certs_from_input(input_data)

    return certs


# examples for testing
# if __name__ == "__main__":
#    a = get_certs_from_file(pathlib.Path("cacert.pem"))
#    b = get_certs_from_url("https://curl.se/ca/cacert.pem")
#    assert(a == b)
