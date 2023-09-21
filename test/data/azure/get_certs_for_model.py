# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# If the user needs/wants to control the certificates used in HTTPS requests to the custom op's endpoint, an 
# in-memory certificate store can be built from certificates included in the model.
#
# The certificates should be added to the first Azure operator in the model in an attribute called 'x509_certificates'.
# The user must determine the correct certificates for their scenario, and add them to the model.
# The PEM file from https://curl.se/docs/caextract.html may be used.
#
# Include this file in the python script that is creating your model with Azure custom operators.
# Set the 'x509_certificates' attribute of the node to the value returned from calling either get_certs_from_file
# with the path to a PEM file, or get_certs_from_url with a URL that returns certificates in PEM format.


import io
import pathlib


def _get_certs_from_input(input_data):
    certs = None

    # strip out everything except the certs as per https://curl.se/libcurl/c/cacertinmem.html example.
    out = io.StringIO()
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

    certs = out.getvalue().encode('utf-8')

    return certs


def get_certs_from_url(url: str):
    """
    Read the contents of a URL that returns a PEM file, and return the certificates as a UTF-8 string
    for inclusion as a node attribute of an Azure custom operator.
    e.g. https://curl.se/ca/cacert.pem
    :param url: URL that returns the PEM file contents
    :return: UTF-8 string containing the certificates
    """
    import urllib.request
    certs = None

    url_content = io.StringIO()
    with urllib.request.urlopen(url) as url_input:
        url_content.write(url_input.read().decode('utf-8'))
        url_content.seek(0)
        certs = _get_certs_from_input(url_content)

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
