Curl + openssl has issues reading the system certificates on Android.
Pending a better solution we use in-memory certificates from https://curl.se/docs/caextract.html.

Usage: If there are new certificates on https://curl.se/docs/caextract.html, download the latest .pem and .sha256,
update the python script to point to those, and run it to re-create cacert.pem.inc.

Notes:

- Supposedly if openssl uses md5 hashing for the certificates in /system/etc/security/cacerts it should work, but
a patched version of openssl with this change still failed.
- The 'better' solution might be to use boringssl instead of openssl as it handles the certificate format in
/system/etc/security/cacerts, although even that is potentially problematic as there's no versioning of boringssl.
- Alternatively we could allow the certs to be read from the model so the user is in control of the certs, with fallback
to our copy of the curl certs if not provided.
