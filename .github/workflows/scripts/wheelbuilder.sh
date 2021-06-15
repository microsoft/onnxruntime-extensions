#!/bin/bash

set -e -x -u

# CLI arguments
PY_VERSION=$1
PLAT=$2
GITHUB_EVENT_NAME=$3
BUILD_REQUIREMENTS='numpy>=1.18.5 wheel'

PY_VER="cp${PY_VERSION//./}-cp${PY_VERSION//./}"
if [ ! -d "/opt/python/${PY_VER}" ] 
then
    PY_VER="${PY_VER}m"
fi

export PATH=/opt/python/${PY_VER}/bin:$PATH

# Update pip
pip install --upgrade --no-cache-dir pip

# Check if requirements were passed
if [ ! -z "$BUILD_REQUIREMENTS" ]; then
    pip install --no-cache-dir ${BUILD_REQUIREMENTS} || { echo "Installing requirements failed."; exit 1; }
fi

# Build wheels
if [ "$GITHUB_EVENT_NAME" == "schedule" ]; then
    python setup.py bdist_wheel --nightly_build || { echo "Building wheels failed."; exit 1; }
else
    python setup.py bdist_wheel || { echo "Building wheels failed."; exit 1; }
fi


# Bundle external shared libraries into the wheels
# find -exec does not preserve failed exit codes, so use an output file for failures
failed_wheels=$PWD/failed-wheels
rm -f "$failed_wheels"
find . -type f -iname "*-linux*.whl" -exec sh -c "auditwheel repair '{}' -w \$(dirname '{}') --plat '${PLAT}' || { echo 'Repairing wheels failed.'; auditwheel show '{}' >> "$failed_wheels"; }" \;

if [[ -f "$failed_wheels" ]]; then
    echo "Repairing wheels failed:"
    cat failed-wheels
    exit 1
fi

# Remove useless *-linux*.whl; only keep manylinux*.whl
rm -f dist/*-linux*.whl

echo "Succesfully build wheels:"
find . -type f -iname "*manylinux*.whl"
