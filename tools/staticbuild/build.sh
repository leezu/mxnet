#!/usr/bin/env bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if [ $# -lt 1 ]; then
    >&2 echo "Usage: build.sh <VARIANT>"
fi

export CURDIR=$PWD
export VARIANT=$(echo $1 | tr '[:upper:]' '[:lower:]')
export PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')

# Copy LICENSE
mkdir -p licenses
cp tools/staticbuild/LICENSE.binary.dependencies licenses/
cp NOTICE licenses/
cp LICENSE licenses/
cp DISCLAIMER-WIP licenses/


# Build mxnet
options=""
if [[ "$VARIANT" == "cu"* ]]; then
    options+=" -o cuda=True"
elif [[ "$VARIANT" == "native" ]]; then
    options+=" -o cuda=False -o dnnl=False"
elif [[ "$VARIANT" == "cpu" ]]; then
    options+=" -o cuda=False"
else
    echo Invalid variant $VARIANT
    exit 1
fi
if [[ "$PLATFORM" == "darwin" ]]; then
    options+=" -o openmp=False -o blas=apple -o opencv:openblas=False"
fi
git submodule update --init --recursive || true

# Check usage of correct C++ ABI
conan profile new default --detect || true  # Generate conan default profile if it does not exist already
if [[ $(conan profile get settings.compiler default) == 'gcc' &&
          $(conan profile get settings.compiler.version default) -ge 5 &&
          $(conan profile get settings.compiler.libcxx default) == 'libstdc++' ]]; then
    echo "WARNING: You are using GCC>=5 but targeting the old libstdc++ ABI. This is not supported for building MXNet."
    echo "We updated your default profile to target the libstdc++11 ABI."
    echo "See https://docs.conan.io/en/latest/howtos/manage_gcc_abi.html for more information"
    conan profile update settings.compiler.libcxx=libstdc++11 default
fi;

# Deploy an extended settings.yml introducing os.force_build_from_source to
# prevent conan from downloading pre-built artifacts (for which we have no
# guarantee about their glibc requirements) while still enabling conan to re-use
# locally cached artifacts. See https://github.com/conan-io/conan/issues/7117
tmpfile=$(mktemp)
zip -r - conan/settings.yml > $tmpfile  # zip the settings.yml
conan config install $tmpfile

# Register local conan recipes
conan export conan/recipes/openblas

# Build libmxnet.so
rm -rf build; mkdir build; cd build
conan install .. ${options} -s os.force_build_from_source=True --build missing
conan build ..  # build mxnet
cd -

# Move to lib
rm -rf lib; mkdir lib;
if [[ $PLATFORM == 'linux' ]]; then
    cp -L build/libmxnet.so lib/libmxnet.so
    cp -L $(ldd lib/libmxnet.so | grep libgfortran |  awk '{print $3}') lib/
    cp -L $(ldd lib/libmxnet.so | grep libquadmath |  awk '{print $3}') lib/
elif [[ $PLATFORM == 'darwin' ]]; then
    cp -L build/libmxnet.dylib lib/libmxnet.dylib
fi

# Print the linked objects on libmxnet.so
>&2 echo "Checking linked objects on libmxnet.so..."
if [[ ! -z $(command -v readelf) ]]; then
    readelf -d lib/libmxnet.so
    strip --strip-unneeded lib/libmxnet.so
elif [[ ! -z $(command -v otool) ]]; then
    otool -L lib/libmxnet.dylib
    strip -u -r -x lib/libmxnet.dylib
else
    >&2 echo "Not available"
fi
