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

"""Conanfile for building MXNet and it's dependencies.

This file is useful to create a build of the libmxnet.so shared library with
statically linked dependencies. Please see the accompanying
tools/staticbuild/build.sh file for initiating the build.

"""

from conans import ConanFile, CMake, tools


class MxnetConan(ConanFile):
    name = "mxnet"
    version = "2.0"
    license = "Apache License 2.0"
    author = "Apache MXNet (incubating)"
    url = "mxnet.apache.org"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "openmp": [True, False],
        "opencv": [True, False],
        "jpegturbo": [True, False],
        "cuda": [True, False],
        "dnnl": [True, False],
        "blas": ["openblas", "apple", "mkl", "atlas"],
        "lapack": [True, False],
        "s3": [True, False],
        "f16c": [True, False],
        "dist_kvstore": [True, False],
    }
    generators = "cmake_find_package"
    default_options = {
        # Default options for MXNet
        "shared": False,  # Only affects BUILD_SHARED_LIBS and thus libdmlc but
                          # not MXNET_BUILD_SHARED_LIBS and thus libmxnet
        "blas": "openblas",
        "lapack": True,
        "openmp": True,
        "opencv": True,
        "jpegturbo": True,
        "cuda": False,
        "dnnl": True,
        "s3": False,
        "f16c": False,  # The conanfile is currently used for building releases for PyPI and
                        # alike. We need to disable f16c to prevent the usage of AVX512
                        # instructions.
        "dist_kvstore": False,
        # Default options for dependencies
        "opencv:eigen": True,
        "opencv:openblas": True,
        "opencv:fPIC": True,
        "opencv:jpeg": True,
        "opencv:jpegturbo": True,
        "opencv:png": True,
        "opencv:protobuf": False,
        "opencv:shared": False,
        "opencv:contrib": False,
        "opencv:tiff": False,
        "opencv:webp": False,
        "opencv:jasper": False,
        "opencv:openexr": False,
        "opencv:nonfree": False,
        "opencv:dc1394": False,
        "opencv:carotene": False,
        "opencv:cuda": False,
        "opencv:freetype": False,
        "opencv:harfbuzz": False,
        "opencv:glog": False,
        "opencv:gflags": False,
        "opencv:gstreamer": False,
        "opencv:ffmpeg": False,
        "opencv:lapack": False,
        "opencv:quirc": False,
        "zeromq:encryption": None,  # No libsodium
    }

    def requirements(self):
        if self.options.opencv:
           self.requires("opencv/4.1.1@conan/stable")
        if self.options.jpegturbo:
            self.requires("libjpeg-turbo/2.0.4", override=True if self.options.opencv else False)
        if self.options.blas == "openblas":
            self.requires("openblas/4a4c50a7cef9fa91f14e508722f502d956ad5192",
                          override=True if self.options.opencv else False)
        if self.options.s3:
            self.requires("OpenSSL/1.1.1@conan/stable")
        if self.options.dist_kvstore:
            self.requires("protobuf/3.11.4")
            self.requires("zeromq/4.3.2")
            self.requires("zlib/1.2.11@conan/stable", override=True)

    def build(self):
        cmake = CMake(self)

        cmake.generator = "Ninja"

        cmake.definitions["USE_MKL_IF_AVAILABLE"] = False

        cmake.definitions["USE_BLAS"] = self.options.blas
        cmake.definitions["USE_LAPACK"] = self.options.lapack
        cmake.definitions["USE_OPENMP"] = self.options.openmp
        cmake.definitions["USE_OPENCV"] = self.options.opencv
        cmake.definitions["USE_LIBJPEG_TURBO"] = self.options.jpegturbo
        cmake.definitions["USE_LAPACK"] = self.options["openblas"].build_lapack
        cmake.definitions["USE_CUDA"] = self.options.cuda
        cmake.definitions["USE_MKLDNN"] = self.options.dnnl
        cmake.definitions["USE_DIST_KVSTORE"] = self.options.dist_kvstore
        cmake.definitions["USE_S3"] = self.options.s3
        cmake.definitions["USE_F16C"] = self.options.f16c

        cmake.configure()
        cmake.build()
