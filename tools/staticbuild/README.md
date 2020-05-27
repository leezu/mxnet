<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# MXNet Static Build

This folder contains the core script used to build the static library. This README provides information on how to use the scripts in this folder. Please be aware, all of the scripts are designed to be run under the root folder.

## `build.sh`
This script is simplifies the build. It relies on `conan` to ensure all static
dependencies are built correcty and to manage the build of MXNet. You need to
install `patchelf` first, for example via `apt install patchelf` on Ubuntu
systems. Further you need cmake and conan, which you can install via `pip
install cmake conan`.

Here are examples you can run with this script:

```
tools/staticbuild/build.sh cu102
```
This would build the mxnet package based on CUDA 10.2. Currently, we support variants cpu, native, cu92, cu100, cu101 and cu102. All of these variants except native have oneDNN backend enabled. 

```
tools/staticbuild/build.sh cpu
```

This would build the mxnet package based on oneDNN.

## `build_wheel.sh`
This script builds the python package. It also runs a sanity test.
