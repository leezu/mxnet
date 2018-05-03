/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file sparse_l2_normalization_op.cc
 * \brief CPU Implementation of sparse_l2_normalization op
 */
#include "./sparse_l2_normalization-inl.h"
#include "./tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SparseL2NormalizationParam);

NNVM_REGISTER_OP(sparse_l2_normalization)
MXNET_ADD_SPARSE_OP_ALIAS(l2_normalization)
.describe(R"code(This operators implements the sparse_l2_normalization function)code" ADD_FILELINE)
.set_attr_parser(ParamParser<SparseL2NormalizationParam>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "norm"};
  })
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
                               [](const nnvm::NodeAttrs &attrs) {
                                 return std::vector<uint32_t>{1};
                               })
.set_attr<nnvm::FInferShape>("FInferShape", SparseL2NormalizationOpShape)
.set_attr<nnvm::FInferType>("FInferType", SparseL2NormalizationOpType)
.set_attr<FInferStorageType>("FInferStorageType", SparseL2NormalizationOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", BinaryBroadcastCompute<cpu, op::mshadow_op::div>)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseL2NormalizationOpForwardEx<cpu>)
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_argument("norm", "NDArray-or-Symbol", "Norm ndarray")
.add_arguments(SparseL2NormalizationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
