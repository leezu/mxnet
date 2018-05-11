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
 * \file sparse_dense_division_op.cc
 * \brief CPU Implementation of sparse_dense_division op
 */
#include "./sparse_dense_division-inl.h"
#include "./tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(sparse_dense_division)
MXNET_ADD_SPARSE_OP_ALIAS(dense_division)
    .describe(
        R"code(This operators divides the rows in the sparse lhs matrix by the values in the rhs vector)code" ADD_FILELINE)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>(
        "FListInputNames",
        [](const NodeAttrs &attrs) {
          return std::vector<std::string>{"data", "norm"};
        })
    .set_attr<nnvm::FMutateInputs>("FMutateInputs",
                                   [](const nnvm::NodeAttrs &attrs) {
                                     return std::vector<uint32_t>{1};
                                   })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs &attrs) {
                                  return std::vector<ResourceRequest>{
                                      ResourceRequest::kTempSpace};
                                })
    .set_attr<nnvm::FInferShape>("FInferShape", SparseDenseDivisionOpShape)
    .set_attr<nnvm::FInferType>("FInferType", SparseDenseDivisionOpType)
    .set_attr<FInferStorageType>("FInferStorageType",
                                 SparseDenseDivisionOpStorageType)
    .set_attr<FCompute>("FCompute<cpu>",
                        BinaryBroadcastCompute<cpu, op::mshadow_op::div>)
    .set_attr<FComputeEx>("FComputeEx<cpu>",
                          SparseDenseDivisionOpForwardEx<cpu>)
    .add_argument("matrix", "NDArray-or-Symbol", "Input 2D matrix")
    .add_argument("vector", "NDArray-or-Symbol", "Input 1D vector");

} // namespace op
} // namespace mxnet
