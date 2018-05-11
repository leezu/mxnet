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
 * \file quad_function-inl.h
 * \brief Operator implementing sparse_dense_division function.
 * For using as an exmaple in the tutorial of adding operators
 * in MXNet backend.
 */
#ifndef MXNET_OPERATOR_TENSOR_SPARSE_DENSE_DIVISION_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_SPARSE_DENSE_DIVISION_OP_INL_H_

#include "./elemwise_op_common.h"
#include "./mshadow_op.h"
#include "./mxnet_op.h"
#include "./operator_common.h"
#include "./tensor/elemwise_binary_broadcast_op.h"
#include <mxnet/operator_util.h>
#include <vector>

namespace mxnet {
namespace op {

inline bool SparseDenseDivisionOpShape(const nnvm::NodeAttrs &attrs,
                                       std::vector<TShape> *in_attrs,
                                       std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

inline bool SparseDenseDivisionOpType(const nnvm::NodeAttrs &attrs,
                                      std::vector<int> *in_attrs,
                                      std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

inline bool SparseDenseDivisionOpStorageType(const nnvm::NodeAttrs &attrs,
                                             const int dev_mask,
                                             DispatchMode *dispatch_mode,
                                             std::vector<int> *in_attrs,
                                             std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int matrix_stype = in_attrs->at(0);
  const int vector_stype = in_attrs->at(1);
  bool dispatched = false;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    dispatched = storage_type_assign(out_attrs, kDefaultStorage, dispatch_mode,
                                     DispatchMode::kFCompute);
  }
  if (!dispatched && matrix_stype == kRowSparseStorage &&
      (vector_stype == kRowSparseStorage || vector_stype == kDefaultStorage)) {
    dispatched = storage_type_assign(
        out_attrs, static_cast<NDArrayStorageType>(matrix_stype), dispatch_mode,
        DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

template <int req, typename xpu> struct sparse_div_kernel {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i, const index_t row_length, DType *out,
                                  const DType *lhs, const DType *rhs) {
    for (index_t j = 0; j < row_length; j++) {
      KERNEL_ASSIGN(out[i * row_length + j], req,
                    lhs[i * row_length + j] / rhs[i]);
    }
  }
};

template <typename xpu>
void SparseDenseDivisionOpForwardEx(const nnvm::NodeAttrs &attrs,
                                    const OpContext &ctx,
                                    const std::vector<NDArray> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const auto lhs_stype = inputs[0].storage_type();
  const auto rhs_stype = inputs[0].storage_type();
  const auto out_stype = outputs[0].storage_type();
  CHECK_EQ(lhs_stype, kRowSparseStorage);
  CHECK_EQ(out_stype, kRowSparseStorage);

  Stream<xpu> *s = ctx.get_stream<xpu>();

  if (!inputs[0].storage_initialized()) {
    FillZerosRspImpl(s, outputs[0]);
    return;
  }

  if (rhs_stype == kRowSparseStorage) {
    CHECK(inputs[0].storage_shape()[0] == inputs[1].storage_shape()[0])
        << "lhs and rhs must have same number of rows.";
    CHECK(inputs[1].storage_shape()[1] == 1)
        << "rhs array should have shape (X, 1)";
  } else if (rhs_stype == kDefaultStorage) {
    CHECK(inputs[0].storage_shape()[0] == inputs[1].shape()[0])
        << "lhs and rhs must have same number of rows.";
    CHECK(inputs[1].shape()[1] == 1) << "rhs array should have shape (X, 1)";
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }

  CHECK_EQ(inputs[0].data().type_flag_, inputs[1].data().type_flag_);
  CHECK_EQ(inputs[0].data().type_flag_, outputs[0].data().type_flag_);

  MSHADOW_TYPE_SWITCH(inputs[0].data().type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      if (req_type == kWriteTo) {
        const nnvm::dim_t num_indices = inputs[0].storage_shape()[0];
        outputs[0].CheckAndAlloc({mshadow::Shape1(num_indices)});
        MSHADOW_IDX_TYPE_SWITCH(outputs[0].aux_type(rowsparse::kIdx), CType, {
          mshadow::Copy(
              outputs[0].aux_data(rowsparse::kIdx).FlatTo1D<xpu, CType>(),
              inputs[0].aux_data(rowsparse::kIdx).FlatTo1D<xpu, CType>(), s);
        });
      }
      const nnvm::dim_t num_rows = inputs[0].storage_shape()[0];
      const nnvm::dim_t row_length = inputs[0].storage_shape()[1];

      const DType *lhs_value = inputs[0].data().dptr<DType>();
      const DType *rhs_value = inputs[1].data().dptr<DType>();
      DType *out_value = outputs[0].data().dptr<DType>();

      Kernel<sparse_div_kernel<req_type, xpu>, xpu>::Launch(
          s, num_rows, row_length, out_value, lhs_value, rhs_value);
    });
  });
}
} // namespace op
} // namespace mxnet

#endif // MXNET_OPERATOR_TENSOR_SPARSE_DENSE_DIVISION_OP_INL_H_
