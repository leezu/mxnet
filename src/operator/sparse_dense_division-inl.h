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

template <typename xpu> struct sparse_div_lhs_rhs_index_map_kernel {
  template <typename IType>
  MSHADOW_XINLINE static void
  Map(int i, const IType lhs_num_indices, const IType rhs_num_indices,
      IType *rhs_idx_lhs_idx_map, const IType *lhs_index,
      const IType *rhs_index) {
    index_t lhs_pointer = 0;
    index_t rhs_pointer = 0;
    while (rhs_pointer < rhs_num_indices && lhs_pointer < lhs_num_indices) {
      if (lhs_index[lhs_pointer] == rhs_index[rhs_pointer]) {
        // inputs[0] and inputs[1] are aligned here
        rhs_idx_lhs_idx_map[rhs_pointer] = lhs_pointer;
        lhs_pointer++;
        rhs_pointer++;
        continue;
      } else if (lhs_index[lhs_pointer] < rhs_index[rhs_pointer]) {
        // inputs[0] has an element missing in inputs[1]
        lhs_pointer++;
        continue;
      } else {
        // TODO
        // std::cout << "inputs[1] has an element missing in inputs[0] "
        //           << "which is unsupported. "
        //           << "In sparse_div_lhs_rhs_index_map_kernel";
        break;
      }
    }

    // Check that we didn't exit prematurely due to running out of indices on
    // lhs
    if (rhs_pointer < rhs_num_indices) {
      // Exited loopprematurely: Ran out of indices on lhs
      // TODO
      // std::cout << "Setting remaining pointers to 0. "
      //           << "This is an invalid operation. "
      //           << "Debug the program!";
      for (; rhs_pointer < rhs_num_indices; rhs_pointer++) {
        rhs_idx_lhs_idx_map[rhs_pointer] = 0;
      }
    }
  }
};

template <int req, typename xpu> struct sparse_div_kernel {
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, const IType row_length, DType *out,
                                  const DType *lhs, const DType *rhs,
                                  const IType *rhs_idx_lhs_idx_map) {
    for (IType j = 0; j < row_length; j++) {
      // TODO(leezu): This leaves empty rows in out if not KWriteTo and rhs row
      // is missing
      KERNEL_ASSIGN(out[rhs_idx_lhs_idx_map[i] * row_length + j], req,
                    lhs[rhs_idx_lhs_idx_map[i] * row_length + j] / rhs[i]);
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
    // CHECK(inputs[0].storage_shape()[0] == inputs[1].storage_shape()[0])
    //     << "lhs and rhs must have same number of rows.";
    CHECK(inputs[1].storage_shape()[1] == 1)
        << "rhs array should have shape (X, 1)";
    // } else if (rhs_stype == kDefaultStorage) {
    //   CHECK(inputs[0].storage_shape()[0] == inputs[1].shape()[0])
    //       << "lhs and rhs must have same number of rows.";
    //   CHECK(inputs[1].shape()[1] == 1) << "rhs array should have shape (X,
    //   1)";
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }

  CHECK_EQ(inputs[0].data().type_flag_, inputs[1].data().type_flag_);
  CHECK_EQ(inputs[0].data().type_flag_, outputs[0].data().type_flag_);

  MSHADOW_IDX_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
    const nnvm::dim_t lhs_num_indices = inputs[0].storage_shape()[0];
    const nnvm::dim_t rhs_num_indices = inputs[1].storage_shape()[0];
    mshadow::Tensor<xpu, 1, IType> rhs_idx_lhs_idx_map =
        ctx.requested[0].get_space_typed<xpu, 1, IType>(
            mshadow::Shape1(rhs_num_indices), s);
    IType *rhs_idx_lhs_idx_map_value = rhs_idx_lhs_idx_map.dptr_;
    const IType *lhs_index = inputs[0].aux_data(rowsparse::kIdx).dptr<IType>();
    const IType *rhs_index = inputs[1].aux_data(rowsparse::kIdx).dptr<IType>();

    Kernel<sparse_div_lhs_rhs_index_map_kernel<xpu>, xpu>::Launch(
        s, 1, lhs_num_indices, rhs_num_indices, rhs_idx_lhs_idx_map_value,
        lhs_index, rhs_index);

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
        const IType num_rows = inputs[1].storage_shape()[0];
        const IType row_length = inputs[0].storage_shape()[1];

        const DType *lhs_value = inputs[0].data().dptr<DType>();
        const DType *rhs_value = inputs[1].data().dptr<DType>();
        DType *out_value = outputs[0].data().dptr<DType>();

        Kernel<sparse_div_kernel<req_type, xpu>, xpu>::Launch(
            s, num_rows, row_length, out_value, lhs_value, rhs_value,
            rhs_idx_lhs_idx_map_value);
      });
    });
  });
} // namespace op
} // namespace op
} // namespace mxnet

#endif // MXNET_OPERATOR_TENSOR_SPARSE_DENSE_DIVISION_OP_INL_H_
