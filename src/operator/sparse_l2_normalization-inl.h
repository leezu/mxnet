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
 * \brief Operator implementing sparse_l2_normalization function.
 * For using as an exmaple in the tutorial of adding operators
 * in MXNet backend.
 */
#ifndef MXNET_OPERATOR_TENSOR_SPARSE_L2_NORMALIZATION_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_SPARSE_L2_NORMALIZATION_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "./mshadow_op.h"
#include "./mxnet_op.h"
#include "./operator_common.h"
#include "./elemwise_op_common.h"
#include "./tensor/elemwise_binary_broadcast_op.h"

namespace mxnet {
namespace op {

struct SparseL2NormalizationParam : public dmlc::Parameter<SparseL2NormalizationParam> {
  float eps;
  DMLC_DECLARE_PARAMETER(SparseL2NormalizationParam) {
    DMLC_DECLARE_FIELD(eps).set_default(1e-10f)
      .describe("A small constant for numerical stability.");
  }
};

inline bool SparseL2NormalizationOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

inline bool SparseL2NormalizationOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

inline bool SparseL2NormalizationOpStorageType(const nnvm::NodeAttrs &attrs,
                                               const int dev_mask,
                                               DispatchMode *dispatch_mode,
                                               std::vector<int> *in_attrs,
                                               std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int in_stype = in_attrs->at(0);
  int &out_stype = out_attrs->at(0);
  bool dispatched = false;
  // TODO check type of both inputs
  if (!dispatched && in_stype == kDefaultStorage) {
    dispatched = storage_type_assign(&out_stype, kDefaultStorage, dispatch_mode,
                                     DispatchMode::kFCompute);
  }
  if (!dispatched && in_stype == kRowSparseStorage) {
    dispatched = storage_type_assign(&out_stype, kRowSparseStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

template <int req, typename DType> struct mask_too_small_norm_entries_kernel {
  MSHADOW_XINLINE static void Map(int i, DType *norm, const DType eps) {
    KERNEL_ASSIGN(norm[i], req, norm[i] > eps ? norm[i] : static_cast<DType>(1.0f));
  }
};

template <typename xpu>
void SparseL2NormalizationOpForwardEx(const nnvm::NodeAttrs &attrs,
                                      const OpContext &ctx,
                                      const std::vector<NDArray> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<NDArray> &outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const SparseL2NormalizationParam &param =
      nnvm::get<SparseL2NormalizationParam>(attrs.parsed);
  const auto in_stype = inputs[0].storage_type();
  const auto out_stype = outputs[0].storage_type();
  if (in_stype == kRowSparseStorage && out_stype == kRowSparseStorage) {
    Stream<xpu> *s = ctx.get_stream<xpu>();

    if (!inputs[0].storage_initialized()) {
      FillZerosRspImpl(s, outputs[0]);
      return;
    }

    CHECK(inputs[0].storage_shape()[0] == inputs[1].storage_shape()[0])
        << "data and norm must have same number of rows.";
    CHECK(inputs[1].storage_shape()[1] == 1) << "norm array should have shape (X, 1)";

    MSHADOW_TYPE_SWITCH(inputs[1].data().type_flag_, DType, {
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        const nnvm::dim_t num_rows = inputs[1].storage_shape()[0];
        mxnet_op::Kernel<mask_too_small_norm_entries_kernel<req_type, DType>,
                         xpu>::Launch(s, num_rows,
                                      inputs[1].data().dptr<DType>(),
                                      param.eps);

        if (req_type == kWriteTo) {
          const nnvm::dim_t num_indices = inputs[0].storage_shape()[0];
          outputs[0].CheckAndAlloc({mshadow::Shape1(num_indices)});
          MSHADOW_IDX_TYPE_SWITCH(outputs[0].aux_type(rowsparse::kIdx), CType, {
            mshadow::Copy(
                outputs[0].aux_data(rowsparse::kIdx).FlatTo1D<xpu, CType>(),
                inputs[0].aux_data(rowsparse::kIdx).FlatTo1D<xpu, CType>(), s);
            const std::vector<TBlob> tblob_inputs = {inputs[0].data(),
                                                     inputs[1].data()};
            const std::vector<TBlob> tblob_outputs = {outputs[0].data()};
            BinaryBroadcastCompute<xpu, op::mshadow_op::div>(
                attrs, ctx, tblob_inputs, req, tblob_outputs);
          });
        } else {
          const std::vector<TBlob> tblob_inputs = {inputs[0].data(),
                                                   inputs[1].data()};
          const std::vector<TBlob> tblob_outputs = {outputs[0].data()};
          BinaryBroadcastCompute<xpu, op::mshadow_op::div>(
              attrs, ctx, tblob_inputs, req, tblob_outputs);
        }
      });
    });
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}
} // namespace op
} // namespace mxnet

#endif // MXNET_OPERATOR_TENSOR_SPARSE_L2_NORMALIZATION_OP_INL_H_
