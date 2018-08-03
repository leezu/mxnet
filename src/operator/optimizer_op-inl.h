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
 *  Copyright (c) 2016 by Contributors
 * \file optimizer_op-inl.h
 * \brief Optimizer operators
 * \author Junyuan Xie
 */
#ifndef MXNET_OPERATOR_OPTIMIZER_OP_INL_H_
#define MXNET_OPERATOR_OPTIMIZER_OP_INL_H_
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mshadow/base.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./elemwise_op_common.h"
#include "mxnet_op.h"
#include "./tensor/init_op.h"
#include "./tensor/util/tensor_util-inl.h"

namespace mxnet {
namespace op {

/*
 * \brief log message for optimizers with lazy update.
 */
inline void LogLazyUpdate() {
  common::LogOnce("Optimizer with lazy_update = True detected. "
                  "Be aware that lazy update with row_sparse gradient is different from "
                  "standard update, and may lead to different empirical results. See "
                  "https://mxnet.incubator.apache.org/api/python/optimization/optimization.html "
                  "for more details.");
}

struct SGDParam : public dmlc::Parameter<SGDParam> {
  float lr;
  float wd;
  float rescale_grad;
  float clip_gradient;
  bool lazy_update;
  DMLC_DECLARE_PARAMETER(SGDParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(lazy_update)
    .set_default(true)
    .describe("If true, lazy updates are applied if gradient's stype is row_sparse.");
  }
};


struct SGDKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* weight_data,
    const DType* grad_data, const DType param_clip_gradient,
    const DType param_lr, const DType param_wd, const DType param_rescale_grad,
    const OpReqType req) {
    if (param_clip_gradient >= 0.0f) {
      KERNEL_ASSIGN(out_data[i], req,
             (1.f-param_lr*param_wd)*weight_data[i]
               - (param_lr)
                 * mshadow_op::clip::Map(param_rescale_grad*grad_data[i], param_clip_gradient));
    } else {
      KERNEL_ASSIGN(out_data[i], req,
             (1.f-param_lr*param_wd)*weight_data[i]
               - (param_lr*param_rescale_grad)*grad_data[i]);
    }
  }
};

template<typename xpu>
inline void SGDUpdate(const nnvm::NodeAttrs& attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  const SGDParam& param = nnvm::get<SGDParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    Kernel<SGDKernel, xpu>::Launch(s, weight.shape_.Size(), out.dptr_, weight.dptr_,
      grad.dptr_, static_cast<DType>(param.clip_gradient),
      static_cast<DType>(param.lr), static_cast<DType>(param.wd),
      static_cast<DType>(param.rescale_grad), req[0]);
  });
}

/*! \brief kernel for sparse sgd
 */
template<int req, typename xpu>
struct SGDDnsRspKernel;

template<int req>
struct SGDDnsRspKernel<req, gpu> {
  // DType is the output data type
  // IType is row sparse idx type
  // i is the ith element in row sparse gradient
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, const index_t row_length, DType* out, const DType* weight,
                                  const IType* grad_idx, const DType *grad_val,
                                  const DType clip_gradient, const DType lr,
                                  const DType wd, const DType rescale_grad) {
    using nnvm::dim_t;
    using namespace mshadow_op;
    const dim_t row_id = i / row_length;
    const dim_t col_id = i % row_length;
    const dim_t row_offset = grad_idx[row_id] * row_length;
    const dim_t data_i = row_offset + col_id;
    if (clip_gradient >= 0.0f) {
      KERNEL_ASSIGN(out[data_i], req, (1.f - lr * wd) * weight[data_i] -
                   (lr) * mshadow_op::clip::Map(rescale_grad * grad_val[i], clip_gradient));
    } else {
      KERNEL_ASSIGN(out[data_i], req, (1.f - lr * wd) * weight[data_i] -
                    (lr * rescale_grad) * grad_val[i]);
    }
  }
};

/*! \brief kernel for sparse sgd
 */
template<int req>
struct SGDDnsRspKernel<req, cpu> {
  // DType is the output data type
  // IType is row sparse idx type
  // i is the ith row in row sparse gradient
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, const index_t row_length, DType* out, const DType* weight,
                                  const IType* grad_idx, const DType *grad_val,
                                  const DType clip_gradient, const DType lr,
                                  const DType wd, const DType rescale_grad) {
    for (index_t j = 0; j < row_length; j++) {
      index_t data_i = grad_idx[i] * row_length + j;
      index_t grad_i = i * row_length + j;
      if (clip_gradient >= 0.0f) {
        KERNEL_ASSIGN(out[data_i], req, (1.f - lr * wd) * weight[data_i] -
                     (lr) * mshadow_op::clip::Map(rescale_grad * grad_val[grad_i], clip_gradient));
      } else {
        KERNEL_ASSIGN(out[data_i], req, (1.f - lr * wd) * weight[data_i] -
                      (lr * rescale_grad) * grad_val[grad_i]);
      }
    }
  }
};

/*
 * \brief SGD update implementation for dense weight and row_sparse grad.
 *        Both standard update and lazy update are supported.
 */
template<typename xpu>
inline void SGDUpdateDnsRspImpl(const SGDParam& param,
                                const OpContext &ctx,
                                const TBlob& weight,
                                const NDArray& grad,
                                const OpReqType& req,
                                TBlob *out) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  using namespace mxnet_op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  CHECK_EQ(grad.storage_type(), kRowSparseStorage);
  // if gradients are zeros, no weights are updated
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteInplace) << "kWriteInplace is expected for sparse sgd_mom_update";
  CHECK_GT(weight.shape_.Size(), 0);

  MSHADOW_REAL_TYPE_SWITCH(weight.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(grad.aux_type(rowsparse::kIdx), IType, {
      MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
        DType* weight_data = weight.dptr<DType>();
        float wd = param.wd;
        // apply standard weight decay if not lazy update
        if (!param.lazy_update) {
          Kernel<op_with_req<mshadow_op::mul, req_type>, xpu>::Launch(s, weight.Size(),
            weight_data, weight_data, static_cast<DType>(1 - param.lr * param.wd));
          wd = 0;
        }
        if (!grad.storage_initialized()) return;
        const IType* grad_idx = grad.aux_data(rowsparse::kIdx).dptr<IType>();
        const DType* grad_val = grad.data().dptr<DType>();
        const nnvm::dim_t num_rows = grad.aux_shape(rowsparse::kIdx)[0];
        const auto row_length = weight.shape_.ProdShape(1, weight.ndim());
        size_t num_threads = num_rows;
        if (std::is_same<xpu, gpu>::value) {
          num_threads = num_rows * row_length;
        }
        Kernel<SGDDnsRspKernel<req_type, xpu>, xpu>::Launch(s, num_threads, row_length,
          out->dptr<DType>(), weight_data, grad_idx, grad_val,
          static_cast<DType>(param.clip_gradient),
          static_cast<DType>(param.lr), static_cast<DType>(wd),
          static_cast<DType>(param.rescale_grad));
      });
    });
  });
}

/*
 * \brief SGD update implementation for row_sparse grad.
 *        Both standard update and lazy update are supported.
 */
template<typename xpu>
inline void SGDUpdateRspImpl(const SGDParam& param,
                             const OpContext& ctx,
                             const NDArray& weight,
                             const NDArray& grad,
                             const OpReqType& req,
                             NDArray *out) {
  CheckAllRowsPresent(weight, "SGDUpdate", "weights");
  // reuse dns rsp implementation when storage_shape == shape
  TBlob out_blob = out->data();
  SGDUpdateDnsRspImpl<xpu>(param, ctx, weight.data(), grad, req, &out_blob);
}

template<typename xpu>
inline void SGDUpdateEx(const nnvm::NodeAttrs& attrs,
                        const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs) {
  const SGDParam& param = nnvm::get<SGDParam>(attrs.parsed);
  const auto w_stype = inputs[0].storage_type();
  const auto g_stype = inputs[1].storage_type();
  const auto o_stype = outputs[0].storage_type();
  if (o_stype == w_stype && g_stype == kRowSparseStorage &&
      (w_stype == kDefaultStorage || w_stype == kRowSparseStorage)) {
    NDArray out = outputs[0];
    // std update and lazy update with rsp grad
    SGDUpdateRspImpl<xpu>(param, ctx, inputs[0], inputs[1], req[0], &out);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

struct SGDMomParam : public dmlc::Parameter<SGDMomParam> {
  float lr;
  float momentum;
  float wd;
  float rescale_grad;
  float clip_gradient;
  bool lazy_update;
  DMLC_DECLARE_PARAMETER(SGDMomParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(momentum)
    .set_default(0.0f)
    .describe("The decay rate of momentum estimates at each epoch.");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(lazy_update)
    .set_default(true)
    .describe("If true, lazy updates are applied if gradient's stype is row_sparse "
              "and both weight and momentum have the same stype");
  }
};


struct SGDMomKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, DType* mom_data, const DType* weight_data,
    const DType* grad_data, const DType param_clip_gradient, const DType param_momentum,
    const DType param_lr, const DType param_wd, const DType param_rescale_grad,
    const OpReqType req) {
    if (param_clip_gradient >= 0.0f) {
      mom_data[i] = param_momentum*mom_data[i]
              - param_lr*param_wd*weight_data[i]
              - param_lr
              *mshadow_op::clip::Map(param_rescale_grad*grad_data[i], param_clip_gradient);
    } else {
      mom_data[i] = param_momentum*mom_data[i]
                - param_lr*param_wd*weight_data[i]
                - param_lr*param_rescale_grad*grad_data[i];
    }
    KERNEL_ASSIGN(out_data[i], req, weight_data[i] + mom_data[i]);
  }
};

template<typename xpu>
inline void SGDMomUpdate(const nnvm::NodeAttrs& attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  SGDMomParam param = nnvm::get<SGDMomParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> mom = inputs[2].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    Kernel<SGDMomKernel, xpu>::Launch(s, weight.shape_.Size(), out.dptr_, mom.dptr_, weight.dptr_,
      grad.dptr_, static_cast<DType>(param.clip_gradient), static_cast<DType>(param.momentum),
      static_cast<DType>(param.lr), static_cast<DType>(param.wd),
      static_cast<DType>(param.rescale_grad), req[0]);
    });
}

template<int n_in, int n_out, int total_in>
inline bool MP_SGD_InferType(const nnvm::NodeAttrs& attrs,
                             std::vector<int> *in_attrs,
                             std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(total_in)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  for (int i = n_in; i < total_in; ++i) {
    TYPE_ASSIGN_CHECK(*in_attrs, i, mshadow::kFloat32);
  }
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string, n_in, n_out>(
      attrs, in_attrs, out_attrs, -1);
}

struct MP_SGDKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* weight_data,
    const DType* grad_data, float* weight32, const float param_clip_gradient,
    const float param_lr, const float param_wd, const float param_rescale_grad,
    const OpReqType req) {
    if (param_clip_gradient >= 0.0f) {
      float w = weight32[i];
      w = (1.f - param_lr*param_wd)*w -
          (param_lr) * mshadow_op::clip::Map(param_rescale_grad*static_cast<float>(grad_data[i]),
                                             param_clip_gradient);
      weight32[i] = w;
      KERNEL_ASSIGN(out_data[i], req, (DType)w);
    } else {
      float w = weight32[i];
      w = (1.f-param_lr*param_wd)*w
               - (param_lr*param_rescale_grad)*static_cast<float>(grad_data[i]);
      weight32[i] = w;
      KERNEL_ASSIGN(out_data[i], req, (DType)w);
    }
  }
};

template<typename xpu>
inline void MP_SGDUpdate(const nnvm::NodeAttrs& attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  const SGDParam& param = nnvm::get<SGDParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, float> weight32 = inputs[2].FlatTo2D<xpu, float>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    Kernel<MP_SGDKernel, xpu>::Launch(s, weight.shape_.Size(), out.dptr_, weight.dptr_,
      grad.dptr_, weight32.dptr_, param.clip_gradient,
      param.lr, param.wd,
      param.rescale_grad, req[0]);
  });
}

struct MP_SGDMomKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, float* mom_data,
    const DType* weight_data, const DType* grad_data, float* weight32,
    const float param_clip_gradient, const float param_momentum, const float param_lr,
    const float param_wd, const float param_rescale_grad, const OpReqType req) {
    float w = weight32[i];
    float mom = mom_data[i];
    if (param_clip_gradient >= 0.0f) {
      mom = param_momentum*mom
              - param_lr*param_wd*w
              - param_lr
              *mshadow_op::clip::Map(param_rescale_grad*static_cast<float>(grad_data[i]),
                                     param_clip_gradient);
    } else {
      mom = param_momentum*mom
                - param_lr*param_wd*w
                - param_lr*param_rescale_grad*static_cast<float>(grad_data[i]);
    }
    mom_data[i] = mom;
    w = w + mom;
    weight32[i] = w;
    KERNEL_ASSIGN(out_data[i], req, w);
  }
};

template<typename xpu>
inline void MP_SGDMomUpdate(const nnvm::NodeAttrs& attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  SGDMomParam param = nnvm::get<SGDMomParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, float> mom = inputs[2].FlatTo2D<xpu, float>(s);
    Tensor<xpu, 2, float> weight32 = inputs[3].FlatTo2D<xpu, float>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    Kernel<MP_SGDMomKernel, xpu>::Launch(s, weight.shape_.Size(), out.dptr_, mom.dptr_,
      weight.dptr_, grad.dptr_, weight32.dptr_, param.clip_gradient, param.momentum,
      param.lr, param.wd, param.rescale_grad, req[0]);
  });
}

template<int req, typename xpu>
struct SGDMomDnsRspDnsKernel;

template<int req>
struct SGDMomDnsRspDnsKernel<req, cpu> {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, index_t row_length, DType* out_data,
    DType* mom_data, const DType* weight_data, const IType* grad_idx,
    const DType* grad_data, const DType clip_gradient, const DType momentum,
    const DType lr, const DType wd, const DType rescale_grad) {
    const DType rate = lr * wd;
    for (index_t j = 0; j < row_length; j++) {
      index_t data_i = grad_idx[i] * row_length + j;
      index_t grad_i = i * row_length + j;
      if (clip_gradient >= 0.0f) {
        mom_data[data_i] = momentum * mom_data[data_i]
                - rate * weight_data[data_i]
                - lr *
                mshadow_op::clip::Map(rescale_grad * grad_data[grad_i],
                                      clip_gradient);
      } else {
        mom_data[data_i] = momentum * mom_data[data_i]
                  - rate * weight_data[data_i]
                  - lr * rescale_grad * grad_data[grad_i];
      }
      KERNEL_ASSIGN(out_data[data_i], req, weight_data[data_i] + mom_data[data_i]);
    }
  }
};

template<int req>
struct SGDMomDnsRspDnsKernel<req, gpu> {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, index_t row_length, DType* out_data,
    DType* mom_data, const DType* weight_data, const IType* grad_idx,
    const DType* grad_data, const DType clip_gradient, const DType momentum,
    const DType lr, const DType wd, const DType rescale_grad) {
    using nnvm::dim_t;
    const DType rate = lr * wd;
    const dim_t row_id = i / row_length;
    const dim_t col_id = i % row_length;
    const dim_t data_i = grad_idx[row_id] * row_length + col_id;
    if (clip_gradient >= 0.0f) {
      mom_data[data_i] = momentum * mom_data[data_i]
              - rate * weight_data[data_i]
              - lr *
              mshadow_op::clip::Map(rescale_grad * grad_data[i],
                                    clip_gradient);
    } else {
      mom_data[data_i] = momentum * mom_data[data_i]
                - rate * weight_data[data_i]
                - lr * rescale_grad * grad_data[i];
    }
    KERNEL_ASSIGN(out_data[data_i], req, weight_data[data_i] + mom_data[data_i]);
  }
};

/*
 * \brief sgd mom lazy update for dense weight, row_sparse grad, dense state.
 */
template<typename xpu>
inline void SGDMomLazyUpdateDnsRspDnsImpl(const SGDMomParam& param,
                                          const OpContext& ctx,
                                          const TBlob& weight,
                                          const NDArray& grad,
                                          const TBlob& mom,
                                          const OpReqType& req,
                                          TBlob *out) {
  using namespace mxnet_op;
  using namespace rowsparse;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  if (!grad.storage_initialized() || req == kNullOp) return;
  CHECK_EQ(req, kWriteInplace) << "kWriteInplace is expected for sparse sgd_mom_update";
  CHECK_GT(weight.shape_.Size(), 0);
  CHECK_GT(mom.shape_.Size(), 0);

  MSHADOW_REAL_TYPE_SWITCH(weight.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(grad.aux_type(kIdx), IType, {
      MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
        DType* weight_data = weight.dptr<DType>();
        IType* grad_idx = grad.aux_data(kIdx).dptr<IType>();
        DType* grad_val = grad.data().dptr<DType>();
        DType* mom_data = mom.dptr<DType>();
        DType* out_data = out->dptr<DType>();
        index_t num_rows = grad.aux_shape(kIdx)[0];
        auto row_length = weight.shape_.ProdShape(1, weight.ndim());
        size_t num_threads = num_rows;
        if (std::is_same<xpu, gpu>::value) {
          num_threads = num_rows * row_length;
        }
        Kernel<SGDMomDnsRspDnsKernel<req_type, xpu>, xpu>::Launch(s, num_threads, row_length,
          out_data, mom_data, weight_data, grad_idx, grad_val,
          static_cast<DType>(param.clip_gradient), static_cast<DType>(param.momentum),
          static_cast<DType>(param.lr), static_cast<DType>(param.wd),
          static_cast<DType>(param.rescale_grad));
      });
    });
  });
}

/*
 * \brief sgd momentum lazy update for row_sparse grad.
 */
template<typename xpu>
inline void SGDMomLazyUpdateRspImpl(const SGDMomParam& param,
                                    const OpContext& ctx,
                                    const NDArray& weight,
                                    const NDArray& grad,
                                    const NDArray& mom,
                                    const OpReqType& req,
                                    NDArray *out) {
  using namespace mxnet_op;
  using namespace rowsparse;
  CheckAllRowsPresent(weight, "SGDMomUpdate", "weights");
  Stream<xpu>* s = ctx.get_stream<xpu>();
  // fill mom with zero values (if it's in rsp storage)
  // in order to reuse the sgd mom dns impl
  if (mom.storage_type() == kRowSparseStorage && !mom.storage_initialized()) {
    NDArray mom_zeros = mom;
    FillDnsZerosRspImpl(s, &mom_zeros);
  }
  TBlob out_blob = out->data();
  // reuse dns rsp implementation when storage_shape == shape
  SGDMomLazyUpdateDnsRspDnsImpl<xpu>(param, ctx, weight.data(), grad,
                                     mom.data(), req, &out_blob);
}

/*!
 * \brief Storge type inference function for optimizers which support both
 *        lazy update and standard update, with states (e.g. 2nd order moment)
 * \param num_states The number of states that could be row_sparse or dense
 */
template<size_t num_states, typename ParamType>
inline bool StdOptStorageType(const nnvm::NodeAttrs& attrs,
                              const int dev_mask,
                              DispatchMode* dispatch_mode,
                              std::vector<int>* in_attrs,
                              std::vector<int>* out_attrs) {
  using namespace common;
  const ParamType& param = nnvm::get<ParamType>(attrs.parsed);
  // weight, grad, state 0, state 1, ... -> weight
  CHECK_EQ(in_attrs->size(), 2 + num_states);
  CHECK_EQ(out_attrs->size(), 1U);
  const int weight_stype = in_attrs->at(0);
  const int grad_stype = in_attrs->at(1);
  const int state_stype = in_attrs->at(2);
  // the storage type of all states should be the same
  for (size_t i = 3; i <  2 + num_states; i++) {
    CHECK_EQ(state_stype, in_attrs->at(i))
      << "Inconsistent storage types detected in state " << i;
  }
  bool dispatched = false;
  if (!dispatched && ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    // dns, ... -> dns
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && grad_stype == kRowSparseStorage &&
      (weight_stype == kRowSparseStorage || weight_stype == kDefaultStorage) &&
      state_stype == weight_stype) {
    // weight and state share stype, grad's stype = rsp
    dispatched = storage_type_assign(out_attrs, static_cast<NDArrayStorageType>(weight_stype),
                                     dispatch_mode, DispatchMode::kFComputeEx);
    // warn users if lazy_update is turned on
    if (dispatched && param.lazy_update) LogLazyUpdate();
  }
  if (!dispatched && grad_stype == kRowSparseStorage &&
      weight_stype == kRowSparseStorage && state_stype == kDefaultStorage) {
    // weight,  grad, state, ...  -> weight
    // rsp,     rsp,  dns,   ...  -> rsp, standard update
    dispatched = storage_type_assign(out_attrs, static_cast<NDArrayStorageType>(weight_stype),
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

/*
 * \brief kernel for standard momentum update for dense weight, sparse grad and dense state.
 */
template<int req, typename xpu>
struct SGDMomStdDnsRspDnsKernel;


/*
 * \brief standard momentum update for dense weight, row_sparse grad and dense states.
 */
template<typename xpu>
void SGDMomStdUpdateDnsRspDnsImpl(const SGDMomParam& param,
                                  const OpContext& ctx,
                                  const TBlob& weight,
                                  const NDArray& grad,
                                  const TBlob& mom,
                                  const OpReqType& req,
                                  TBlob *out);

/*
 * \brief standard momentum update for row_sparse grad.
 *        both row_sparse and dense weight are supported.
 */
template<typename xpu>
inline void SGDMomStdUpdateRspImpl(const SGDMomParam& param,
                                   const OpContext& ctx,
                                   const NDArray& weight,
                                   const NDArray& grad,
                                   const NDArray& mom,
                                   const OpReqType& req,
                                   NDArray *out) {
  using namespace mxnet_op;
  using namespace rowsparse;
  CheckAllRowsPresent(weight, "SGDMomUpdate", "weights");
  Stream<xpu>* s = ctx.get_stream<xpu>();
  // fill mom with zero values (if it's in rsp storage)
  // in order to reuse the sgd mom dns impl
  if (mom.storage_type() == kRowSparseStorage && !mom.storage_initialized()) {
    NDArray mom_zeros = mom;
    FillDnsZerosRspImpl(s, &mom_zeros);
  }
  TBlob out_blob = out->data();
  SGDMomStdUpdateDnsRspDnsImpl<xpu>(param, ctx, weight.data(), grad,
                                    mom.data(), req, &out_blob);
}

template<typename xpu>
inline void SGDMomUpdateEx(const nnvm::NodeAttrs& attrs,
                           const OpContext &ctx,
                           const std::vector<NDArray> &inputs,
                           const std::vector<OpReqType> &req,
                           const std::vector<NDArray> &outputs) {
  using namespace mxnet_op;
  const SGDMomParam& param = nnvm::get<SGDMomParam>(attrs.parsed);
  auto &weight = inputs[0];
  auto &grad = inputs[1];
  auto &mom = inputs[2];
  const auto w_stype = weight.storage_type();
  const auto m_stype = mom.storage_type();
  const auto out_stype = outputs[0].storage_type();
  NDArray out = outputs[0];
  const bool valid_weight = w_stype == kDefaultStorage || w_stype == kRowSparseStorage;
  const bool valid_grad = grad.storage_type() == kRowSparseStorage;
  const bool lazy_update = param.lazy_update;
  CHECK(w_stype == out_stype) << "Inconsistent weight stype and output stype";
  if (valid_weight && valid_grad && m_stype == w_stype) {
    if (lazy_update) {
      // rsp grad && m.stype = w.stype && lazy_update = true -> lazy update
      SGDMomLazyUpdateRspImpl<xpu>(param, ctx, weight, grad, mom, req[0], &out);
    } else {
      // rsp grad && m.stype = w.stype && lazy_update = false -> std update
      SGDMomStdUpdateRspImpl<xpu>(param, ctx, weight, grad, mom, req[0], &out);
    }
  } else if (w_stype == kRowSparseStorage && valid_grad && m_stype == kDefaultStorage) {
    // rsp weight, rsp grad, dns state -> std update
    SGDMomStdUpdateRspImpl<xpu>(param, ctx, weight, grad, mom, req[0], &out);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

struct ProximalSGDParam : public dmlc::Parameter<ProximalSGDParam> {
  float lr;
  float rescale_grad;
  float clip_gradient;
  float clip_group_gradient_norm;
  float l2_regularization_strength;
  float current_update;
  bool lazy_update;
  DMLC_DECLARE_PARAMETER(ProximalSGDParam) {
    DMLC_DECLARE_FIELD(lr).describe("Learning rate");
    DMLC_DECLARE_FIELD(rescale_grad)
        .set_default(1.0f)
        .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
        .set_default(-1.0f)
        .describe(
            "Clip gradient to the range of [-clip_gradient, clip_gradient] "
            "If clip_gradient <= 0, gradient clipping is turned off. "
            "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(clip_group_gradient_norm)
        .set_default(-1.0f)
        .describe(
            "Rescale gradient group-wise so that the L2 norm of each group "
            "is smaller than clip_group_gradient_norm."
            "If clip_group_gradient_norm <= 0, rescaling is turned off. ");
    DMLC_DECLARE_FIELD(l2_regularization_strength)
        .set_default(0.0f)
        .describe("Lambda term for group lasso objective.");
    DMLC_DECLARE_FIELD(current_update)
        .set_default(0.0f)
        .describe("Current update iteration for lazy update with group lasso "
                  "objective.");
    DMLC_DECLARE_FIELD(lazy_update)
        .set_default(true)
        .describe("If true, lazy updates are applied if gradient's stype is "
                  "row_sparse.");
  }
};

/*! \brief kernel for enforcing group sparsity after sparse sgd update
 */
template <typename xpu> struct ProximalSGDDnsRspKernel {
  // DType is the output data type
  // IType is row sparse idx type
  // i is the ith row in row sparse gradient
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void
  Map(int i, const index_t row_length, DType *out, const DType *weight,
      const IType *grad_idx, const DType *grad_val, DType *last_update_buffer,
      const DType current_update, const DType clip_gradient,
      const DType clip_group_gradient_norm, const DType lr,
      const DType rescale_grad, const DType l2_regularization_strength) {

    DType group_rescale = 1;
    // Check if gradient needs to be rescaled
    if (clip_group_gradient_norm >= 0.0f) {
      DType sum, residual;
      mshadow::red::sum::SetInitValue(sum, residual);
      if (clip_gradient >= 0.0f) {
        for (index_t j = 0; j < row_length; j++) {
          index_t grad_i = i * row_length + j;
          DType grad_ = mshadow_op::clip::Map(rescale_grad * grad_val[grad_i],
                                              clip_gradient);
          mshadow::red::sum::Reduce(sum, grad_ * grad_, residual);
        }
      } else {
        for (index_t j = 0; j < row_length; j++) {
          index_t grad_i = i * row_length + j;
          DType grad_ = rescale_grad * grad_val[grad_i];
          mshadow::red::sum::Reduce(sum, grad_ * grad_, residual);
        }
      }
      DType grad_norm = std::sqrt(sum);
      if (grad_norm > clip_group_gradient_norm) {
        group_rescale =  clip_group_gradient_norm / grad_norm;
      }
    }

    // Apply gradient
    for (index_t j = 0; j < row_length; j++) {
      index_t data_i = grad_idx[i] * row_length + j;
      index_t grad_i = i * row_length + j;
      // No need to use KERNEL_ASSIGN, as we already checked req is
      // kWriteInplace
      out[data_i] = weight[data_i] -
        lr * group_rescale * rescale_grad * grad_val[grad_i];
    }

    // Apply proximal operator
    DType sum, residual;
    mshadow::red::sum::SetInitValue(sum, residual);
    for (index_t j = 0; j < row_length; j++) {
      index_t data_i = grad_idx[i] * row_length + j;
      mshadow::red::sum::Reduce(sum, out[data_i] * out[data_i], residual);
    }
    DType weight_norm = std::sqrt(sum);

    // Compute number of weight updates skipped due to lazy_update
    DType num_skipped = current_update - last_update_buffer[grad_idx[i]];
    last_update_buffer[grad_idx[i]] = current_update;

    // In case of erroneous last_update_buffer
    if (num_skipped < 1) {
      std::printf("Got invalid last_update_buffer in proximal_sgd_update. "
                  "Ignoring.");
      num_skipped = 1;
    }
    DType scaled_l2_regularization_strength =
        l2_regularization_strength * num_skipped * lr * group_rescale;

    // Soft threshold weights (proximal map for group lasso)
    if (scaled_l2_regularization_strength >= weight_norm) {
      for (index_t j = 0; j < row_length; j++) {
        // No need to use KERNEL_ASSIGN, as we already checked req is
        // kWriteInplace
        index_t data_i = grad_idx[i] * row_length + j;
        out[data_i] = 0;
      }
    } else {
      for (index_t j = 0; j < row_length; j++) {
        // No need to use KERNEL_ASSIGN, as we already checked req is
        // kWriteInplace
        index_t data_i = grad_idx[i] * row_length + j;
        out[data_i] = out[data_i] - (scaled_l2_regularization_strength *
                                     out[data_i] / weight_norm);
      }
    }
  }
};

template <typename xpu> struct EagerProximalSGDDnsRspKernel {
  // DType is the output data type
  // IType is row sparse idx type
  // i is the ith row in row sparse gradient
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void
  Map(int i, const index_t row_length, const index_t num_grad, DType *out,
      const DType *weight, const IType *grad_idx, const DType *grad_val,
      DType *last_update_buffer, const DType current_update,
      const DType clip_gradient, const DType clip_group_gradient_norm,
      const DType lr, const DType rescale_grad,
      const DType l2_regularization_strength) {
    // Check if there is a gradient for this group
    // TODO use binary search
    index_t i_grad_idx = num_grad;
    for (index_t j = 0; j < num_grad; j++) {
      if (grad_idx[j] == i) {
        i_grad_idx = j;
        break;
      }
    }

    DType group_rescale = 1;
    if (i_grad_idx != num_grad) {
      // Check if gradient needs to be rescaled
      if (clip_group_gradient_norm >= 0.0f) {
        DType sum, residual;
        mshadow::red::sum::SetInitValue(sum, residual);
        if (clip_gradient >= 0.0f) {
          for (index_t j = 0; j < row_length; j++) {
            index_t grad_i = i_grad_idx * row_length + j;
            DType grad_ = mshadow_op::clip::Map(rescale_grad * grad_val[grad_i],
                                                clip_gradient);
            mshadow::red::sum::Reduce(sum, grad_ * grad_, residual);
          }
        } else {
          for (index_t j = 0; j < row_length; j++) {
            index_t grad_i = i_grad_idx * row_length + j;
            DType grad_ = rescale_grad * grad_val[grad_i];
            mshadow::red::sum::Reduce(sum, grad_ * grad_, residual);
          }
        }
        DType grad_norm = std::sqrt(sum);
        if (grad_norm > clip_group_gradient_norm) {
          group_rescale = clip_group_gradient_norm / grad_norm;
        }
      }

      // Apply gradient
      for (index_t j = 0; j < row_length; j++) {
        index_t data_i = i * row_length + j;
        index_t grad_i = i_grad_idx * row_length + j;
        // No need to use KERNEL_ASSIGN, as we already checked req is
        // kWriteInplace
        out[data_i] = weight[data_i] -
                      lr * group_rescale * rescale_grad * grad_val[grad_i];
      }
    }

    // Apply proximal operator
    DType sum, residual;
    mshadow::red::sum::SetInitValue(sum, residual);
    for (index_t j = 0; j < row_length; j++) {
      index_t data_i = i * row_length + j;
      mshadow::red::sum::Reduce(sum, out[data_i] * out[data_i], residual);
    }
    DType weight_norm = std::sqrt(sum);

    // Compute number of weight updates skipped due to lazy_update
    DType num_skipped = current_update - last_update_buffer[i];
    last_update_buffer[i] = current_update;

    // In case of erroneous last_update_buffer
    if (num_skipped < 1) {
      std::printf("Got invalid last_update_buffer in proximal_sgd_update. "
                  "Ignoring.");
      num_skipped = 1;
    }
    DType scaled_l2_regularization_strength =
        l2_regularization_strength * num_skipped * lr * group_rescale;

    // Soft threshold weights (proximal map for group lasso)
    if (scaled_l2_regularization_strength >= weight_norm) {
      for (index_t j = 0; j < row_length; j++) {
        // No need to use KERNEL_ASSIGN, as we already checked req is
        // kWriteInplace
        index_t data_i = i * row_length + j;
        out[data_i] = 0;
      }
    } else {
      for (index_t j = 0; j < row_length; j++) {
        // No need to use KERNEL_ASSIGN, as we already checked req is
        // kWriteInplace
        index_t data_i = i * row_length + j;
        out[data_i] = out[data_i] - (scaled_l2_regularization_strength *
                                     out[data_i] / weight_norm);
      }
    }
  }
};

/*
 * \brief SGD update implementation for dense weight and row_sparse grad.
 *        Both standard update and lazy update are supported.
 */
template <typename xpu>
inline void
ProximalSGDUpdateDnsRspImpl(const ProximalSGDParam &param, const OpContext &ctx,
                            const TBlob &weight, const NDArray &grad,
                            const TBlob &last_update_buffer,
                            const OpReqType &req, TBlob *out) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(grad.storage_type(), kRowSparseStorage);
  // if gradients are zeros, no weights are updated
  if (req == kNullOp)
    return;
  CHECK_EQ(req, kWriteInplace)
      << "kWriteInplace is expected for sparse proximal_sgd_update";
  CHECK_GT(weight.shape_.Size(), 0);

  MSHADOW_REAL_TYPE_SWITCH(weight.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(grad.aux_type(rowsparse::kIdx), IType, {
      DType *weight_data = weight.dptr<DType>();
      DType *out_data = out->dptr<DType>();
      const IType *grad_idx = grad.aux_data(rowsparse::kIdx).dptr<IType>();
      const DType *grad_val = grad.data().dptr<DType>();
      DType *last_update_data = last_update_buffer.dptr<DType>();
      const nnvm::dim_t num_grad = grad.aux_shape(rowsparse::kIdx)[0];
      const auto row_length = weight.shape_.ProdShape(1, weight.ndim());

      // When performing eager update, iterate over all rows
      if (!param.lazy_update) {
        size_t num_threads = weight.shape_[0];
        Kernel<EagerProximalSGDDnsRspKernel<xpu>, xpu>::Launch(
            s, num_threads, row_length, num_grad, out_data, weight_data,
            grad_idx, grad_val, last_update_data,
            static_cast<DType>(param.current_update),
            static_cast<DType>(param.clip_gradient),
            static_cast<DType>(param.clip_group_gradient_norm),
            static_cast<DType>(param.lr),
            static_cast<DType>(param.rescale_grad),
            static_cast<DType>(param.l2_regularization_strength));
      } else if (grad.storage_initialized()) {
        size_t num_threads = num_grad;
        Kernel<ProximalSGDDnsRspKernel<xpu>, xpu>::Launch(
            s, num_threads, row_length, out_data, weight_data, grad_idx,
            grad_val, last_update_data,
            static_cast<DType>(param.current_update),
            static_cast<DType>(param.clip_gradient),
            static_cast<DType>(param.clip_group_gradient_norm),
            static_cast<DType>(param.lr),
            static_cast<DType>(param.rescale_grad),
            static_cast<DType>(param.l2_regularization_strength));
      } else { // Lazy update with 0 gradient
        return;
      }
    });
  });
}

/*
 * \brief SGD update implementation for row_sparse grad.
 *        Both standard update and lazy update are supported.
 */
template <typename xpu>
inline void ProximalSGDUpdateRspImpl(const ProximalSGDParam &param,
                                     const OpContext &ctx,
                                     const NDArray &weight, const NDArray &grad,
                                     const NDArray &last_update_buffer,
                                     const OpReqType &req, NDArray *out) {
  CheckAllRowsPresent(weight, "SGDUpdate", "weights");
  // reuse dns rsp implementation when storage_shape == shape
  TBlob out_blob = out->data();
  ProximalSGDUpdateDnsRspImpl<xpu>(param, ctx, weight.data(), grad,
                                   last_update_buffer.data(), req, &out_blob);
}

template <typename xpu>
inline void ProximalSGDUpdate(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
  CHECK_EQ(0, 1) << "unimplemented";
}

template <typename xpu>
inline void ProximalSGDUpdateEx(const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const std::vector<NDArray> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &outputs) {
  const ProximalSGDParam &param = nnvm::get<ProximalSGDParam>(attrs.parsed);
  const auto w_stype = inputs[0].storage_type();
  const auto g_stype = inputs[1].storage_type();
  const auto o_stype = outputs[0].storage_type();
  if (o_stype == w_stype && g_stype == kRowSparseStorage &&
      (w_stype == kDefaultStorage || w_stype == kRowSparseStorage)) {
    NDArray out = outputs[0];
    // std update and lazy update with rsp grad
    ProximalSGDUpdateRspImpl<xpu>(param, ctx, inputs[0], inputs[1], inputs[2],
                                  req[0], &out);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}


struct FTMLParam : public dmlc::Parameter<FTMLParam> {
  float lr;
  float beta1;
  float beta2;
  double epsilon;
  int t;
  float wd;
  float rescale_grad;
  float clip_grad;
  DMLC_DECLARE_PARAMETER(FTMLParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate.");
    DMLC_DECLARE_FIELD(beta1)
    .set_default(0.6f)
    .set_range(0.0f, 1.0f)
    .describe("Generally close to 0.5.");
    DMLC_DECLARE_FIELD(beta2)
    .set_default(0.999f)
    .set_range(0.0f, 1.0f)
    .describe("Generally close to 1.");
    DMLC_DECLARE_FIELD(epsilon)
    .set_default(1e-8f)
    .describe("Epsilon to prevent div 0.");
    DMLC_DECLARE_FIELD(t)
    .describe("Number of update.");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_grad)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
  }
};


struct FTMLKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, DType* weight, DType* grad,
    DType* d, DType* v, DType* z, const DType lr, const DType beta1,
    const DType beta2, const DType epsilon, const DType t,
    const DType wd, const DType rescale_grad, const DType clip_grad,
    const OpReqType req) {
    using namespace mshadow_op;
    const DType grad_i = clip_grad >= 0.0f
        ? clip::Map(rescale_grad * grad[i] + wd * weight[i], clip_grad)
        : (rescale_grad * grad[i] + wd * weight[i]);
    v[i] = beta2 * v[i] + (1 - beta2) * square::Map(grad_i);
    const DType d_t = (1 - power::Map(beta1, t)) / lr *
        (square_root::Map(v[i] / (1 - power::Map(beta2, t))) + epsilon);
    z[i] = beta1 * z[i] + (1 - beta1) * grad_i - (d_t - beta1 * d[i]) * weight[i];
    d[i] = d_t;
    KERNEL_ASSIGN(out[i], req, - z[i] / d_t);
  }
};


template<typename xpu>
inline void FTMLUpdate(const nnvm::NodeAttrs& attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  FTMLParam param = nnvm::get<FTMLParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    DType* weight_data = inputs[0].dptr<DType>();
    DType* grad_data = inputs[1].dptr<DType>();
    DType* d_data = inputs[2].dptr<DType>();
    DType* v_data = inputs[3].dptr<DType>();
    DType* z_data = inputs[4].dptr<DType>();
    DType* out_data = outputs[0].dptr<DType>();
    Kernel<FTMLKernel, xpu>::Launch(s, inputs[0].shape_.Size(), out_data,
      weight_data, grad_data, d_data, v_data, z_data, static_cast<DType>(param.lr),
      static_cast<DType>(param.beta1), static_cast<DType>(param.beta2),
      static_cast<DType>(param.epsilon), static_cast<DType>(param.t), static_cast<DType>(param.wd),
      static_cast<DType>(param.rescale_grad), static_cast<DType>(param.clip_grad),
      req[0]);
  });
}

struct AdamParam : public dmlc::Parameter<AdamParam> {
  float lr;
  float beta1;
  float beta2;
  float epsilon;
  float wd;
  float rescale_grad;
  float clip_gradient;
  bool lazy_update;
  DMLC_DECLARE_PARAMETER(AdamParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(beta1)
    .set_default(0.9f)
    .describe("The decay rate for the 1st moment estimates.");
    DMLC_DECLARE_FIELD(beta2)
    .set_default(0.999f)
    .describe("The decay rate for the 2nd moment estimates.");
    DMLC_DECLARE_FIELD(epsilon)
    .set_default(1e-8f)
    .describe("A small constant for numerical stability.");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(lazy_update)
    .set_default(true)
    .describe("If true, lazy updates are applied if gradient's stype is row_sparse "
              "and all of w, m and v have the same stype");
  }
};

template<typename xpu>
inline void AdamUpdate(const nnvm::NodeAttrs& attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  const AdamParam& param = nnvm::get<AdamParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> mean = inputs[2].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> var = inputs[3].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);

    grad = scalar<DType>(param.rescale_grad) * grad +
      scalar<DType>(param.wd) * weight;

    if (param.clip_gradient >= 0.0f) {
      mean = scalar<DType>(param.beta1)*mean + scalar<DType>(1.f-param.beta1) *
          F<clip>(grad, DType(param.clip_gradient));
      var = scalar<DType>(param.beta2)*var + scalar<DType>(1.f-param.beta2)*F<square>(
          F<clip>(grad, DType(param.clip_gradient)));
    } else {
      mean = scalar<DType>(param.beta1)*mean + scalar<DType>(1.f-param.beta1) * grad;
      var = scalar<DType>(param.beta2)*var + scalar<DType>(1.f-param.beta2) * F<square>(grad);
    }
    Assign(out, req[0],
           weight -
           scalar<DType>(param.lr) * mean /
           (F<square_root>(var) + scalar<DType>(param.epsilon)));
  });
}

template<int req, typename xpu>
struct AdamDnsRspDnsKernel;

/*!
 * Note: this kernel performs sparse adam update. For each row-slice in row_sparse
 * gradient, it finds the corresponding elements in weight, mean and var and performs
 * the update.
 * The kernel assumes dense weight/mean/var, and row_sparse gradient
 */
template<int req>
struct AdamDnsRspDnsKernel<req, cpu> {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, const nnvm::dim_t row_length, DType* out_data,
    DType* mean_data, DType* var_data, const DType* weight_data, const IType* grad_idx,
    const DType* grad_data, const DType clip_gradient, const DType beta1, const DType beta2,
    const DType lr, const DType wd, const DType epsilon, const DType rescale_grad) {
    using nnvm::dim_t;
    using namespace mshadow_op;
    const dim_t row_offset = grad_idx[i] * row_length;
    for (dim_t j = 0; j < row_length; j++) {
      // index in data/mean/var
      const dim_t data_i = row_offset + j;
      // index in grad
      const dim_t grad_i = i * row_length + j;
      const DType grad_rescaled = grad_data[grad_i] * rescale_grad + weight_data[data_i] * wd;
      if (clip_gradient >= 0.0f) {
        mean_data[data_i] = beta1 * mean_data[data_i] + (1.f - beta1) *
                            clip::Map(grad_rescaled, clip_gradient);
        var_data[data_i] =  beta2 * var_data[data_i] + (1.f - beta2) * square::Map(
                            clip::Map(grad_rescaled, clip_gradient));
      } else {
        mean_data[data_i] = beta1 * mean_data[data_i] + (1.f - beta1) * grad_rescaled;
        var_data[data_i] = beta2 * var_data[data_i] +
                           (1.f - beta2) * grad_rescaled * grad_rescaled;
      }
      KERNEL_ASSIGN(out_data[data_i], req, weight_data[data_i] - lr * mean_data[data_i] /
                    (square_root::Map(var_data[data_i]) + epsilon));
    }
  }
};


template<int req>
struct AdamDnsRspDnsKernel<req, gpu> {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, const nnvm::dim_t row_length, DType* out_data,
    DType* mean_data, DType* var_data, const DType* weight_data, const IType* grad_idx,
    const DType* grad_data, const DType clip_gradient, const DType beta1, const DType beta2,
    const DType lr, const DType wd, const DType epsilon, const DType rescale_grad) {
    using nnvm::dim_t;
    using namespace mshadow_op;
    const dim_t row_id = i / row_length;
    const dim_t col_id = i % row_length;
    const dim_t row_offset = grad_idx[row_id] * row_length;
    // index in data/mean/var
    const dim_t data_i = row_offset + col_id;
    // index in grad
    DType grad_rescaled = grad_data[i] * rescale_grad + weight_data[data_i] * wd;
    if (clip_gradient >= 0.0f) {
      grad_rescaled = clip::Map(grad_rescaled, clip_gradient);
    }
    mean_data[data_i] = beta1 * mean_data[data_i] + (1.f - beta1) * grad_rescaled;
    var_data[data_i] = beta2 * var_data[data_i] +
                       (1.f - beta2) * grad_rescaled * grad_rescaled;
    KERNEL_ASSIGN(out_data[data_i], req, weight_data[data_i] - lr * mean_data[data_i] /
                  (square_root::Map(var_data[data_i]) + epsilon));
  }
};

/*
 * \brief lazy adam update for dense weight, dense states and rsp grad.
 */
template<typename xpu>
inline void AdamLazyUpdateDnsRspDnsImpl(const AdamParam& param,
                                        const OpContext& ctx,
                                        const TBlob& weight,
                                        const NDArray& grad,
                                        const TBlob& mean,
                                        const TBlob& var,
                                        const OpReqType& req,
                                        TBlob *out) {
  using namespace mxnet_op;
  using namespace rowsparse;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  if (!grad.storage_initialized() || req == kNullOp) return;
  CHECK_EQ(req, kWriteInplace) << "kWriteInplace is expected for sparse adam_update";
  CHECK_GT(weight.shape_.Size(), 0);
  CHECK_GT(mean.shape_.Size(), 0);
  CHECK_GT(var.shape_.Size(), 0);

  MSHADOW_REAL_TYPE_SWITCH(weight.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(grad.aux_type(kIdx), IType, {
      MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
        const DType* weight_data = weight.dptr<DType>();
        const IType* grad_idx = grad.aux_data(kIdx).dptr<IType>();
        const DType* grad_val = grad.data().dptr<DType>();
        DType* mean_data = mean.dptr<DType>();
        DType* var_data = var.dptr<DType>();
        DType* out_data = out->dptr<DType>();
        nnvm::dim_t num_rows = grad.aux_shape(kIdx)[0];
        const auto row_length = weight.shape_.ProdShape(1, weight.ndim());
        size_t num_threads = num_rows;
        if (std::is_same<xpu, gpu>::value) {
          num_threads = num_rows * row_length;
        }
        Kernel<AdamDnsRspDnsKernel<req_type, xpu>, xpu>::Launch(s, num_threads,
          row_length, out_data, mean_data, var_data, weight_data, grad_idx, grad_val,
          static_cast<DType>(param.clip_gradient), static_cast<DType>(param.beta1),
          static_cast<DType>(param.beta2), static_cast<DType>(param.lr),
          static_cast<DType>(param.wd), static_cast<DType>(param.epsilon),
          static_cast<DType>(param.rescale_grad));
      });
    });
  });
}

/*
 * \brief lazy adam update for both row_sparse and dense weight.
 *        grad is expected to be row_sparse.
 */
template<typename xpu>
inline void AdamLazyUpdateRspImpl(const AdamParam& param,
                                  const OpContext& ctx,
                                  const NDArray& weight,
                                  const NDArray& grad,
                                  const NDArray& mean,
                                  const NDArray& var,
                                  const OpReqType& req,
                                  NDArray *out) {
  using namespace mxnet_op;
  using namespace rowsparse;
  CheckAllRowsPresent(weight, "AdamUpdate", "weights");
  Stream<xpu>* s = ctx.get_stream<xpu>();
  // fill mean and variance with zero values in order to reuse the sgd mom dns impl
  if (mean.storage_type() == kRowSparseStorage && !mean.storage_initialized()) {
    NDArray mean_zeros = mean;
    FillDnsZerosRspImpl(s, &mean_zeros);
  }
  if (var.storage_type() == kRowSparseStorage && !var.storage_initialized()) {
    NDArray var_zeros = var;
    FillDnsZerosRspImpl(s, &var_zeros);
  }
  TBlob out_blob = out->data();
  // reuse dns rsp implementation when storage_shape == shape
  AdamLazyUpdateDnsRspDnsImpl<xpu>(param, ctx, weight.data(), grad, mean.data(),
                                   var.data(), req, &out_blob);
}

/*
 * \brief kernel for standard adam update for dense weight, row_sparse grad and dense states.
 */
template<int req, typename xpu>
struct AdamStdDnsRspDnsKernel;

/*
 * \brief standard adam update for dense weight, row_sparse grad and dense states.
 */
template<typename xpu>
void AdamStdUpdateDnsRspDnsImpl(const AdamParam& param,
                                const OpContext& ctx,
                                const TBlob& weight,
                                const NDArray& grad,
                                const TBlob& mean,
                                const TBlob& var,
                                const OpReqType& req,
                                TBlob *out);

/*
 * \brief standard adam update for both row_sparse and dense weight.
 *        states are expected to be dense, while grad is expected to be row_sparse.
 */
template<typename xpu>
inline void AdamStdUpdateRspImpl(const AdamParam& param,
                                 const OpContext& ctx,
                                 const NDArray& weight,
                                 const NDArray& grad,
                                 const NDArray& mean,
                                 const NDArray& var,
                                 const OpReqType& req,
                                 NDArray *out) {
  using namespace mxnet_op;
  using namespace rowsparse;
  CheckAllRowsPresent(weight, "AdamStdUpdate", "weights");
  TBlob out_blob = out->data();
  // reuse dns rsp implementation when storage_shape == shape
  AdamStdUpdateDnsRspDnsImpl<xpu>(param, ctx, weight.data(), grad, mean.data(),
                                  var.data(), req, &out_blob);
}

template<typename xpu>
inline void AdamUpdateEx(const nnvm::NodeAttrs& attrs,
                         const OpContext &ctx,
                         const std::vector<NDArray> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &outputs) {
  const AdamParam& param = nnvm::get<AdamParam>(attrs.parsed);
  const auto w_stype = inputs[0].storage_type();
  const auto g_stype = inputs[1].storage_type();
  const auto m_stype = inputs[2].storage_type();
  const auto v_stype = inputs[3].storage_type();
  const auto out_stype = outputs[0].storage_type();
  NDArray out = outputs[0];
  const bool valid_weight = w_stype == kDefaultStorage || w_stype == kRowSparseStorage;
  CHECK(w_stype == out_stype) << "Inconsistent weight stype and output stype";
  CHECK(m_stype == v_stype) << "Inconsistent mean stype and var stype";
  if (valid_weight && g_stype == kRowSparseStorage && m_stype == w_stype) {
     if (param.lazy_update) {
       // rsp grad && m.stype = w.stype && lazy_update = true -> lazy update
       AdamLazyUpdateRspImpl<xpu>(param, ctx, inputs[0], inputs[1], inputs[2],
                                  inputs[3], req[0], &out);
     } else {
       // rsp grad && m.stype = w.stype && lazy_update = false -> std update
       AdamStdUpdateRspImpl<xpu>(param, ctx, inputs[0], inputs[1], inputs[2],
                                 inputs[3], req[0], &out);
     }
  } else if (w_stype == kRowSparseStorage && g_stype == kRowSparseStorage &&
             m_stype == kDefaultStorage) {
     // rsp, rsp, dns, dns -> rsp, standard update
     AdamStdUpdateRspImpl<xpu>(param, ctx, inputs[0], inputs[1], inputs[2],
                               inputs[3], req[0], &out);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

// This RMSProp code follows the version in
// http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45)
// by Alex Graves, 2013.
struct RMSPropAlexParam : public dmlc::Parameter<RMSPropAlexParam> {
  float lr;
  float gamma1;
  float gamma2;
  float epsilon;
  float wd;
  float rescale_grad;
  float clip_gradient;
  float clip_weights;
  DMLC_DECLARE_PARAMETER(RMSPropAlexParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(gamma1).set_default(0.95f)
    .describe("Decay rate.");
    DMLC_DECLARE_FIELD(gamma2).set_default(0.9f)
    .describe("Decay rate.");
    DMLC_DECLARE_FIELD(epsilon).set_default(1e-8f)
    .describe("A small constant for numerical stability.");
    DMLC_DECLARE_FIELD(wd).set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(clip_weights)
    .set_default(-1.0f)
    .describe("Clip weights to the range of [-clip_weights, clip_weights] "
              "If clip_weights <= 0, weight clipping is turned off. "
              "weights = max(min(weights, clip_weights), -clip_weights).");
  }
};

template <typename xpu>
inline void RMSPropAlexUpdate(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  const RMSPropAlexParam &param = nnvm::get<RMSPropAlexParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> state_n = inputs[2].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> state_g = inputs[3].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> delta = inputs[4].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);

    grad = scalar<DType>(param.rescale_grad) * grad +
           scalar<DType>(param.wd) * weight;

    if (param.clip_gradient >= 0.0f) {
      state_n = scalar<DType>(1.f - param.gamma1) *
                    F<clip>(grad, DType(param.clip_gradient)) *
                    F<clip>(grad, DType(param.clip_gradient)) +
                scalar<DType>(param.gamma1) * state_n;
      state_g = scalar<DType>(1.f - param.gamma1) *
                    F<clip>(grad, DType(param.clip_gradient)) +
                scalar<DType>(param.gamma1) * state_g;
      delta = scalar<DType>(param.gamma2) * delta -
              scalar<DType>(param.lr) *
                  (F<clip>(grad, DType(param.clip_gradient)) /
                   (F<square_root>(state_n - state_g * state_g +
                                   scalar<DType>(param.epsilon))));
    } else {
      state_n = scalar<DType>(1.f - param.gamma1) * (grad * grad) +
                scalar<DType>(param.gamma1) * state_n;
      state_g = scalar<DType>(1.f - param.gamma1) * grad +
                scalar<DType>(param.gamma1) * state_g;
      delta = scalar<DType>(param.gamma2) * delta -
              scalar<DType>(param.lr) *
                  (grad / (F<square_root>(state_n - state_g * state_g +
                                          scalar<DType>(param.epsilon))));
    }

    if (param.clip_weights >= 0.0f) {
      Assign(out, req[0], F<clip>(weight + delta, DType(param.clip_weights)));
    } else {
      Assign(out, req[0], weight + delta);
    }
  });
}

// This RMSProp code follows the version in
// http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
// by Tieleman & Hinton, 2012
struct RMSPropParam : public dmlc::Parameter<RMSPropParam> {
  float lr;
  float gamma1;
  float epsilon;
  float wd;
  float rescale_grad;
  float clip_gradient;
  float clip_weights;
  DMLC_DECLARE_PARAMETER(RMSPropParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(gamma1).set_default(0.95f)
    .describe("The decay rate of momentum estimates.");
    DMLC_DECLARE_FIELD(epsilon).set_default(1e-8f)
    .describe("A small constant for numerical stability.");
    DMLC_DECLARE_FIELD(wd).set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(clip_weights)
    .set_default(-1.0f)
    .describe("Clip weights to the range of [-clip_weights, clip_weights] "
              "If clip_weights <= 0, weight clipping is turned off. "
              "weights = max(min(weights, clip_weights), -clip_weights).");
  }
};

template <typename xpu>
inline void RMSPropUpdate(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                          const std::vector<TBlob> &inputs,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  const RMSPropParam &param = nnvm::get<RMSPropParam>(attrs.parsed);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> state_n = inputs[2].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);

    grad = scalar<DType>(param.rescale_grad) * grad +
           scalar<DType>(param.wd) * weight;

    if (param.clip_gradient >= 0.0f) {
      state_n = scalar<DType>(1.f - param.gamma1) *
                    F<clip>(grad, DType(param.clip_gradient)) *
                    F<clip>(grad, DType(param.clip_gradient)) +
                scalar<DType>(param.gamma1) * state_n;
      if (param.clip_weights >= 0.0f) {
        Assign(out, req[0],
               F<clip>(weight -
                       scalar<DType>(param.lr) *
                           (F<clip>(grad, DType(param.clip_gradient)) /
                            (F<square_root>(state_n +
                                            scalar<DType>(param.epsilon)))),
                       DType(param.clip_weights)));
      } else {
        Assign(out, req[0], weight -
                            scalar<DType>(param.lr) *
                              (F<clip>(grad, DType(param.clip_gradient)) /
                               (F<square_root>(state_n +
                                               scalar<DType>(param.epsilon)))));
      }
    } else {
      state_n = scalar<DType>(1.f - param.gamma1) * (grad * grad) +
                scalar<DType>(param.gamma1) * state_n;
      if (param.clip_weights >= 0.0f) {
        Assign(out, req[0],
               F<clip>(weight -
                       scalar<DType>(param.lr) *
                           (grad /
                            (F<square_root>(state_n +
                                            scalar<DType>(param.epsilon)))),
                       DType(param.clip_weights)));
      } else {
        Assign(out, req[0], weight -
                            scalar<DType>(param.lr) *
                              (grad /
                               (F<square_root>(state_n +
                                               scalar<DType>(param.epsilon)))));
      }
    }
  });
}

struct FtrlParam : public dmlc::Parameter<FtrlParam> {
  float lr;
  float lamda1;
  float beta;
  float wd;
  float rescale_grad;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(FtrlParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(lamda1)
    .set_default(0.01f)
    .describe("The L1 regularization coefficient.");
    DMLC_DECLARE_FIELD(beta)
    .set_default(1.0f)
    .describe("Per-Coordinate Learning Rate beta.");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
  }
};

template<typename xpu>
inline void FtrlUpdate(const nnvm::NodeAttrs& attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  const FtrlParam& param = nnvm::get<FtrlParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> z = inputs[2].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> n = inputs[3].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);

    grad = scalar<DType>(param.rescale_grad) * grad;

    if (param.clip_gradient >= 0.0f) {
      z += F<clip>(grad, DType(param.clip_gradient)) - (F<square_root>(n +
           F<square>(F<clip>(grad, DType(param.clip_gradient)))) - F<square_root>(n)) *
           weight / scalar<DType>(param.lr);
      n += F<square>(F<clip>(grad, DType(param.clip_gradient)));
    } else {
      z += grad - (F<square_root>(n + F<square>(grad)) - F<square_root>(n)) *
           weight / scalar<DType>(param.lr);
      n += F<square>(grad);
    }
    Assign(out, req[0],
           (F<sign>(z) * scalar<DType>(param.lamda1) - z) /
           ((scalar<DType>(param.beta) + F<square_root>(n)) /
           scalar<DType>(param.lr) + scalar<DType>(param.wd)) *
           F<gt>(F<abs>(z), scalar<DType>(param.lamda1)));
  });
}

template<int req>
struct FtrlDnsRspDnsKernel {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, const nnvm::dim_t row_length, DType* out_data,
    DType* z_data, DType* n_data, const DType* weight_data, const IType* grad_idx,
    const DType* grad_data, const DType clip_gradient, const DType lamda1, const DType beta,
    const DType lr, const DType wd, const DType rescale_grad) {
    using nnvm::dim_t;
    using namespace mshadow_op;
    const dim_t row_offset = grad_idx[i] * row_length;
    for (dim_t j = 0; j < row_length; j++) {
      // index in data/z/n
      const dim_t data_i = row_offset + j;
      // index in grad
      const dim_t grad_i = i * row_length + j;
      const DType grad_rescaled = grad_data[grad_i] * rescale_grad;
      if (clip_gradient >= 0.0f) {
        z_data[data_i] += clip::Map(grad_rescaled, clip_gradient) -
                          (square_root::Map(n_data[data_i] +
                          square::Map(clip::Map(grad_rescaled, clip_gradient))) -
                          square_root::Map(n_data[data_i])) * weight_data[data_i] / lr;
        n_data[data_i] += square::Map(clip::Map(grad_rescaled, clip_gradient));
      } else {
        z_data[data_i] += grad_rescaled - (square_root::Map(n_data[data_i] +
                          square::Map(grad_rescaled)) - square_root::Map(n_data[data_i])) *
                          weight_data[data_i] / lr;
        n_data[data_i] += square::Map(grad_rescaled);
      }
      KERNEL_ASSIGN(out_data[data_i], req,
                    (sign::Map(z_data[data_i]) * lamda1 - z_data[data_i]) /
                    ((beta + square_root::Map(n_data[data_i])) / lr + wd) *
                    gt::Map(abs::Map(z_data[data_i]), lamda1));
    }
  }
};


template<typename xpu>
inline void FtrlUpdateDnsRspDnsImpl(const FtrlParam& param,
                                    const OpContext& ctx,
                                    const TBlob& weight,
                                    const NDArray& grad,
                                    const TBlob& z,
                                    const TBlob& n,
                                    const OpReqType& req,
                                    TBlob *out) {
  using namespace mxnet_op;
  using namespace rowsparse;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  if (!grad.storage_initialized() || req == kNullOp) return;
  CHECK_EQ(req, kWriteInplace) << "kWriteInplace is expected for sparse ftrl_update";
  CHECK_GT(weight.shape_.Size(), 0);
  CHECK_GT(z.shape_.Size(), 0);
  CHECK_GT(n.shape_.Size(), 0);

  MSHADOW_REAL_TYPE_SWITCH(weight.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(grad.aux_type(kIdx), IType, {
      MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
        const DType* weight_data = weight.dptr<DType>();
        const IType* grad_idx = grad.aux_data(kIdx).dptr<IType>();
        const DType* grad_val = grad.data().dptr<DType>();
        DType* z_data = z.dptr<DType>();
        DType* n_data = n.dptr<DType>();
        DType* out_data = out->dptr<DType>();
        nnvm::dim_t num_rows = grad.aux_shape(kIdx)[0];
        const auto row_length = weight.shape_.ProdShape(1, weight.ndim());
        Kernel<FtrlDnsRspDnsKernel<req_type>, xpu>::Launch(s, num_rows, row_length,
          out_data, z_data, n_data, weight_data, grad_idx, grad_val,
          static_cast<DType>(param.clip_gradient), static_cast<DType>(param.lamda1),
          static_cast<DType>(param.beta), static_cast<DType>(param.lr),
          static_cast<DType>(param.wd), static_cast<DType>(param.rescale_grad));
      });
    });
  });
}

template<typename xpu>
inline void FtrlUpdateRspRspRspImpl(const FtrlParam& param,
                                    const OpContext& ctx,
                                    const NDArray& weight,
                                    const NDArray& grad,
                                    const NDArray& z,
                                    const NDArray& n,
                                    const OpReqType& req,
                                    NDArray *out) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet_op;
  using namespace rowsparse;
  CheckAllRowsPresent(weight, "FtrlUpdate", "weights");
  Stream<xpu>* s = ctx.get_stream<xpu>();
  // fill z and n with zero values in order to reuse the sgd mom dns impl
  if (!z.storage_initialized()) {
    NDArray z_zeros = z;
    FillDnsZerosRspImpl(s, &z_zeros);
  }
  if (!n.storage_initialized()) {
    NDArray n_zeros = n;
    FillDnsZerosRspImpl(s, &n_zeros);
  }
  TBlob out_blob = out->data();
  // reuse dns rsp implementation when storage_shape == shape
  FtrlUpdateDnsRspDnsImpl<xpu>(param, ctx, weight.data(), grad, z.data(),
                               n.data(), req, &out_blob);
}

template<typename xpu>
inline void FtrlUpdateEx(const nnvm::NodeAttrs& attrs,
                         const OpContext &ctx,
                         const std::vector<NDArray> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<NDArray> &outputs) {
  const FtrlParam& param = nnvm::get<FtrlParam>(attrs.parsed);
  const auto weight_stype = inputs[0].storage_type();
  const auto z_stype = inputs[2].storage_type();
  const auto n_stype = inputs[3].storage_type();

  const auto out_stype = outputs[0].storage_type();
  CHECK_EQ(z_stype, weight_stype) << "Inconsistent storage type detected between "
           << " z.stype = " << z_stype << " and weight.stype = " << weight_stype;
  CHECK_EQ(n_stype, weight_stype) << "Inconsistent storage type detected between "
           << " n.stype = " << n_stype << " and weight.stype = " << weight_stype;
  if (common::ContainsOnlyStorage(inputs, kRowSparseStorage) && out_stype == kRowSparseStorage) {
     NDArray out = outputs[0];
     FtrlUpdateRspRspRspImpl<xpu>(param, ctx, inputs[0], inputs[1], inputs[2],
                                  inputs[3], req[0], &out);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}


// Implementation for signSGD and Signum

struct SignSGDParam : public dmlc::Parameter<SignSGDParam> {
  float lr;
  float wd;
  float rescale_grad;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(SignSGDParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
  }
};


struct SignSGDKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* weight_data,
    const DType* grad_data, const DType param_clip_gradient,
    const DType param_lr, const DType param_wd, const DType param_rescale_grad,
    const OpReqType req) {

    // param_clip_gradient has no effect for SignSGD
    KERNEL_ASSIGN(out_data[i], req,
             (1.f-param_lr*param_wd)*weight_data[i]
               - (param_lr)*((grad_data[i] > 0) - (grad_data[i] < 0)));
  }
};

template<typename xpu>
inline void SignSGDUpdate(const nnvm::NodeAttrs& attrs,
                      const OpContext &ctx,
                      const std::vector<TBlob> &inputs,
                      const std::vector<OpReqType> &req,
                      const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  const SignSGDParam& param = nnvm::get<SignSGDParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    Kernel<SignSGDKernel, xpu>::Launch(s, weight.shape_.Size(), out.dptr_, weight.dptr_,
      grad.dptr_, static_cast<DType>(param.clip_gradient),
      static_cast<DType>(param.lr), static_cast<DType>(param.wd),
      static_cast<DType>(param.rescale_grad), req[0]);
  });
}


struct SignumParam : public dmlc::Parameter<SignumParam> {
  float lr;
  float momentum;
  float wd;
  float rescale_grad;
  float clip_gradient;
  float wd_lh;  // the amount of algorithmic weight decay by Loshchilov and Frank Hutter
  DMLC_DECLARE_PARAMETER(SignumParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(momentum)
    .set_default(0.0f)
    .describe("The decay rate of momentum estimates at each epoch.");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("Weight decay augments the objective function with a "
              "regularization term that penalizes large weights. "
              "The penalty scales with the square of the magnitude of each weight.");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(wd_lh)
    .set_default(0.0f)
    .describe("The amount of weight decay that does not go into gradient/momentum calculations"
              "otherwise do weight decay algorithmically only.");
  }
};

struct SignumKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, DType* mom_data, const DType* weight_data,
    const DType* grad_data, const DType param_clip_gradient, const DType param_momentum,
    const DType param_lr, const DType param_wd, const DType param_rescale_grad,
    const DType param_wd_lh, const OpReqType req) {
    if (param_clip_gradient >= 0.0f) {
      mom_data[i] = param_momentum*mom_data[i]
              - (1-param_momentum)*param_wd*weight_data[i]
              - (1-param_momentum)
              *mshadow_op::clip::Map(param_rescale_grad*grad_data[i], param_clip_gradient);
    } else {
      mom_data[i] = param_momentum*mom_data[i]
                - (1-param_momentum)*param_wd*weight_data[i]
                - (1-param_momentum)*param_rescale_grad*grad_data[i];
    }
    KERNEL_ASSIGN(out_data[i], req, (1.f-param_lr*param_wd_lh)*weight_data[i]
      + (param_lr)*((mom_data[i] > 0) - (mom_data[i] < 0)));
  }
};

template<typename xpu>
inline void SignumUpdate(const nnvm::NodeAttrs& attrs,
                         const OpContext &ctx,
                         const std::vector<TBlob> &inputs,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &outputs) {
  using namespace mxnet_op;
  SignumParam param = nnvm::get<SignumParam>(attrs.parsed);
  Stream<xpu>* s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    Tensor<xpu, 2, DType> weight = inputs[0].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> grad = inputs[1].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> mom = inputs[2].FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = outputs[0].FlatTo2D<xpu, DType>(s);
    Kernel<SignumKernel, xpu>::Launch(s, weight.shape_.Size(), out.dptr_, mom.dptr_, weight.dptr_,
      grad.dptr_, static_cast<DType>(param.clip_gradient), static_cast<DType>(param.momentum),
      static_cast<DType>(param.lr), static_cast<DType>(param.wd),
      static_cast<DType>(param.rescale_grad), static_cast<DType>(param.wd_lh), req[0]);
    });
}

struct AdagradParam : public dmlc::Parameter<AdagradParam> {
  float lr;
  float epsilon;
  float rescale_grad;
  float clip_gradient;
  float wd;
  DMLC_DECLARE_PARAMETER(AdagradParam) {
    DMLC_DECLARE_FIELD(lr)
    .describe("Learning rate");
    DMLC_DECLARE_FIELD(epsilon)
    .set_default(1.0e-7)
    .describe("epsilon");
    DMLC_DECLARE_FIELD(wd)
    .set_default(0.0f)
    .describe("weight decay");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("Clip gradient to the range of [-clip_gradient, clip_gradient] "
              "If clip_gradient <= 0, gradient clipping is turned off. "
              "grad = max(min(grad, clip_gradient), -clip_gradient).");
  }
};

inline bool AdagradStorageType(const nnvm::NodeAttrs& attrs,
                               const int dev_mask,
                               DispatchMode* dispatch_mode,
                               std::vector<int>* in_attrs,
                               std::vector<int>* out_attrs) {
  const AdagradParam& param = nnvm::get<AdagradParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int weight_stype = in_attrs->at(0);
  const int grad_stype = in_attrs->at(1);
  const int state_stype = in_attrs->at(2);
  bool dispatched = false;
  if (!dispatched && grad_stype == kRowSparseStorage &&
      (weight_stype == kRowSparseStorage || weight_stype == kDefaultStorage) &&
      state_stype == weight_stype && param.wd == 0.0f) {
    // weight and state share stype, grad's stype = rsp
    dispatched = storage_type_assign(
        out_attrs, static_cast<NDArrayStorageType>(weight_stype), dispatch_mode,
        DispatchMode::kFComputeEx);
  }
  return dispatched;
}

template<typename xpu>
struct AdagradDnsRspDnsKernel;

template<>
struct AdagradDnsRspDnsKernel<cpu> {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, index_t row_length, DType* out_data,
    DType* state_data, const DType* weight_data, const IType* grad_idx,
    const DType* grad_data, const DType clip_gradient, const DType epsilon,
    const DType lr, const DType rescale_grad) {
    using nnvm::dim_t;
    using namespace mshadow_op;
    const dim_t data_i = grad_idx[i] * row_length;
    const dim_t grad_i = i * row_length;
    for (dim_t j = 0; j < row_length; j++) {
      const dim_t data_j = data_i + j;
      const dim_t grad_j = grad_i + j;
      DType grad_rescaled = grad_data[grad_j] * rescale_grad;
      if (clip_gradient >= 0.0f) {
        grad_rescaled = clip::Map(grad_rescaled, clip_gradient);
      }
      const DType grad_squared = grad_rescaled * grad_rescaled;
      state_data[data_j] += grad_squared;
      const DType div = grad_rescaled / square_root::Map(state_data[data_j] + epsilon);
      // No need to use KERNEL_ASSIGN, as we already checked req is kWriteInplace
      out_data[data_j] = weight_data[data_j] - div * lr;
    }
  }
};

template<>
struct AdagradDnsRspDnsKernel<gpu> {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, index_t row_length, DType* out_data,
    DType* state_data, const DType* weight_data, const IType* grad_idx,
    const DType* grad_data, const DType clip_gradient, const DType epsilon,
    const DType lr, const DType rescale_grad) {
    using nnvm::dim_t;
    using namespace mshadow_op;
    const dim_t row_id = i / row_length;
    const dim_t col_id = i % row_length;
    const dim_t data_i = grad_idx[row_id] * row_length + col_id;
    DType grad_rescaled = grad_data[i] * rescale_grad;
    if (clip_gradient >= 0.0f) {
      grad_rescaled = clip::Map(grad_rescaled, clip_gradient);
    }
    const DType grad_squared = grad_rescaled * grad_rescaled;
    state_data[data_i] += grad_squared;
    const DType div = grad_rescaled / square_root::Map(state_data[data_i] + epsilon);
    // No need to use KERNEL_ASSIGN, as we already checked req is kWriteInplace
    out_data[data_i] = weight_data[data_i] - div * lr;
  }
};

template<typename xpu>
void AdagradUpdateDnsRspDnsImpl(const AdagradParam& param,
                                const OpContext& ctx,
                                const TBlob& weight,
                                const NDArray& grad,
                                const TBlob& state,
                                const OpReqType& req,
                                TBlob *out) {
  using namespace mxnet_op;
  using namespace rowsparse;
  using namespace mshadow;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  CHECK_EQ(param.wd, 0.0f)
    << "sparse adagrad_update does not support wd.";
  if (req == kNullOp || !grad.storage_initialized()) return;
  CHECK_EQ(req, kWriteInplace) << "kWriteInplace is expected for sparse adagrad_update";
  CHECK_GT(weight.shape_.Size(), 0);
  CHECK_GT(state.shape_.Size(), 0);
  MSHADOW_REAL_TYPE_SWITCH(weight.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(grad.aux_type(kIdx), IType, {
      const DType* weight_data = weight.dptr<DType>();
      const IType* grad_idx = grad.aux_data(kIdx).dptr<IType>();
      const DType* grad_val = grad.data().dptr<DType>();
      DType* state_data = state.dptr<DType>();
      DType* out_data = out->dptr<DType>();
      const nnvm::dim_t nnr = grad.storage_shape()[0];
      const auto row_length = weight.shape_.ProdShape(1, weight.ndim());
      size_t num_threads = nnr;
      if (std::is_same<xpu, gpu>::value) {
        num_threads = nnr * row_length;
      }
      Kernel<AdagradDnsRspDnsKernel<xpu>, xpu>::Launch(s, num_threads, row_length,
        out_data, state_data, weight_data, grad_idx, grad_val,
        static_cast<DType>(param.clip_gradient), static_cast<DType>(param.epsilon),
        static_cast<DType>(param.lr), static_cast<DType>(param.rescale_grad));
    });
  });
}

template<typename xpu>
inline void AdagradUpdateRspRspRspImpl(const AdagradParam& param,
                                       const OpContext& ctx,
                                       const NDArray& weight,
                                       const NDArray& grad,
                                       const NDArray& state,
                                       const OpReqType& req,
                                       NDArray *out) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace rowsparse;
  CheckAllRowsPresent(weight, "AdagradUpdate", "weights");
  Stream<xpu>* s = ctx.get_stream<xpu>();
  // fill history with zero values
  if (!state.storage_initialized()) {
    NDArray state_zeros = state;
    FillDnsZerosRspImpl(s, &state_zeros);
  }
  TBlob out_blob = out->data();
  // reuse dns rsp implementation when storage_shape == shape
  AdagradUpdateDnsRspDnsImpl<xpu>(param, ctx, weight.data(), grad,
                                  state.data(), req, &out_blob);
}

template<typename xpu>
inline void AdagradUpdateEx(const nnvm::NodeAttrs& attrs,
                            const OpContext &ctx,
                            const std::vector<NDArray> &inputs,
                            const std::vector<OpReqType> &req,
                            const std::vector<NDArray> &outputs) {
  using namespace mxnet_op;
  const AdagradParam& param = nnvm::get<AdagradParam>(attrs.parsed);

  const auto weight_stype = inputs[0].storage_type();
  const auto grad_stype = inputs[1].storage_type();
  const auto state_stype = inputs[2].storage_type();
  const auto output_stype = outputs[0].storage_type();

  if (common::ContainsOnlyStorage(inputs, kRowSparseStorage) &&
      common::ContainsOnlyStorage(outputs, kRowSparseStorage)) {
    NDArray out = outputs[0];
    AdagradUpdateRspRspRspImpl<xpu>(param, ctx, inputs[0], inputs[1], inputs[2],
                                    req[0], &out);
  } else if (state_stype == weight_stype && output_stype == weight_stype &&
             weight_stype == kDefaultStorage &&
             grad_stype == kRowSparseStorage) {
    TBlob out_blob = outputs[0].data();
    AdagradUpdateDnsRspDnsImpl<xpu>(param, ctx, inputs[0].data(), inputs[1],
                                    inputs[2].data(), req[0],
                                    &out_blob);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

struct ProximalAdagradParam : public dmlc::Parameter<ProximalAdagradParam> {
  float lr;
  float float_stable_epsilon;
  float bisection_epsilon;
  float rescale_grad;
  float clip_gradient;
  float l2_regularization_strength;
  float current_update;
  bool lazy_update;
  bool decay_states;
  float decay_factor;
  bool lazy_decay;
  bool groupwise_lr;
  DMLC_DECLARE_PARAMETER(ProximalAdagradParam) {
    DMLC_DECLARE_FIELD(lr).describe("Learning rate");
    DMLC_DECLARE_FIELD(rescale_grad)
        .set_default(1.0f)
        .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
        .set_default(-1.0f)
        .describe(
            "Clip gradient to the range of [-clip_gradient, clip_gradient] "
            "If clip_gradient <= 0, gradient clipping is turned off. "
            "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(l2_regularization_strength)
        .set_default(0.0f)
        .describe("Lambda term for group lasso objective.");
    DMLC_DECLARE_FIELD(float_stable_epsilon)
      .set_default(1.0e-5)
      .describe("Epsilon for numerical stability");
    DMLC_DECLARE_FIELD(bisection_epsilon)
      .set_default(1.0)
      .describe("Epsilon for bisection algorithm for group sparsity.");
    DMLC_DECLARE_FIELD(current_update)
        .set_default(0.0f)
        .describe("Current update iteration for lazy update with group lasso "
                  "objective.");
    DMLC_DECLARE_FIELD(lazy_update)
        .set_default(true)
        .describe("If true, lazy updates are applied if gradient's stype is "
                  "row_sparse.");
    DMLC_DECLARE_FIELD(decay_states)
      .set_default(false)
      .describe("Decay states as in RMSProp.");
    DMLC_DECLARE_FIELD(decay_factor)
      .set_default(0.9f)
      .describe("Decay factor for states. New gradient is weighted with (1-decay_factor).");
    DMLC_DECLARE_FIELD(lazy_decay)
      .set_default(true)
      .describe("If true, decay is applied lazily.");
    DMLC_DECLARE_FIELD(groupwise_lr)
      .set_default(false)
      .describe("If true, only one learning rate per group is used.");
  }
};

inline bool ProximalAdagradStorageType(const nnvm::NodeAttrs &attrs,
                                       const int dev_mask,
                                       DispatchMode *dispatch_mode,
                                       std::vector<int> *in_attrs,
                                       std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 4U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int weight_stype = in_attrs->at(0);
  const int grad_stype = in_attrs->at(1);
  const int state_stype = in_attrs->at(2);
  const int counter_stype = in_attrs->at(3);
  bool dispatched = false;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    // dns, ... -> dns
    dispatched = storage_type_assign(out_attrs, kDefaultStorage, dispatch_mode,
                                     DispatchMode::kFCompute);
  }
  if (!dispatched && grad_stype == kRowSparseStorage &&
      counter_stype == kDefaultStorage &&
      (weight_stype == kRowSparseStorage || weight_stype == kDefaultStorage) &&
      state_stype == weight_stype) {
    // weight and state share stype, grad's stype = rsp
    dispatched = storage_type_assign(
        out_attrs, static_cast<NDArrayStorageType>(weight_stype), dispatch_mode,
        DispatchMode::kFComputeEx);
  }
  return dispatched;
}

/*! \brief kernel for sparse adagrad update with group sparsity regularization
 */
template <typename xpu> struct ProximalAdagradDnsRspKernel {
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void
  Map(int i, const index_t row_length, const index_t num_grad, DType *out_data,
      DType *state_data, DType *weight_data, const IType *grad_idx,
      const DType *grad_data, DType *last_update_data,
      const DType current_update, const DType clip_gradient,
      const DType rescale_grad, const DType l2_regularization_strength,
      const DType lr, const DType float_stable_epsilon,
      const DType bisection_epsilon, const bool lazy_update,
      const bool decay_states, const DType decay_factor, const bool lazy_decay) {
    using namespace mshadow_op;

    // Eager update: Find location in gradient index
    index_t i_grad_idx = static_cast<index_t>(i);
    if (!lazy_update) {
      i_grad_idx = num_grad; // No gradient for this i
      for (index_t j = 0; j < num_grad; j++) {
        if (grad_idx[j] == i) {
          i_grad_idx = j; // Found a gradient
          break;
        }
      }
    }

    // Helper to obtain index into weight / state arrays for eager / lazy update
    auto get_data_j = [&lazy_update, &i, &grad_idx,
                       &row_length](index_t j) -> index_t {
      if (lazy_update) {
        return grad_idx[i] * row_length + j;
      } else {
        return i * row_length + j;
      }
    };
    // Helper to obtain explicit rescaled and clipped grad for lazy and eager
    // update
    auto get_grad_rescaled = [&i_grad_idx, &lazy_update, &row_length, &num_grad,
                              &grad_data, &rescale_grad,
                              &clip_gradient](index_t j) -> DType {
      if (!lazy_update && i_grad_idx == num_grad) {
        return 0; // No gradient for this parameter
      }

      index_t grad_j = i_grad_idx * row_length + j;
      DType grad_rescaled = grad_data[grad_j] * rescale_grad;
      if (clip_gradient >= 0.0f) {
        grad_rescaled = clip::Map(grad_rescaled, clip_gradient);
      }
      return grad_rescaled;
    };

    // Compute number of weight updates skipped due to lazy_update
    DType num_skipped;
    if (lazy_update) {
      num_skipped = current_update - last_update_data[grad_idx[i]];
      last_update_data[grad_idx[i]] = current_update;
    } else {
      num_skipped = current_update - last_update_data[i];
      last_update_data[i] = current_update;
    }
    // Warn in case of erroneous last_update_buffer
    if (num_skipped < 0) {
      num_skipped = 0;
      std::printf("Got invalid last_update in proximal_adagrad_update. "
                  "Using standard Adagrad update.\n");
    }

    // Update history states
    for (index_t j = 0; j < row_length; j++) {
      const DType grad_rescaled = get_grad_rescaled(j);
      const DType grad_squared = grad_rescaled * grad_rescaled;
      index_t data_j = get_data_j(j);
      if (!decay_states) {
        state_data[data_j] += grad_squared;
      } else {
        if (!lazy_decay) {
          state_data[data_j] =
              state_data[data_j] * std::pow(decay_factor, num_skipped) +
              grad_squared * (1 - decay_factor);
        } else {
          state_data[data_j] = state_data[data_j] * decay_factor +
                               grad_squared * (1 - decay_factor);
        }
      }
    }

    // Compute ||u||₂ using scaled sum of squares
    DType u_ssq, u_scale;
    mshadow_op::nrm2::SetInitValue(u_ssq, u_scale);
    for (index_t j = 0; j < row_length; j++) {
      const DType grad_rescaled = get_grad_rescaled(j);
      index_t data_j = get_data_j(j);
      const DType val = (lr * grad_rescaled) -
                        (std::sqrt(state_data[data_j] + float_stable_epsilon) *
                         weight_data[data_j]);
      mshadow_op::nrm2::Reduce(u_ssq, val, u_scale);
    }
    mshadow_op::nrm2::Finalize(u_ssq, u_scale);
    DType u_norm = u_ssq;

    DType scaled_sparsity = l2_regularization_strength * num_skipped * lr;
    if (scaled_sparsity <= 0) {
      // Standard Adagrad Update
      for (index_t j = 0; j < row_length; j++) {
        // clang-format off
        const DType grad_rescaled = get_grad_rescaled(j);
        index_t data_j = get_data_j(j);
        const DType div = lr * grad_rescaled / square_root::Map(state_data[data_j] + float_stable_epsilon);
        out_data[data_j] = weight_data[data_j] - div;
        // clang-format on
      }
    } else if (u_norm <= scaled_sparsity) {
      // Soft threshold weights (proximal map for group lasso)
      for (index_t j = 0; j < row_length; j++) {
        index_t data_j = get_data_j(j);
        out_data[data_j] = 0;
      }
    } else {
      // Find θ with Algorithm 4 of Dutchi 2011 paper
      // Compute ||ν||₂ using scaled sum of squares
      DType nu_ssq, nu_scale;
      mshadow_op::nrm2::SetInitValue(nu_ssq, nu_scale);
      for (index_t j = 0; j < row_length; j++) {
        // clang-format off
        const DType grad_rescaled = get_grad_rescaled(j);
        index_t data_j = get_data_j(j);
        const DType nu = (lr * grad_rescaled /
                          std::sqrt(state_data[data_j] +
                                    float_stable_epsilon)) -
                          weight_data[data_j];
        mshadow_op::nrm2::Reduce(nu_ssq, nu, nu_scale);
        // clang-format on
      }
      mshadow_op::nrm2::Finalize(nu_ssq, nu_scale);
      DType nu_norm = nu_ssq;

      DType sigma_min, sigma_max;
      mshadow::red::minimum::SetInitValue(sigma_min);
      mshadow::red::maximum::SetInitValue(sigma_max);
      for (index_t j = 0; j < row_length; j++) {
        index_t data_j = get_data_j(j);
        // clang-format off
        const DType sigma = square_root::Map(state_data[data_j] + float_stable_epsilon);
        mshadow::red::minimum::Reduce(sigma_min, sigma);
        mshadow::red::maximum::Reduce(sigma_max, sigma);
        // clang-format on
      }

      DType theta_min = (nu_norm / scaled_sparsity) - (1 / sigma_min);
      if (theta_min < static_cast<DType>(0)) {
        theta_min = static_cast<DType>(0);
      }
      DType theta_max = (nu_norm / scaled_sparsity) - (1 / sigma_max);

      DType theta;
      do {
        theta = theta_max / 2.0 + theta_min / 2.0;

        // Compute ||α(θ)||₂
        DType alpha_ssq, alpha_scale;
        mshadow_op::nrm2::SetInitValue(alpha_ssq, alpha_scale);
        for (index_t j = 0; j < row_length; j++) {
          // clang-format off
          const DType grad_rescaled = get_grad_rescaled(j);
          index_t data_j = get_data_j(j);
          const DType alpha = -(1 / ((1 / square_root::Map(state_data[data_j] + float_stable_epsilon)) + theta)) *
                               ((lr * grad_rescaled / square_root::Map(state_data[data_j] + float_stable_epsilon)) -
                                  weight_data[data_j]);
          mshadow_op::nrm2::Reduce(alpha_ssq, alpha, alpha_scale);
          // clang-format on
        }
        mshadow_op::nrm2::Finalize(alpha_ssq, alpha_scale);
        DType alpha_norm = alpha_ssq;

        if (alpha_norm > scaled_sparsity) {
          theta_min = theta;
        } else {
          theta_max = theta;
        }
        // rescale bisection_epsilon to reasonable value based on magnitude of
        // theta_max due to exponantial distance of ULP for float
      } while (std::abs(theta_max - theta_min) >
               (bisection_epsilon * std::abs(theta_max)));

      // DEBUG
      if (theta <= 0) {
        std::printf("Proximal adagrad - sigma_min: %f\tsigma_max %f\ttheta: "
                    "%f\tnu_norm %f\tl2 %f\n",
                    (double)sigma_min, (double)sigma_max, (double)theta,
                    (double)nu_norm, (double)scaled_sparsity);
        std::printf("theta < 0 should never have happened. Skipping update.\n");
        return;
      }

      for (index_t j = 0; j < row_length; j++) {
        // clang-format off
        const DType grad_rescaled = get_grad_rescaled(j);
        index_t data_j = get_data_j(j);
        const DType div = lr * grad_rescaled / square_root::Map(state_data[data_j] + float_stable_epsilon);
        out_data[data_j] = (weight_data[data_j] - div -
                            (weight_data[data_j] - div) / (1 + theta));
        // clang-format on
      }
    }
  }
};

/*! \brief kernel for sparse adagrad update with group sparsity regularization
 */
template <typename xpu> struct ProximalGroupAdagradDnsRspKernel {
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void
  Map(int i, const index_t row_length, DType *out_data,
      DType *state_data, DType *weight_data, const IType *grad_idx,
      const DType *grad_data, DType *last_update_data,
      const DType current_update, const DType clip_gradient,
      const DType rescale_grad, const DType l2_regularization_strength,
      const DType lr, const DType eps) {
    using namespace mshadow_op;

    // Helper to obtain index into weight / state arrays
    auto get_data_j = [&i, &grad_idx,
                       &row_length](index_t j) -> index_t {
        return grad_idx[i] * row_length + j;
    };
    // Helper to obtain explicit rescaled and clipped grad
    auto get_grad_rescaled = [&i, &row_length, &grad_data, &rescale_grad,
                              &clip_gradient](index_t j) -> DType {
      index_t grad_j = i * row_length + j;
      DType grad_rescaled = grad_data[grad_j] * rescale_grad;
      if (clip_gradient >= 0.0f) {
        grad_rescaled = clip::Map(grad_rescaled, clip_gradient);
      }
      return grad_rescaled;
    };

    // Compute number of weight updates skipped due to lazy_update
    DType num_skipped = current_update - last_update_data[grad_idx[i]];
    last_update_data[grad_idx[i]] = current_update;
    // Warn in case of erroneous last_update_buffer
    if (num_skipped < 0) {
      num_skipped = 0;
      std::printf("Got invalid last_update in proximal_adagrad_update. "
                  "Using standard Adagrad update.\n");
    }

    // Update history states
    DType grad_ssq = 0;
    for (index_t j = 0; j < row_length; j++) {
      const DType grad_rescaled = get_grad_rescaled(j);
      grad_ssq += grad_rescaled * grad_rescaled;
    }
    state_data[grad_idx[i]] += grad_ssq / row_length;

    DType scaled_sparsity = l2_regularization_strength * num_skipped * lr /
                            square_root::Map(state_data[grad_idx[i]] + eps);
    if (scaled_sparsity <= 0) {
      // Standard Adagrad Update
      for (index_t j = 0; j < row_length; j++) {
        // clang-format off
        const DType grad_rescaled = get_grad_rescaled(j);
        index_t data_j = get_data_j(j);
        const DType div = lr * grad_rescaled / square_root::Map(state_data[grad_idx[i]] + eps);
        out_data[data_j] = weight_data[data_j] - div;
        // clang-format on
      }
    } else {
      // Compute L2 norm of updated parameter using scaled sum of squares
      DType norm, scale;
      mshadow_op::nrm2::SetInitValue(norm, scale);
      for (index_t j = 0; j < row_length; j++) {
        const DType grad_rescaled = get_grad_rescaled(j);
        index_t data_j = get_data_j(j);
        const DType val =
          (weight_data[data_j] -
           lr / std::sqrt(state_data[grad_idx[i]] + eps) *
           grad_rescaled);
        mshadow_op::nrm2::Reduce(norm, val, scale);
      }
      mshadow_op::nrm2::Finalize(norm, scale);

      if (norm <= scaled_sparsity) {
        // Soft threshold weights (proximal map for group lasso)
        for (index_t j = 0; j < row_length; j++) {
          index_t data_j = get_data_j(j);
          out_data[data_j] = 0;
        }
      } else {
        for (index_t j = 0; j < row_length; j++) {
          // clang-format off
          const DType grad_rescaled = get_grad_rescaled(j);
          index_t data_j = get_data_j(j);
          const DType div = lr * grad_rescaled / square_root::Map(state_data[grad_idx[i]] + eps);
          out_data[data_j] = (weight_data[data_j] - div) * (1 - scaled_sparsity / norm);
          // clang-format on
        }
      }
    }
  }
};

/*
 * \brief Adagrad update implementation for dense weight and row_sparse grad.
 *        Both standard update and lazy update are supported.
 */
template <typename xpu>
inline void ProximalAdagradUpdateDnsRspDnsImpl(
    const ProximalAdagradParam &param, const OpContext &ctx,
    const TBlob &weight, const NDArray &grad, const TBlob &state,
    const TBlob &last_update_buffer, const OpReqType &req, TBlob *out) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(grad.storage_type(), kRowSparseStorage);
  // if gradients are zeros, no weights are updated
  if (req == kNullOp) {
    // TODO(leezu) support eager update
    CHECK_EQ(param.lazy_update, true);
    return;
  }
  CHECK_EQ(req, kWriteInplace)
      << "kWriteInplace is expected for sparse proximal_adagrad_update";
  CHECK_GT(weight.shape_.Size(), 0);
  CHECK_GT(state.shape_.Size(), 0);

  MSHADOW_REAL_TYPE_SWITCH(weight.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(grad.aux_type(rowsparse::kIdx), IType, {
      DType *weight_data = weight.dptr<DType>();
      DType *out_data = out->dptr<DType>();
      const IType *grad_idx = grad.aux_data(rowsparse::kIdx).dptr<IType>();
      const DType *grad_val = grad.data().dptr<DType>();
      DType *state_data = state.dptr<DType>();
      DType *last_update_data = last_update_buffer.dptr<DType>();
      const nnvm::dim_t num_grad = grad.aux_shape(rowsparse::kIdx)[0];
      const auto row_length = weight.shape_.ProdShape(1, weight.ndim());

      size_t num_threads;
      if (!param.lazy_update) {
        // When performing eager update, iterate over all rows of the weight
        // array
        num_threads = weight.shape_[0];
      } else if (grad.storage_initialized()) {
        num_threads = num_grad;
      } else { // Lazy update with 0 gradient
        return;
      }

      if (param.groupwise_lr) {
        CHECK_EQ(param.lazy_update, true);
        CHECK_EQ(param.decay_states, false);
        Kernel<ProximalGroupAdagradDnsRspKernel<xpu>, xpu>::Launch(
            s, num_threads, row_length, out_data, state_data, weight_data,
            grad_idx, grad_val, last_update_data,
            static_cast<DType>(param.current_update),
            static_cast<DType>(param.clip_gradient),
            static_cast<DType>(param.rescale_grad),
            static_cast<DType>(param.l2_regularization_strength),
            static_cast<DType>(param.lr),
            static_cast<DType>(param.float_stable_epsilon));
      } else {
        Kernel<ProximalAdagradDnsRspKernel<xpu>, xpu>::Launch(
            s, num_threads, row_length, num_grad, out_data, state_data,
            weight_data, grad_idx, grad_val, last_update_data,
            static_cast<DType>(param.current_update),
            static_cast<DType>(param.clip_gradient),
            static_cast<DType>(param.rescale_grad),
            static_cast<DType>(param.l2_regularization_strength),
            static_cast<DType>(param.lr),
            static_cast<DType>(param.float_stable_epsilon),
            static_cast<DType>(param.bisection_epsilon), param.lazy_update,
            param.decay_states, static_cast<DType>(param.decay_factor),
            param.lazy_decay);
      }
    });
  });
}

/*
 * \brief Proximal adagrad update implementation for row_sparse grad.
 *        Both standard update and lazy update are supported.
 */
template <typename xpu>
inline void ProximalAdagradUpdateRspRspRspImpl(
    const ProximalAdagradParam &param, const OpContext &ctx,
    const NDArray &weight, const NDArray &grad, const NDArray &state,
    const NDArray &last_update_buffer, const OpReqType &req, NDArray *out) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace rowsparse;
  CheckAllRowsPresent(weight, "ProximalAdagradUpdate", "weights");
  Stream<xpu>* s = ctx.get_stream<xpu>();
  // fill history with zero values
  if (!state.storage_initialized()) {
    NDArray state_zeros = state;
    FillDnsZerosRspImpl(s, &state_zeros);
  } else {
    CheckAllRowsPresent(state, "ProximalAdagradUpdate", "states");
  }
  // reuse dns rsp implementation when storage_shape == shape
  TBlob out_blob = out->data();
  ProximalAdagradUpdateDnsRspDnsImpl<xpu>(
      param, ctx, weight.data(), grad, state.data(), last_update_buffer.data(),
      req, &out_blob);
}

template <typename xpu>
inline void ProximalAdagradUpdateEx(const nnvm::NodeAttrs &attrs,
                                    const OpContext &ctx,
                                    const std::vector<NDArray> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &outputs) {
  const ProximalAdagradParam &param =
      nnvm::get<ProximalAdagradParam>(attrs.parsed);
  const auto weight_stype = inputs[0].storage_type();
  const auto grad_stype = inputs[1].storage_type();
  const auto state_stype = inputs[2].storage_type();
  const auto counter_stype = inputs[3].storage_type();
  const auto output_stype = outputs[0].storage_type();

  if (state_stype == weight_stype && output_stype == weight_stype &&
      weight_stype == kRowSparseStorage && grad_stype == kRowSparseStorage &&
      counter_stype == kDefaultStorage) {
    NDArray out = outputs[0];
    ProximalAdagradUpdateRspRspRspImpl<xpu>(param, ctx, inputs[0], inputs[1],
                                            inputs[2], inputs[3], req[0], &out);
  } else if (state_stype == weight_stype && output_stype == weight_stype &&
             weight_stype == kDefaultStorage &&
             grad_stype == kRowSparseStorage &&
             counter_stype == kDefaultStorage) {
    TBlob out_blob = outputs[0].data();
    ProximalAdagradUpdateDnsRspDnsImpl<xpu>(
        param, ctx, inputs[0].data(), inputs[1], inputs[2].data(),
        inputs[3].data(), req[0], &out_blob);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

}  // namespace op
}  // namespace mxnet


#endif  // MXNET_OPERATOR_OPTIMIZER_OP_INL_H_
