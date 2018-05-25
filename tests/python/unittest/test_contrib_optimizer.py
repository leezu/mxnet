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

import itertools

import numpy as np

import mxnet as mx
from mxnet.test_utils import *


# ProximalGroupAdaGrad
class PyProximalGroupAdaGrad(mx.optimizer.Optimizer):
    """The python reference of Proximal Group AdaGrad optimizer.

    Parameters
    ----------
    l2_regularization_strength : float
       Strength of group lasso L2 regularization.
    eps: float, optional
        Small value to avoid division by 0.

    """

    def __init__(self, l2_regularization_strength=0.0, eps=1e-5, **kwargs):
        super(PyProximalGroupAdaGrad, self).__init__(**kwargs)
        self.l2_regularization_strength = l2_regularization_strength
        self.float_stable_eps = eps

    def create_state(self, index, weight):
        assert len(weight.shape) == 2
        history = mx.nd.zeros(
            (weight.shape[0], 1), weight.context, stype=weight.stype)
        return history

    def update(self, index, weight, grad, state):
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        assert wd == 0

        history = state
        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)
        history[:] += mx.nd.mean(mx.nd.square(grad), axis=1, keepdims=True)
        div = lr * grad / mx.nd.sqrt(history + self.float_stable_eps)

        if self.l2_regularization_strength > 0:
            scaled_l2 = lr / mx.nd.sqrt(history + self.float_stable_eps) \
                * self.l2_regularization_strength
            norm = mx.nd.norm(weight - div, ord=2, axis=1, keepdims=True)
            weight[:] = (weight - div) * \
                (1 - scaled_l2 / norm)
            weight[:] *= norm > scaled_l2
        else:
            weight[:] -= div


def test_proximal_group_adagrad():
    mx.random.seed(0)
    opt1 = PyProximalGroupAdaGrad
    opt2 = mx.optimizer.contrib.ProximalGroupAdaGrad
    shape = (3, 4)
    eps_options = [{}, {'eps': 1e-8}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    l2_options = [{
        'l2_regularization_strength': 0.0
    }, {
        'l2_regularization_strength': 0.05
    }]
    for dtype in [np.float32]:
        for options in itertools.product(eps_options, cg_options, rg_options,
                                         l2_options):
            kwarg = dict(wd=0.0)
            for option in options:
                kwarg.update(option)
            compare_optimizer(
                opt1(**kwarg),
                opt2(**kwarg),
                shape,
                dtype,
                compare_states=False)
            if kwarg.get('l2_regularization_strength', 0.0) == 0.0:
                # By design results for PyOp which always performs
                # dense update will differ if
                # l2_regularization_strength > 0
                compare_optimizer(
                    opt1(**kwarg),
                    opt2(**kwarg),
                    shape,
                    dtype,
                    w_stype='row_sparse',
                    g_stype='row_sparse',
                    compare_states=False)
                compare_optimizer(
                    opt1(**kwarg),
                    opt2(**kwarg),
                    shape,
                    dtype,
                    g_stype='row_sparse',
                    compare_states=False)


if __name__ == '__main__':
    import nose
    nose.runmodule()
