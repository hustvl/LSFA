# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool




class ChooseOldKeyFeatOperator(mx.operator.CustomOp):
    def __init__(self):
        super(ChooseOldKeyFeatOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        feat_key_old = in_data[0]
        feat_key = in_data[1]
        _, c, h, w = feat_key_old.shape
        if c == 1024 and h == 1 and w == 1:
            self.assign(out_data[0], req[0], feat_key)
            self.assign(out_data[1], req[1], 1)
        else:
            self.assign(out_data[0], req[0], feat_key_old)
            self.assign(out_data[1], req[1], 0)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)



@mx.operator.register('ChooseOldKeyFeat')
class ChooseOldKeyFeatProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ChooseOldKeyFeatProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['feat_key_old', 'feat_key']

    def list_outputs(self):
        return ['choose_old_feat', 'is_first_frame']

    def infer_shape(self, in_shape):

        return in_shape, \
               [in_shape[1], (1,)]

    def create_operator(self, ctx, shapes, dtypes):
        return ChooseOldKeyFeatOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
