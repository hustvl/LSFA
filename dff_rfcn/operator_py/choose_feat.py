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




class ChooseFeatOperator(mx.operator.CustomOp):
    def __init__(self): 
        super(ChooseFeatOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        conv_feat = in_data[0]
        conv_feat_prop = in_data[1]
        eq_flag = in_data[2].asnumpy()[0]
        eq_flag_old = in_data[3].asnumpy()[0]
        if eq_flag == 1 or eq_flag_old == 1:
            self.assign(out_data[0], req[0], conv_feat)
        else:
            self.assign(out_data[0], req[0], conv_feat_prop)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        eq_flag = in_data[2].asnumpy()[0]
        eq_flag_old = in_data[2].asnumpy()[0]
        if eq_flag == 1 or eq_flag_old:
            self.assign(in_grad[0], req[0], out_grad[0])
            self.assign(in_grad[1], req[1], 0)
            self.assign(in_grad[2], req[2], 0)
            self.assign(in_grad[3], req[3], 0)
        else:
            self.assign(in_grad[0], req[0], 0)
            self.assign(in_grad[1], req[1], out_grad[0])
            self.assign(in_grad[2], req[2], 0)
            self.assign(in_grad[3], req[3], 0)

@mx.operator.register('ChooseFeat')
class ChooseFeatProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ChooseFeatProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['conv_feat', 'conv_feat_prop', 'eq_flag', 'eq_flag_old']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):

        return in_shape, \
               [in_shape[1]]

    def create_operator(self, ctx, shapes, dtypes):
        return ChooseFeatOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return [in_data[2], in_data[3], out_grad[0]]
