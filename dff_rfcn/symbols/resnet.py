import mxnet as mx     
import sym_common as sym
import numpy as np
from sym_common import cfg

class ResNet(object):  
    def __init__(self, num_layer):
        self.num_layer = num_layer

    def get_resnet_params(self, num_layer, inv_resolution=32):
        if num_layer == 18:
            units = [2, 2, 2, 2]
        elif num_layer == 26:
            units = [3, 3, 3, 3]
        elif num_layer == 34:
            units = [3, 4, 6, 3]
        elif num_layer == 50:
            units = [3, 4, 6, 3]
        elif num_layer == 101:
            units = [3, 4, 23, 3]
        elif num_layer == 152:
            units = [3, 8, 36, 3]
        elif num_layer == 200:
            units = [3, 24, 36, 3]
        elif num_layer == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(num_layer))
        filter_list = [256, 512, 1024, 2048] if num_layer >= 50 else [64, 128, 256, 512]
        bottle_neck = True if num_layer >= 50 else False
        if inv_resolution == 32:
            inc_dilate_list = [False, False, False, False]
        elif inv_resolution == 16:
            inc_dilate_list = [False, False, False, True]
        elif inv_resolution == 8:
            inc_dilate_list = [False, False, True, True]
        else:
            raise ValueError("no experiments done on resolution {}".format(inv_resolution))
        return units, filter_list, bottle_neck, inc_dilate_list

    def non_local_block(self, data, num_filter, mode, prefix=''):
        mid_num_filter = int(num_filter * 0.5)
        if mode == 'gaussian':
            x_reshape = mx.sym.Reshape(data=data, shape=(0, 0, -1))
            f = mx.sym.batch_dot(x_reshape, x_reshape, transpose_a=True, transpose_b=False)
            f = mx.sym.SoftmaxActivation(data=f, mode='channel')
        elif 'embedded_gaussian' in mode:
            x1 = sym.conv(data=data, name=prefix + 'conv_x1', num_filter=mid_num_filter)
            x2 = sym.conv(data=data, name=prefix + 'conv_x2', num_filter=mid_num_filter)
            if 'compress' in mode:
                x1 = sym.pool(data=x1, name=prefix + 'pool_x1', kernel=3, stride=2, pad=1, pool_type='max')
            x1_reshape = mx.sym.Reshape(data=x1, shape=(0, 0, -1))
            x2_reshape = mx.sym.Reshape(data=x2, shape=(0, 0, -1))
            f = mx.sym.batch_dot(x1_reshape, x2_reshape, transpose_a=True, transpose_b=False)
            f = mx.sym.SoftmaxActivation(data=f, mode='channel')
        else:
            raise ValueError("unknown non-local mode {}".format(mode))
    
        g = sym.conv(data=data, name=prefix + 'conv_g', num_filter=mid_num_filter)
        if mode == 'embedded_gaussian_compress':
            g_pool = sym.pool(data=g, name=prefix + 'pool_g', kernel=3, stride=2, pad=1, pool_type='max')
        else:
            g_pool = g
        g_reshape = mx.sym.Reshape(data=g_pool, shape=(0, 0, -1))
        y = mx.sym.batch_dot(g_reshape, f)
        y = mx.sym.reshape_like(y, g)
        y = sym.conv(data=y, name=prefix + 'conv_y', num_filter=num_filter)
        return data + y

    def resnet_residual_unit(self, data, num_filter, stride, dim_match, num_deformable_group=0,
                             dilate=1, inc_dilate=False, bottle_neck=True, prefix=''):
        dilate_factor = 1
        if inc_dilate:
            assert stride > 1
            dilate_factor = stride
            stride = 1
        if bottle_neck:
            bn1, relu1, conv1 = sym.bnreluconv(data=data, num_filter=int(num_filter * 0.25),
                                               kernel=1, stride=1,
                                               prefix=prefix, suffix='1')
            bn2, relu2, conv2 = sym.bnreluconv(data=conv1, num_filter=int(num_filter * 0.25),
                                               num_deformable_group=num_deformable_group, 
                                               kernel=3, stride=stride, dilate=dilate,
                                               prefix=prefix, suffix='2')
            bn3, relu3, conv = sym.bnreluconv(data=conv2, num_filter=num_filter,
                                              kernel=1, stride=1, 
                                              prefix=prefix, suffix='3')
        else:
            bn1, relu1, conv1 = sym.bnreluconv(data=data, num_filter=num_filter,
                                               num_deformable_group=num_deformable_group,
                                               kernel=3, stride=stride, dilate=dilate,
                                               prefix=prefix, suffix='1')
            bn2, relu2, conv = sym.bnreluconv(data=conv1, num_filter=num_filter,
                                              kernel=3, stride=1, dilate=1,
                                              prefix=prefix, suffix='2')
        if dim_match:
            shortcut = data
        else:
            shortcut = sym.conv(data=relu1, name=prefix + 'sc', num_filter=num_filter,
                                kernel=1, stride=stride, no_bias=True)
        return conv + shortcut, dilate * dilate_factor

    def resnet_v1_residual_unit(self, data, num_filter, stride, dim_match, num_deformable_group=0,
                                dilate=1, inc_dilate=False, bottle_neck=True, prefix=''):
        dilate_factor = 1
        if inc_dilate:
            assert stride > 1
            dilate_factor = stride
            stride = 1
        if bottle_neck:
            conv1, bn1, relu1 = sym.convbnrelu(data=data, num_filter=int(num_filter * 0.25),
                                               kernel=1, stride=1,
                                               prefix=prefix, suffix='1')
            conv2, bn2, relu2 = sym.convbnrelu(data=relu1, num_filter=int(num_filter * 0.25),
                                               num_deformable_group=num_deformable_group,
                                               kernel=3, stride=stride, dilate=dilate,
                                               prefix=prefix, suffix='2')
            conv3, bn = sym.convbn(data=relu2, num_filter=num_filter,
                                   kernel=1, stride=1,
                                   prefix=prefix, suffix='3')
        else:
            conv1, bn1, relu1 = sym.convbnrelu(data=data, num_filter=num_filter,
                                               num_deformable_group=num_deformable_group,
                                               kernel=3, stride=stride, dilate=dilate,
                                               prefix=prefix, suffix='1')
            conv2, bn = sym.convbn(data=relu1, num_filter=num_filter,
                                   kernel=3, stride=1, dilate=1,
                                   prefix=prefix, suffix='2')
        if dim_match:
            shortcut = data
        else:
            _, shortcut = sym.convbn(data=data, num_filter=num_filter,
                                     kernel=1, stride=stride, no_bias=True,
                                     prefix=prefix, suffix='sc')
        res_relu = mx.sym.Activation(data=bn + shortcut, act_type='relu', name=prefix + 'relu')
        return res_relu, dilate * dilate_factor
    
    def get_backbone(self, data, net_type='resnet', need_part=False, inv_resolution=16, use_fp16=False, non_local_mode=None,
                   use_dilate=True, deformable_units=[0, 0, 0, 0], num_deformable_group=[0, 0, 0, 0], param_reuse=False,
                   prefix='', **kwargs):
        cfg.reuse = param_reuse
        
        num_layer = self.num_layer
        units, filter_list, bottle_neck, inc_dilate_list = self.get_resnet_params(num_layer, inv_resolution)
        assert len(units) == 4
        assert len(num_deformable_group) == 4
        assert len(deformable_units) == 4
    
        if net_type == 'resnet':
            residual_unit = self.resnet_residual_unit
            body = sym.bn(data=data, name=prefix + 'bn_data', fix_gamma=True)
        elif net_type == 'resnet_v1':
            residual_unit = self.resnet_v1_residual_unit
            body = data
        elif net_type == 'resnet_v2':
            residual_unit = self.resnet_residual_unit
            body = data
        else:
            raise ValueError("unknown net type {}".format(net_type))
    
        # res1
        _, _, body = sym.convbnrelu(data=body, num_filter=64, kernel=7, stride=2, pad=3, prefix=prefix, suffix='0')
        body = sym.pool(data=body, name=prefix + 'pool0', kernel=3, stride=2, pad=1, pool_type='max')
    
        in_layer_list = []
        dilate = 1
        # res2
        body, dilate = residual_unit(data=body, num_filter=filter_list[0], stride=1,
                                     num_deformable_group=num_deformable_group[0] if 1 >= units[0] - deformable_units[0] + 1 else 0,
                                     dim_match=False, dilate=dilate, inc_dilate=inc_dilate_list[0],
                                     bottle_neck=bottle_neck, prefix=prefix + 'stage1_unit1_')
        dilate = dilate if use_dilate else 1
        for i in range(2, units[0] + 1):
            body, _ = residual_unit(data=body, num_filter=filter_list[0], stride=1,
                                    num_deformable_group=num_deformable_group[0] if i >= units[0] - deformable_units[0] + 1 else 0,
                                    dim_match=True, dilate=dilate, inc_dilate=False,
                                    bottle_neck=bottle_neck, prefix=prefix + 'stage1_unit%d_' % i)
        in_layer_list.append(body)
        if use_fp16: 
            body = mx.sym.cast(data=body, dtype=np.float32)
 
        # res3
        body, dilate = residual_unit(data=body, num_filter=filter_list[1], stride=2,
                                     num_deformable_group=num_deformable_group[1] if 1 >= units[1] - deformable_units[1] + 1 else 0,
                                     dim_match=False, dilate=dilate, inc_dilate=inc_dilate_list[1],
                                     bottle_neck=bottle_neck, prefix=prefix + 'stage2_unit1_')
        if use_fp16:
            body = mx.sym.cast(data=body, dtype=np.float16)
        dilate = dilate if use_dilate else 1
        for i in range(2, units[1] + 1):
            body, _ = residual_unit(data=body, num_filter=filter_list[1], stride=1,
                                    num_deformable_group=num_deformable_group[1] if i >= units[1] - deformable_units[1] + 1 else 0,
                                    dim_match=True, dilate=dilate, inc_dilate=False,
                                    bottle_neck=bottle_neck, prefix=prefix + 'stage2_unit%d_' % i)
        in_layer_list.append(body)
    
        # res4
        if use_fp16:
            body = mx.sym.cast(data=body, dtype=np.float32)
        body, dilate = residual_unit(data=body, num_filter=filter_list[2], stride=2,
                                     num_deformable_group=num_deformable_group[2] if 1 >= units[2] - deformable_units[2] + 1 else 0,
                                     dim_match=False, dilate=dilate, inc_dilate=inc_dilate_list[2],
                                     bottle_neck=bottle_neck, prefix=prefix + 'stage3_unit1_')
        if use_fp16:
            body = mx.sym.cast(data=body, dtype=np.float16)
        dilate = dilate if use_dilate else 1
        for i in range(2, units[2] + 1):
            body, _ = residual_unit(data=body, num_filter=filter_list[2], stride=1,
                                    num_deformable_group=num_deformable_group[2] if i >= units[2] - deformable_units[2] + 1 else 0,
                                    dim_match=True, dilate=dilate, inc_dilate=False,
                                    bottle_neck=bottle_neck, prefix=prefix + 'stage3_unit%d_' % i)
            if i == units[2] - 1 and non_local_mode is not None:
                body = non_local_block(body, num_filter=filter_list[2], mode=non_local_mode)
        in_layer_list.append(body)
    
        # res5
        if use_fp16:
            body = mx.sym.cast(data=body, dtype=np.float32)
        body, dilate = residual_unit(data=body, num_filter=filter_list[3], stride=2,
                                     num_deformable_group=num_deformable_group[3] if 1 >= units[3] - deformable_units[3] + 1 else 0,
                                     dim_match=False, dilate=dilate, inc_dilate=inc_dilate_list[3],
                                     bottle_neck=bottle_neck, prefix=prefix + 'stage4_unit1_')
        dilate = dilate if use_dilate else 1
        if use_fp16:
            body = mx.sym.cast(data=body, dtype=np.float16)
        for i in range(2, units[3] + 1):
            body, _ = residual_unit(data=body, num_filter=filter_list[3], stride=1,
                                    num_deformable_group=num_deformable_group[3] if i >= units[3] - deformable_units[3] + 1 else 0,
                                    dim_match=True, dilate=dilate, inc_dilate=False,
                                    bottle_neck=bottle_neck, prefix=prefix + 'stage4_unit%d_' % i)
        in_layer_list.append(body)
    
        if net_type == 'resnet' or net_type == 'resnet_v2':
            bn1 = sym.bn(data=body, name=prefix + 'bn1', fix_gamma=False)
            relu1 = sym.relu(data=bn1, name=prefix + 'relu1')
            in_layer_list.append(relu1)
        if not need_part:
            return in_layer_list[-1]
        else:
            return in_layer_list

