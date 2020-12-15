# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong, Xizhou Zhu
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.choose_old_key_feat import *
from operator_py.choose_feat import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *
from operator_py.rpn_inv_normalize import *
from operator_py.tile_as import *
from symbols.resnet import *
from symbols.mobilenetv2 import *
from symbols.mobilenetv2_hobot import *

class resnet_v1_101_flownet_rfcn(Symbol):

    def __init__(self, cfg):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3) # use for 101
        self.filter_list = [256, 512, 1024, 2048]
        if cfg.network.nettype == 'resnet':
            self.net = ResNet(cfg.network.num_layer)
        elif cfg.network.nettype == 'mobilenet':
            self.net = MobileNetV2() 
        elif cfg.network.nettype == 'mobilenet_hobot':
            self.net = MobileNetV2Hobot() 
        else:
            raise RuntimeError("unknow nettype: %s", cfg.network.nettype)
        self.share_params = {}


    def get_resnet_v1(self, data, cfg, param_reuse=False):
        deformable_units = [0, 1, 1, 3] if cfg.network.add_dcn else [0, 0, 0, 0]
        num_deformable_group = [0, 4, 4, 4] if cfg.network.add_dcn else [0, 0, 0, 0]
        out_feat = self.net.get_backbone(data, deformable_units=deformable_units, num_deformable_group=num_deformable_group, param_reuse=param_reuse) 
 
        if 'feat_conv_3x3_weight' not in self.share_params:
            self.share_params['feat_conv_3x3_weight'] = mx.sym.Variable("feat_conv_3x3_weight", lr_mult='1.') 
            self.share_params['feat_conv_3x3_bias']   = mx.sym.Variable("feat_conv_3x3_bias", lr_mult='2.')
        feat_conv_3x3 = mx.sym.Convolution(
            data=out_feat, weight=self.share_params['feat_conv_3x3_weight'], bias=self.share_params['feat_conv_3x3_bias'], kernel=(3, 3), pad=(6, 6), dilate=(6, 6), num_filter=1024, name="feat_conv_3x3")
        feat_conv_3x3_relu = mx.sym.Activation(data=feat_conv_3x3, act_type="relu", name="feat_conv_3x3_relu")
        return feat_conv_3x3_relu

    def res_diff_ada(self, res_diff, cfg):
        num_conv = cfg.network.rnet_num_conv
        assert num_conv >= 0
        res_diff_conv_feat = res_diff
        if cfg.network.res_diff_bn:
            res_diff_conv_feat = mx.symbol.BatchNorm(name='res_diff_bn', data=res_diff_conv_feat,  eps=2e-5, use_global_stats=False, fix_gamma=False)
        for i in range(num_conv):
            res_diff_conv_feat = mx.sym.Convolution(data=res_diff_conv_feat, name='rnet_conv%d'%i, num_filter=256, kernel=(3, 3), pad=(1, 1), no_bias=False)
            res_diff_conv_feat = mx.sym.Activation(data=res_diff_conv_feat, act_type='relu')
        res_diff_conv_feat =  mx.symbol.Convolution(name='rnet_conv%d'%(num_conv), data=res_diff_conv_feat, num_filter=1024, kernel=(1, 1), stride=(1,1), no_bias=False)
        return res_diff_conv_feat

    def fuse_ada(self, fuse_conv_feat, cfg):
        fnet_type = cfg.network.fnet_type
        if 'conv' in fnet_type:
            num_conv = int(fnet_type.split('#')[1])
            for i in range(num_conv):
                fuse_conv_feat = mx.sym.Convolution(data=fuse_conv_feat, name='fnet_conv%d'%i, num_filter=1024, kernel=(3, 3), pad=(1, 1), no_bias=False)
                fuse_conv_feat = mx.sym.Activation(data=fuse_conv_feat, act_type='relu')
        elif 'res' in fnet_type:
            input_data = fuse_conv_feat
            #1*1
            fuse_conv_feat = mx.sym.Convolution(data=fuse_conv_feat, name='fnet_conv0', num_filter=256, kernel=(1, 1), no_bias=False)
            fuse_conv_feat = mx.sym.Activation(data=fuse_conv_feat, act_type='relu')
            #3*3
            fuse_conv_feat = mx.sym.Convolution(data=fuse_conv_feat, name='fnet_conv1', num_filter=256, kernel=(3, 3), pad=(1, 1), no_bias=False)
            fuse_conv_feat = mx.sym.Activation(data=fuse_conv_feat, act_type='relu')
            #1*1
            fuse_conv_feat = mx.sym.Convolution(data=fuse_conv_feat, name='fnet_conv2', num_filter=1024, kernel=(1, 1), no_bias=False)
            fuse_conv_feat = mx.sym.Activation(data=fuse_conv_feat, act_type='relu')
            #add
            fuse_conv_feat = fuse_conv_feat + input_data
        else:
            raise RuntimeError('unknow fnet type: %s', fnet_type)

        return fuse_conv_feat

    def Nq_net(self, warp_feat, conv_feat):
        concat_feat =mx.sym.Concat(warp_feat, conv_feat, dim=0)
        #Nq_net
        Nq_conv1 = mx.sym.Convolution(data=concat_feat, name='Nq_conv1', num_filter=256, kernel=(3, 3), pad=(1, 1), no_bias=False)
        Nq_conv1_relu = mx.sym.Activation(data=Nq_conv1, act_type='relu')
        Nq_conv2 = mx.sym.Convolution(data=Nq_conv1_relu, name='Nq_conv2', num_filter=16, kernel=(1, 1), no_bias=False)
        Nq_conv2_relu = mx.sym.Activation(data=Nq_conv2, act_type='relu')
        Nq_conv3 = mx.sym.Convolution(data=Nq_conv2_relu, name='Nq_conv3', num_filter=1, kernel=(1, 1), no_bias=False)

        #slice 
        weights = mx.sym.softmax(data=Nq_conv3, axis=0)
        weights = mx.sym.SliceChannel(weights, axis=0, num_outputs=2)
        weight1 = mx.symbol.tile(data=weights[0], reps=(1, 1024, 1, 1))
        weight2 = mx.symbol.tile(data=weights[1], reps=(1, 1024, 1, 1))
        out_feat = weight1 * warp_feat + weight2 * conv_feat
        return out_feat
        
    def compute_weight(self, embed_flow, embed_conv_feat):
        embed_flow_norm = mx.symbol.L2Normalization(data=embed_flow, mode='channel')
        embed_conv_norm = mx.symbol.L2Normalization(data=embed_conv_feat, mode='channel')
        weight = mx.symbol.sum(data=embed_flow_norm * embed_conv_norm, axis=1, keepdims=True)

        return weight
        
    def get_embednet(self, data):
        em_conv1 = mx.symbol.Convolution(name='em_conv1', data=data, num_filter=512, pad=(0, 0),
                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
        em_ReLU1 = mx.symbol.Activation(name='em_ReLU1', data=em_conv1, act_type='relu')

        em_conv2 = mx.symbol.Convolution(name='em_conv2', data=em_ReLU1, num_filter=512, pad=(1, 1), kernel=(3, 3),
                                         stride=(1, 1), no_bias=False)
        em_ReLU2 = mx.symbol.Activation(name='em_ReLU2', data=em_conv2, act_type='relu')

        em_conv3 = mx.symbol.Convolution(name='em_conv3', data=em_ReLU2, num_filter=2048, pad=(0, 0), kernel=(1, 1),
                                         stride=(1, 1), no_bias=False)

        return em_conv3

    def Fgfa_net(self, warp_feat, conv_feat):
        concat_embed_data = mx.symbol.Concat(*[conv_feat, warp_feat], dim=0)
        embed_output = self.get_embednet(concat_embed_data)
        embed_output = mx.sym.SliceChannel(embed_output, axis=0, num_outputs=2)
        
        unnormalize_weight1 = self.compute_weight(embed_output[1], embed_output[0])
        unnormalize_weight2 = self.compute_weight(embed_output[0], embed_output[0])
        unnormalize_weights = mx.symbol.Concat(unnormalize_weight1, unnormalize_weight2, dim=0)

        weights = mx.symbol.softmax(data=unnormalize_weights, axis=0)
        weights = mx.sym.SliceChannel(weights, axis=0, num_outputs=2)

        # tile the channel dim of weights
        weight1 = mx.symbol.tile(data=weights[0], reps=(1, 1024, 1, 1))
        weight2 = mx.symbol.tile(data=weights[1], reps=(1, 1024, 1, 1))
        out_feat = weight1 * warp_feat + weight2 * conv_feat
        return out_feat

    def get_flownet(self, img_cur, img_ref):
        data = mx.symbol.Concat(img_cur / 255.0, img_ref / 255.0, dim=1)
        resize_data = mx.symbol.Pooling(name='resize_data', data=data , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
        flow_conv1 = mx.symbol.Convolution(name='flow_conv1', data=resize_data , num_filter=64, pad=(3,3), kernel=(7,7), stride=(2,2), no_bias=False)
        ReLU1 = mx.symbol.LeakyReLU(name='ReLU1', data=flow_conv1 , act_type='leaky', slope=0.1)
        conv2 = mx.symbol.Convolution(name='conv2', data=ReLU1 , num_filter=128, pad=(2,2), kernel=(5,5), stride=(2,2), no_bias=False)
        ReLU2 = mx.symbol.LeakyReLU(name='ReLU2', data=conv2 , act_type='leaky', slope=0.1)
        conv3 = mx.symbol.Convolution(name='conv3', data=ReLU2 , num_filter=256, pad=(2,2), kernel=(5,5), stride=(2,2), no_bias=False)
        ReLU3 = mx.symbol.LeakyReLU(name='ReLU3', data=conv3 , act_type='leaky', slope=0.1)
        conv3_1 = mx.symbol.Convolution(name='conv3_1', data=ReLU3 , num_filter=256, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        ReLU4 = mx.symbol.LeakyReLU(name='ReLU4', data=conv3_1 , act_type='leaky', slope=0.1)
        conv4 = mx.symbol.Convolution(name='conv4', data=ReLU4 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=False)
        ReLU5 = mx.symbol.LeakyReLU(name='ReLU5', data=conv4 , act_type='leaky', slope=0.1)
        conv4_1 = mx.symbol.Convolution(name='conv4_1', data=ReLU5 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        ReLU6 = mx.symbol.LeakyReLU(name='ReLU6', data=conv4_1 , act_type='leaky', slope=0.1)
        conv5 = mx.symbol.Convolution(name='conv5', data=ReLU6 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=False)
        ReLU7 = mx.symbol.LeakyReLU(name='ReLU7', data=conv5 , act_type='leaky', slope=0.1)
        conv5_1 = mx.symbol.Convolution(name='conv5_1', data=ReLU7 , num_filter=512, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        ReLU8 = mx.symbol.LeakyReLU(name='ReLU8', data=conv5_1 , act_type='leaky', slope=0.1)
        conv6 = mx.symbol.Convolution(name='conv6', data=ReLU8 , num_filter=1024, pad=(1,1), kernel=(3,3), stride=(2,2), no_bias=False)
        ReLU9 = mx.symbol.LeakyReLU(name='ReLU9', data=conv6 , act_type='leaky', slope=0.1)
        conv6_1 = mx.symbol.Convolution(name='conv6_1', data=ReLU9 , num_filter=1024, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        ReLU10 = mx.symbol.LeakyReLU(name='ReLU10', data=conv6_1 , act_type='leaky', slope=0.1)
        Convolution1 = mx.symbol.Convolution(name='Convolution1', data=ReLU10 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        deconv5 = mx.symbol.Deconvolution(name='deconv5', data=ReLU10 , num_filter=512, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_deconv5 = mx.symbol.Crop(name='crop_deconv5', *[deconv5,ReLU8] , offset=(1,1))
        ReLU11 = mx.symbol.LeakyReLU(name='ReLU11', data=crop_deconv5 , act_type='leaky', slope=0.1)
        upsample_flow6to5 = mx.symbol.Deconvolution(name='upsample_flow6to5', data=Convolution1 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_upsampled_flow6_to_5 = mx.symbol.Crop(name='crop_upsampled_flow6_to_5', *[upsample_flow6to5,ReLU8] , offset=(1,1))
        Concat2 = mx.symbol.Concat(name='Concat2', *[ReLU8,ReLU11,crop_upsampled_flow6_to_5] )
        Convolution2 = mx.symbol.Convolution(name='Convolution2', data=Concat2 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        deconv4 = mx.symbol.Deconvolution(name='deconv4', data=Concat2 , num_filter=256, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_deconv4 = mx.symbol.Crop(name='crop_deconv4', *[deconv4,ReLU6] , offset=(1,1))
        ReLU12 = mx.symbol.LeakyReLU(name='ReLU12', data=crop_deconv4 , act_type='leaky', slope=0.1)
        upsample_flow5to4 = mx.symbol.Deconvolution(name='upsample_flow5to4', data=Convolution2 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_upsampled_flow5_to_4 = mx.symbol.Crop(name='crop_upsampled_flow5_to_4', *[upsample_flow5to4,ReLU6] , offset=(1,1))
        Concat3 = mx.symbol.Concat(name='Concat3', *[ReLU6,ReLU12,crop_upsampled_flow5_to_4] )
        Convolution3 = mx.symbol.Convolution(name='Convolution3', data=Concat3 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        deconv3 = mx.symbol.Deconvolution(name='deconv3', data=Concat3 , num_filter=128, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_deconv3 = mx.symbol.Crop(name='crop_deconv3', *[deconv3,ReLU4] , offset=(1,1))
        ReLU13 = mx.symbol.LeakyReLU(name='ReLU13', data=crop_deconv3 , act_type='leaky', slope=0.1)
        upsample_flow4to3 = mx.symbol.Deconvolution(name='upsample_flow4to3', data=Convolution3 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_upsampled_flow4_to_3 = mx.symbol.Crop(name='crop_upsampled_flow4_to_3', *[upsample_flow4to3,ReLU4] , offset=(1,1))
        Concat4 = mx.symbol.Concat(name='Concat4', *[ReLU4,ReLU13,crop_upsampled_flow4_to_3] )
        Convolution4 = mx.symbol.Convolution(name='Convolution4', data=Concat4 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
        deconv2 = mx.symbol.Deconvolution(name='deconv2', data=Concat4 , num_filter=64, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_deconv2 = mx.symbol.Crop(name='crop_deconv2', *[deconv2,ReLU2] , offset=(1,1))
        ReLU14 = mx.symbol.LeakyReLU(name='ReLU14', data=crop_deconv2 , act_type='leaky', slope=0.1)
        upsample_flow3to2 = mx.symbol.Deconvolution(name='upsample_flow3to2', data=Convolution4 , num_filter=2, pad=(0,0), kernel=(4,4), stride=(2,2), no_bias=False)
        crop_upsampled_flow3_to_2 = mx.symbol.Crop(name='crop_upsampled_flow3_to_2', *[upsample_flow3to2,ReLU2] , offset=(1,1))
        Concat5 = mx.symbol.Concat(name='Concat5', *[ReLU2,ReLU14,crop_upsampled_flow3_to_2] )
        Concat5 = mx.symbol.Pooling(name='resize_concat5', data=Concat5 , pooling_convention='full', pad=(0,0), kernel=(2,2), stride=(2,2), pool_type='avg')
        Convolution5 = mx.symbol.Convolution(name='Convolution5', data=Concat5 , num_filter=2, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)

        Convolution5_scale_bias = mx.sym.Variable(name='Convolution5_scale_bias', lr_mult=0.0)
        Convolution5_scale = mx.symbol.Convolution(name='Convolution5_scale', data=Concat5 , num_filter=1024, pad=(0,0), kernel=(1,1), stride=(1,1),
                                                   bias=Convolution5_scale_bias, no_bias=False)
        return Convolution5 * 2.5, Convolution5_scale
   
    def fuse_small_net(self, warp_conv_feat, cur_img, cfg, is_train=True):
        small_net_stride = cfg.network.small_net_stride
        small_net_fuse_type = cfg.network.small_net_fuse_type
        small_net_bn_before_fuse = cfg.network.small_net_bn_before_fuse
        small_net_scale_before_fuse = cfg.network.small_net_scale_before_fuse
        num_filters = -1
        if small_net_stride == 4:
            cur_img = mx.symbol.Pooling(name='resize_data', data=cur_img , pooling_convention='full', pad=(0,0), kernel=(4, 4), stride=(4, 4), pool_type='avg')
            small_net_feat = self.net.get_backbone(cur_img, need_part=True, prefix='small_net_')
            cur_feat = small_net_feat[0]
            num_filters = 256 #res18: 64, res101: 256
        elif small_net_stride == 8:
            cur_img = mx.symbol.Pooling(name='resize_data', data=cur_img , pooling_convention='full', pad=(0,0), kernel=(2, 2), stride=(2, 2), pool_type='avg')
            small_net_feat = self.net.get_backbone(cur_img, need_part=True, prefix='small_net_')
            cur_feat = small_net_feat[1]
            num_filters = 512 #res18: 128, res101: 512
        else:
            raise RuntimeError("unknow small_net_tride: %d", small_net_stride)
        if small_net_scale_before_fuse:
            cur_feat =  mx.symbol.Convolution(name='cur_scale', data=cur_feat, num_filter=num_filters, kernel=(1, 1), stride=(1,1), no_bias=False)
            #warp_conv_feat =  mx.symbol.Convolution(name='warp_scale', data=warp_conv_feat, num_filter=1024, kernel=(1, 1), stride=(1,1), no_bias=False)
        if small_net_fuse_type == 'add':
            cur_feat = mx.sym.Convolution(data=cur_feat, kernel=(3, 3), pad=(1, 1), num_filter=1024, name="fuse_reduce_add")
            if small_net_bn_before_fuse:
                use_global_stats = False if is_train else True
                cur_feat = mx.symbol.BatchNorm(name='cur_feat_bn', data=cur_feat,  eps=2e-5, use_global_stats=use_global_stats, fix_gamma=False)
                warp_conv_feat = mx.symbol.BatchNorm(name='warp_conv_feat_bn', data=warp_conv_feat,  eps=2e-5, use_global_stats=use_global_stats, fix_gamma=False)
            out_feat = cur_feat + warp_conv_feat
        elif small_net_fuse_type == 'addv2':
            cur_feat = mx.sym.Convolution(data=cur_feat, kernel=(3, 3), pad=(1, 1), num_filter=num_filters, name="fuse_reduce_add_conv1")
            cur_feat = mx.sym.Activation(data=cur_feat, act_type='relu')
            cur_feat = mx.sym.Convolution(data=cur_feat, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fuse_reduce_add_conv2")
            if small_net_bn_before_fuse:
                cur_feat = mx.symbol.BatchNorm(name='cur_feat_bn', data=cur_feat,  eps=2e-5, use_global_stats=False, fix_gamma=False)
                warp_conv_feat = mx.symbol.BatchNorm(name='warp_conv_feat_bn', data=warp_conv_feat,  eps=2e-5, use_global_stats=False, fix_gamma=False)
            out_feat = cur_feat + warp_conv_feat
        elif small_net_fuse_type == 'concat':
            cur_feat = mx.sym.Convolution(data=cur_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="fuse_reduce_c1")
            warp_conv_feat = mx.sym.Convolution(data=warp_conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="fuse_reduce_c2")
            out_feat = mx.sym.Concat(*[warp_conv_feat, cur_feat], dim=1)
            out_feat = mx.sym.Convolution(data=out_feat, kernel=(3, 3), pad=(1, 1), num_filter=1024, name="fuse_reduce")
        elif small_net_fuse_type == 'concatv1':
            cur_feat = mx.sym.Convolution(data=cur_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="fuse_reduce_c1")
            warp_conv_feat = mx.sym.Convolution(data=warp_conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="fuse_reduce_c2")
            cat_feat = mx.sym.Concat(*[warp_conv_feat, cur_feat], dim=1)
            cat_feat = mx.sym.Convolution(data=cat_feat, kernel=(3, 3), pad=(1, 1), num_filter=1024, name="fuse_reduce")
            cat_feat = mx.sym.Activation(data=cat_feat, act_type='relu')
            s_feat = mx.symbol.Pooling(data=cat_feat, name = "global_pool", global_pool=True, pool_type = 'avg')
            s_feat = mx.sym.Convolution(data=s_feat, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="s_feat_conv1")
            s_feat = mx.sym.Activation(data=s_feat, act_type='relu')
            s_feat = mx.sym.Convolution(data=s_feat, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="s_feat_conv2")
            s_feat = mx.sym.Activation(data=s_feat, act_type='sigmoid')
            s_feat = mx.sym.broadcast_mul(cat_feat, s_feat)
            out_feat = s_feat + cat_feat
        elif small_net_fuse_type == 'concatv2':
            cur_feat = mx.sym.Convolution(data=cur_feat, kernel=(3, 3), pad=(1, 1), num_filter=1024, name="fuse_reduce_c1")
            cat_feat = mx.sym.Concat(*[warp_conv_feat, cur_feat], dim=1)
            s_feat = mx.symbol.Pooling(data=cat_feat, name = "global_pool", pool_type = 'avg', global_pool=True)
            s_feat = mx.sym.Convolution(data=s_feat, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="s_feat_conv1")
            s_feat = mx.sym.Activation(data=s_feat, act_type='relu')
            s_feat = mx.sym.Convolution(data=s_feat, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="s_feat_conv2")
            s_feat = mx.sym.Activation(data=s_feat, act_type='sigmoid')
            s_feat = mx.sym.broadcast_mul(cur_feat, s_feat)
            out_feat = s_feat + warp_conv_feat 
        else:
            raise RuntimeError("unknow small_net_fuse_type: %s", small_net_fuse_type)
        return out_feat

    def get_train_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data = mx.sym.Variable(name="data")
        data_ref = mx.sym.Variable(name="data_ref")
        data_ref_old = mx.sym.Variable(name="data_ref_old")
        if not cfg.network.add_small_net:
            data_ref = data_ref + 0 * data
        eq_flag = mx.sym.Variable(name="eq_flag")
        eq_flag_old = mx.sym.Variable(name="eq_flag_old")
        im_info = mx.sym.Variable(name="im_info")
        gt_boxes = mx.sym.Variable(name="gt_boxes")
        motion_vector = mx.sym.Variable(name='motion_vector')
        res_diff = mx.sym.Variable(name='res_diff')
        rpn_label = mx.sym.Variable(name='label')
        rpn_bbox_target = mx.sym.Variable(name='bbox_target')
        rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
 
        #key frame propagation
        concat_data = mx.sym.Concat(data_ref, data_ref_old, dim=0)
        concat_data_feats = self.get_resnet_v1(concat_data, cfg)
        concat_data_feats = mx.sym.SliceChannel(concat_data_feats, axis=0, num_outputs=2)
        conv_feat = concat_data_feats[0]
        conv_feat_old = concat_data_feats[1]
        flow, scale_map = self.get_flownet(data_ref, data_ref_old)
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp', name='flow_grid')
        flow_warp_conv_feat = mx.sym.BilinearSampler(data=conv_feat_old, grid=flow_grid, name='flow_warping_feat')
        flow_warp_conv_feat = flow_warp_conv_feat * scale_map
        
        if cfg.network.add_Nq_net:
            conv_feat_prop = self.Nq_net(flow_warp_conv_feat, conv_feat)
        elif cfg.network.add_Fgfa_net:
            conv_feat_prop = self.Fgfa_net(flow_warp_conv_feat, conv_feat)
        else:
            conv_feat_prop = 0.5 * (flow_warp_conv_feat + conv_feat)
        conv_feat = mx.sym.Custom(conv_feat=conv_feat, conv_feat_prop=conv_feat_prop, eq_flag=eq_flag, eq_flag_old=eq_flag_old, name='choose_feat', op_type='ChooseFeat')
 
        # shared convolutional layers
        flow = motion_vector
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp', name='motion_grid')
        warp_conv_feat = mx.sym.BilinearSampler(data=conv_feat, grid=flow_grid, name='motion_warping_feat')
        #res diff net
        res_diff = self.res_diff_ada(res_diff, cfg)
        #fusing type
        if cfg.network.fuse_type == 'add':
            warp_conv_feat = warp_conv_feat + res_diff
        elif cfg.network.fuse_type == 'concat':
            warp_conv_feat = mx.sym.Concat(*[warp_conv_feat, res_diff], dim=1)
            warp_conv_feat = mx.sym.Convolution(data=warp_conv_feat, name='fuse_downsample', num_filter=1024, kernel=(1,1),no_bias=False)
        else:
            raise RuntimeError('unknow fuse type: %s', cfg.network.fuse_type)
        # adaptive net for fusing feature
        if 'conv' in cfg.network.fnet_type:
            warp_conv_feat = self.fuse_ada(warp_conv_feat, cfg)
        if cfg.network.add_small_net:
            warp_conv_feat = self.fuse_small_net(warp_conv_feat, data, cfg, is_train=True)    
        #
        select_conv_feat = mx.sym.take(mx.sym.Concat(*[warp_conv_feat, conv_feat], dim=0), eq_flag)

        conv_feats = mx.sym.SliceChannel(select_conv_feat, axis=1, num_outputs=2)

        # RPN layers
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        # prepare rpn data
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

        # classification
        rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                               normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
        # bounding box regression
        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=1.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)
        else:
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
        rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

        # ROI proposal
        rpn_cls_act = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
        rpn_cls_act_reshape = mx.sym.Reshape(
            data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')

        if cfg.TRAIN.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)

         # ROI proposal target
        gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
        rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)

        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        if True: #not cfg.network.add_dcn:
            psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7,pooled_size=7,
                                                   output_dim=num_classes, spatial_scale=0.0625)
            psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7,pooled_size=7,
                                                   output_dim=8, spatial_scale=0.0625)
        else:
            rfcn_cls_offset_t = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=2 * 7 * 7 * num_classes, name="rfcn_cls_offset_t")
            rfcn_bbox_offset_t = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7 * 7 * 2, name="rfcn_bbox_offset_t")
            rfcn_cls_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_cls_offset', data=rfcn_cls_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                        sample_per_part=4, no_trans=True, part_size=7, output_dim=2 * num_classes, spatial_scale=0.0625)
            rfcn_bbox_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_bbox_offset', data=rfcn_bbox_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                        sample_per_part=4, no_trans=True, part_size=7, output_dim=2, spatial_scale=0.0625)
            psroipooled_cls_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, trans=rfcn_cls_offset,
                                                                            group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1, output_dim=num_classes, spatial_scale=0.0625, part_size=7)    
            psroipooled_loc_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, trans=rfcn_bbox_offset,
                                                                            group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1, output_dim=8, spatial_scale=0.0625, part_size=7)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))


        # classification
        if cfg.TRAIN.ENABLE_OHEM:
            print 'use ohem!'
            labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                           num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                           cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                           bbox_targets=bbox_target, bbox_weights=bbox_weight)
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid', use_ignore=True, ignore_label=-1)
            bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
            rcnn_label = labels_ohem
        else:
            cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
            bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
            bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS)
            rcnn_label = label

        # reshape output
        rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
        bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_loss_reshape')
        
        group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)])
        self.sym = group
        return group

    def get_key_test_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")
        data_key = mx.sym.Variable(name="data_key")
        data_key_old = mx.sym.Variable(name='data_key_old')
        feat_key = mx.sym.Variable(name="feat_key")
        feat_key_old = mx.sym.Variable(name="feat_key_old")
        motion_vector = mx.sym.Variable(name='motion_vector')
        res_diff = mx.sym.Variable(name='res_diff')
        # shared convolutional layers
        conv_feat = self.get_resnet_v1(data, cfg)
        feat_key_old, is_first_frame = mx.sym.Custom(feat_key_old=feat_key_old, feat_key=conv_feat, name='choose_old_key_feat', op_type='ChooseOldKeyFeat')
       
        flow, scale_map = self.get_flownet(data, data_key_old)
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp', name='flow_grid')
        flow_warp_conv_feat = mx.sym.BilinearSampler(data=feat_key_old, grid=flow_grid, name='flow_warping_feat')
        flow_warp_conv_feat = flow_warp_conv_feat * scale_map
        if cfg.network.add_Nq_net:
            conv_feat_prop = self.Nq_net(flow_warp_conv_feat, conv_feat)
        elif cfg.network.add_Fgfa_net:
            conv_feat_prop = self.Fgfa_net(flow_warp_conv_feat, conv_feat)
        else:
            conv_feat_prop = 0.5 * (flow_warp_conv_feat + conv_feat)
        conv_feat = mx.sym.Custom(conv_feat=conv_feat, conv_feat_prop=conv_feat_prop, eq_flag=is_first_frame, eq_flag_old=is_first_frame, name='choose_feat', op_type='ChooseFeat')
        
        conv_feats = mx.sym.SliceChannel(conv_feat, axis=1, num_outputs=2)

        # RPN
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)

        # ROI Proposal
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
        rpn_cls_prob_reshape = mx.sym.Reshape(
            data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
        if cfg.TEST.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        if True: #not cfg.network.add_dcn:
            psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7,pooled_size=7,
                                                   output_dim=num_classes, spatial_scale=0.0625)
            psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7,pooled_size=7,
                                                   output_dim=8, spatial_scale=0.0625)
        else:
            rfcn_cls_offset_t = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=2 * 7 * 7 * num_classes, name="rfcn_cls_offset_t")
            rfcn_bbox_offset_t = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7 * 7 * 2, name="rfcn_bbox_offset_t")
            rfcn_cls_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_cls_offset', data=rfcn_cls_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                        sample_per_part=4, no_trans=True, part_size=7, output_dim=2 * num_classes, spatial_scale=0.0625)
            rfcn_bbox_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_bbox_offset', data=rfcn_bbox_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                        sample_per_part=4, no_trans=True, part_size=7, output_dim=2, spatial_scale=0.0625)
            psroipooled_cls_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, trans=rfcn_cls_offset,
                                                                            group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1, output_dim=num_classes, spatial_scale=0.0625, part_size=7)    
            psroipooled_loc_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, trans=rfcn_bbox_offset,
                                                                            group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1, output_dim=8, spatial_scale=0.0625, part_size=7)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))

        # classification
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        # bounding box regression
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        # reshape output
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')

        # group output
        group = mx.sym.Group([data_key, motion_vector, res_diff, feat_key, conv_feat, rois, cls_prob, bbox_pred])
        self.sym = group
        return group

    def get_cur_test_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data_cur = mx.sym.Variable(name="data")
        im_info = mx.sym.Variable(name="im_info")
        data_key = mx.sym.Variable(name="data_key")
        data_key_old = mx.sym.Variable(name="data_key_old")
        conv_feat = mx.sym.Variable(name="feat_key")
        feat_key_old = mx.sym.Variable(name="feat_key_old")
        motion_vector = mx.sym.Variable(name='motion_vector')
        res_diff = mx.sym.Variable(name='res_diff')
        
        # shared convolutional layers
        flow = motion_vector
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp', name='flow_grid')
        conv_feat = mx.sym.BilinearSampler(data=conv_feat, grid=flow_grid, name='warping_feat')
        res_diff = self.res_diff_ada(res_diff, cfg)
        #fusing type
        if cfg.network.fuse_type == 'add':
            conv_feat = conv_feat + res_diff
        elif cfg.network.fuse_type == 'concat':
            conv_feat = mx.sym.Concat(*[conv_feat, res_diff], dim=1)
            conv_feat = mx.sym.Convolution(data=conv_feat, name='fuse_downsample', num_filter=1024, kernel=(1,1),no_bias=False)
        else:
            raise RuntimeError('unknow fuse type: %s', cfg.network.fuse_type)
        # adaptive net for fusing feature
        if 'conv' in cfg.network.fnet_type:
            conv_feat = self.fuse_ada(conv_feat, cfg)
        if cfg.network.add_small_net:
            conv_feat = self.fuse_small_net(conv_feat, data_cur, cfg, is_train=False)    
        conv_feats = mx.sym.SliceChannel(conv_feat, axis=1, num_outputs=2)

        # RPN
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)

        # ROI Proposal
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
        rpn_cls_prob_reshape = mx.sym.Reshape(
            data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
        if cfg.TEST.CXX_PROPOSAL:
            rois = mx.contrib.sym.Proposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            rois = mx.sym.Custom(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        if True: #not cfg.network.add_dcn:
            psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7,pooled_size=7,
                                                   output_dim=num_classes, spatial_scale=0.0625)
            psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7,pooled_size=7,
                                                   output_dim=8, spatial_scale=0.0625)
        else:
            rfcn_cls_offset_t = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=2 * 7 * 7 * num_classes, name="rfcn_cls_offset_t")
            rfcn_bbox_offset_t = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7 * 7 * 2, name="rfcn_bbox_offset_t")
            rfcn_cls_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_cls_offset', data=rfcn_cls_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                        sample_per_part=4, no_trans=True, part_size=7, output_dim=2 * num_classes, spatial_scale=0.0625)
            rfcn_bbox_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_bbox_offset', data=rfcn_bbox_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                        sample_per_part=4, no_trans=True, part_size=7, output_dim=2, spatial_scale=0.0625)
            psroipooled_cls_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, trans=rfcn_cls_offset,
                                                                            group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1, output_dim=num_classes, spatial_scale=0.0625, part_size=7)    
            psroipooled_loc_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, trans=rfcn_bbox_offset,
                                                                            group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1, output_dim=8, spatial_scale=0.0625, part_size=7)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))

        # classification
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        # bounding box regression
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        # reshape output
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')

        # group output
        group = mx.sym.Group([data_cur, data_key, data_key_old, feat_key_old, rois, cls_prob, bbox_pred])
        self.sym = group
        return group

    def get_batch_test_symbol(self, cfg):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        data_key = mx.sym.Variable(name="data_key")
        data_other = mx.sym.Variable(name="data_other")
        im_info = mx.sym.Variable(name="im_info")

        # shared convolutional layers
        conv_feat_key = self.get_resnet_v1(data_key, cfg)

        data_key_tiled = mx.sym.Custom(data_content=data_key, data_shape=data_other, op_type='tile_as')
        conv_feat_key_tiled = mx.sym.Custom(data_content=conv_feat_key, data_shape=data_other, op_type='tile_as')
        flow, scale_map = self.get_flownet(data_other, data_key_tiled)
        flow_grid = mx.sym.GridGenerator(data=flow, transform_type='warp', name='flow_grid')
        conv_feat_other = mx.sym.BilinearSampler(data=conv_feat_key_tiled, grid=flow_grid, name='warping_feat')
        conv_feat_other = conv_feat_other * scale_map

        conv_feat = mx.symbol.Concat(conv_feat_key, conv_feat_other, dim=0)

        conv_feats = mx.sym.SliceChannel(conv_feat, axis=1, num_outputs=2)

        # RPN
        rpn_feat = conv_feats[0]
        rpn_cls_score = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
        rpn_bbox_pred = mx.sym.Convolution(
            data=rpn_feat, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

        if cfg.network.NORMALIZE_RPN:
            rpn_bbox_pred = mx.sym.Custom(
                bbox_pred=rpn_bbox_pred, op_type='rpn_inv_normalize', num_anchors=num_anchors,
                bbox_mean=cfg.network.ANCHOR_MEANS, bbox_std=cfg.network.ANCHOR_STDS)

        # ROI Proposal
        rpn_cls_score_reshape = mx.sym.Reshape(
            data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
        rpn_cls_prob = mx.sym.SoftmaxActivation(
            data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
        rpn_cls_prob_reshape = mx.sym.Reshape(
            data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
        if cfg.TEST.CXX_PROPOSAL:
            rois = mx.contrib.sym.MultiProposal(
                cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                ratios=tuple(cfg.network.ANCHOR_RATIOS),
                rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
        else:
            NotImplemented

        # res5
        rfcn_feat = conv_feats[1]
        rfcn_cls = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*num_classes, name="rfcn_cls")
        rfcn_bbox = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7*7*4*num_reg_classes, name="rfcn_bbox")
        if True: #not cfg.network.add_dcn:
            psroipooled_cls_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, group_size=7,pooled_size=7,
                                                   output_dim=num_classes, spatial_scale=0.0625)
            psroipooled_loc_rois = mx.contrib.sym.PSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, group_size=7,pooled_size=7,
                                                   output_dim=8, spatial_scale=0.0625)
        else:
            rfcn_cls_offset_t = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=2 * 7 * 7 * num_classes, name="rfcn_cls_offset_t")
            rfcn_bbox_offset_t = mx.sym.Convolution(data=rfcn_feat, kernel=(1, 1), num_filter=7 * 7 * 2, name="rfcn_bbox_offset_t")
            rfcn_cls_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_cls_offset', data=rfcn_cls_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                        sample_per_part=4, no_trans=True, part_size=7, output_dim=2 * num_classes, spatial_scale=0.0625)
            rfcn_bbox_offset = mx.contrib.sym.DeformablePSROIPooling(name='rfcn_bbox_offset', data=rfcn_bbox_offset_t, rois=rois, group_size=7, pooled_size=7,
                                                                        sample_per_part=4, no_trans=True, part_size=7, output_dim=2, spatial_scale=0.0625)
            psroipooled_cls_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_cls_rois', data=rfcn_cls, rois=rois, trans=rfcn_cls_offset,
                                                                            group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1, output_dim=num_classes, spatial_scale=0.0625, part_size=7)    
            psroipooled_loc_rois = mx.contrib.sym.DeformablePSROIPooling(name='psroipooled_loc_rois', data=rfcn_bbox, rois=rois, trans=rfcn_bbox_offset,
                                                                            group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1, output_dim=8, spatial_scale=0.0625, part_size=7)
        cls_score = mx.sym.Pooling(name='ave_cls_scors_rois', data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        bbox_pred = mx.sym.Pooling(name='ave_bbox_pred_rois', data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))

        # classification
        cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
        cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
        # bounding box regression
        bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        # reshape output
        cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
        bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')

        # group output
        group = mx.sym.Group([rois, cls_prob, bbox_pred])
        self.sym = group
        return group

    def init_weight(self, cfg, arg_params, aux_params):
        if cfg.network.add_small_net:
            for sym_param in self.sym.list_arguments():
                if 'small_net_' in sym_param and sym_param not in arg_params.keys():
                    arg_params[sym_param] = arg_params[sym_param.replace('small_net_', '')]
            for sym_param in self.sym.list_auxiliary_states():
                if 'small_net_' in sym_param and sym_param not in aux_params.keys():
                    aux_params[sym_param] = aux_params[sym_param.replace('small_net_', '')]
            if cfg.network.small_net_fuse_type == 'add':
                arg_params['fuse_reduce_add_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fuse_reduce_add_weight'])
                arg_params['fuse_reduce_add_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fuse_reduce_add_bias'])
            if cfg.network.small_net_fuse_type == 'addv2':
                arg_params['fuse_reduce_add_conv1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fuse_reduce_add_conv1_weight'])
                arg_params['fuse_reduce_add_conv1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fuse_reduce_add_conv1_bias'])
                arg_params['fuse_reduce_add_conv2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fuse_reduce_add_conv2_weight'])
                arg_params['fuse_reduce_add_conv2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fuse_reduce_add_conv2_bias'])
            if cfg.network.small_net_fuse_type == 'concat':
                arg_params['fuse_reduce_c1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fuse_reduce_c1_weight'])
                arg_params['fuse_reduce_c1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fuse_reduce_c1_bias'])
                arg_params['fuse_reduce_c2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fuse_reduce_c2_weight'])
                arg_params['fuse_reduce_c2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fuse_reduce_c2_bias'])
                arg_params['fuse_reduce_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fuse_reduce_weight'])
                arg_params['fuse_reduce_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fuse_reduce_bias'])
            if cfg.network.small_net_fuse_type == 'concatv1':
                arg_params['fuse_reduce_c1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fuse_reduce_c1_weight'])
                arg_params['fuse_reduce_c1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fuse_reduce_c1_bias'])
                arg_params['fuse_reduce_c2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fuse_reduce_c2_weight'])
                arg_params['fuse_reduce_c2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fuse_reduce_c2_bias'])
                arg_params['fuse_reduce_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fuse_reduce_weight'])
                arg_params['fuse_reduce_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fuse_reduce_bias'])
                arg_params['s_feat_conv1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['s_feat_conv1_weight'])
                arg_params['s_feat_conv1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['s_feat_conv1_bias'])
                arg_params['s_feat_conv2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['s_feat_conv2_weight'])
                arg_params['s_feat_conv2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['s_feat_conv2_bias'])
            if cfg.network.small_net_fuse_type == 'concatv2':
                arg_params['fuse_reduce_c1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fuse_reduce_c1_weight'])
                arg_params['fuse_reduce_c1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fuse_reduce_c1_bias'])
                arg_params['s_feat_conv1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['s_feat_conv1_weight'])
                arg_params['s_feat_conv1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['s_feat_conv1_bias'])
                arg_params['s_feat_conv2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['s_feat_conv2_weight'])
                arg_params['s_feat_conv2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['s_feat_conv2_bias'])
            if cfg.network.small_net_scale_before_fuse:
                arg_params['cur_scale_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cur_scale_weight'])
                arg_params['cur_scale_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cur_scale_bias'])
            if cfg.network.small_net_bn_before_fuse:
                arg_params['warp_conv_feat_bn_gamma'] = mx.nd.ones(shape=self.arg_shape_dict['warp_conv_feat_bn_gamma'])
                arg_params['warp_conv_feat_bn_beta'] = mx.nd.zeros(shape=self.arg_shape_dict['warp_conv_feat_bn_beta'])
                aux_params['warp_conv_feat_bn_moving_mean'] = mx.nd.zeros(shape=self.aux_shape_dict['warp_conv_feat_bn_moving_mean'])
                aux_params['warp_conv_feat_bn_moving_var'] = mx.nd.ones(shape=self.aux_shape_dict['warp_conv_feat_bn_moving_var'])
                arg_params['cur_feat_bn_gamma'] = mx.nd.ones(shape=self.arg_shape_dict['cur_feat_bn_gamma'])
                arg_params['cur_feat_bn_beta'] = mx.nd.zeros(shape=self.arg_shape_dict['cur_feat_bn_beta'])
                aux_params['cur_feat_bn_moving_mean'] = mx.nd.zeros(shape=self.aux_shape_dict['cur_feat_bn_moving_mean'])
                aux_params['cur_feat_bn_moving_var'] = mx.nd.ones(shape=self.aux_shape_dict['cur_feat_bn_moving_var'])
        if cfg.network.add_dcn:
            arg_params['stage2_unit4_conv2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage2_unit4_conv2_offset_weight'])
            arg_params['stage2_unit4_conv2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage2_unit4_conv2_offset_bias'])
            arg_params['stage3_unit23_conv2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage3_unit23_conv2_offset_weight'])
            arg_params['stage3_unit23_conv2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage3_unit23_conv2_offset_bias'])
            arg_params['stage4_unit1_conv2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit1_conv2_offset_weight'])
            arg_params['stage4_unit1_conv2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit1_conv2_offset_bias'])
            arg_params['stage4_unit2_conv2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_conv2_offset_weight'])
            arg_params['stage4_unit2_conv2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit2_conv2_offset_bias'])
            arg_params['stage4_unit3_conv2_offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_conv2_offset_weight'])
            arg_params['stage4_unit3_conv2_offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['stage4_unit3_conv2_offset_bias'])
            #arg_params['rfcn_cls_offset_t_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_offset_t_weight'])
            #arg_params['rfcn_cls_offset_t_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_offset_t_bias'])
            #arg_params['rfcn_bbox_offset_t_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_offset_t_weight'])
            #arg_params['rfcn_bbox_offset_t_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_offset_t_bias'])
        arg_params['feat_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['feat_conv_3x3_weight'])
        arg_params['feat_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['feat_conv_3x3_bias'])

        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

        arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_cls_weight'])
        arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_bias'])
        arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_bbox_weight'])
        arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_bias'])

        #rnet params
        if cfg.network.res_diff_bn:
            arg_params['res_diff_bn_gamma'] = mx.nd.ones(shape=self.arg_shape_dict['res_diff_bn_gamma'])
            arg_params['res_diff_bn_beta'] = mx.nd.zeros(shape=self.arg_shape_dict['res_diff_bn_beta'])
            aux_params['res_diff_bn_moving_mean'] = mx.nd.zeros(shape=self.aux_shape_dict['res_diff_bn_moving_mean'])
            aux_params['res_diff_bn_moving_var'] = mx.nd.ones(shape=self.aux_shape_dict['res_diff_bn_moving_var'])
        rnet_num_conv = cfg.network.rnet_num_conv
        for i in range(rnet_num_conv+1):
            arg_params['rnet_conv%d_weight'%i] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rnet_conv%d_weight'%i])
            arg_params['rnet_conv%d_bias'%i] = mx.nd.zeros(shape=self.arg_shape_dict['rnet_conv%d_bias'%i])
        if cfg.network.fuse_type == 'concat':
            arg_params['fuse_downsample_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fuse_downsample_weight'])
            arg_params['fuse_downsample_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fuse_downsample_bias'])
        if 'conv' in cfg.network.fnet_type:
            fnet_num_conv = int(cfg.network.fnet_type.split('#')[1])
            for i in range(fnet_num_conv):
                arg_params['fnet_conv%d_weight'%i] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fnet_conv%d_weight'%i])
                arg_params['fnet_conv%d_bias'%i] = mx.nd.zeros(shape=self.arg_shape_dict['fnet_conv%d_bias'%i])
        if cfg.network.add_Nq_net:
            arg_params['Nq_conv1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['Nq_conv1_weight'])
            arg_params['Nq_conv1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['Nq_conv1_bias'])
            arg_params['Nq_conv2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['Nq_conv2_weight'])
            arg_params['Nq_conv2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['Nq_conv2_bias'])
            arg_params['Nq_conv3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['Nq_conv3_weight'])
            arg_params['Nq_conv3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['Nq_conv3_bias'])
        if cfg.network.add_Fgfa_net:
            arg_params['em_conv1_weight'] = mx.random.normal(0, self.get_msra_std(self.arg_shape_dict['em_conv1_weight']),
                                                         shape=self.arg_shape_dict['em_conv1_weight'])
            arg_params['em_conv1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['em_conv1_bias'])
            arg_params['em_conv2_weight'] = mx.random.normal(0, self.get_msra_std(self.arg_shape_dict['em_conv2_weight']),
                                                         shape=self.arg_shape_dict['em_conv2_weight'])
            arg_params['em_conv2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['em_conv2_bias'])
            arg_params['em_conv3_weight'] = mx.random.normal(0, self.get_msra_std(self.arg_shape_dict['em_conv3_weight']),
                                                         shape=self.arg_shape_dict['em_conv3_weight'])
            arg_params['em_conv3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['em_conv3_bias'])        
        arg_params['Convolution5_scale_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['Convolution5_scale_weight'])
        arg_params['Convolution5_scale_bias'] = mx.nd.ones(shape=self.arg_shape_dict['Convolution5_scale_bias'])
 



