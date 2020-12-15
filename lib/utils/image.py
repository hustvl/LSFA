# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import numpy as np
import os
import cv2
import random
from PIL import Image
from bbox.bbox_transform import clip_boxes
import coviar_py2 

GOP_SIZE = 12

# TODO: This two functions should be merged with individual data loader
def get_image(roidb, config, cur_frame_id):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    processed_motion_vector = []
    processed_res_diff = []
    for i in range(num_images):
        roi_rec = roidb[i]
        im_h = int(roi_rec['height'])
        im_w = int(roi_rec['width'])
        motion_vector = np.zeros((im_h, im_w, 2), dtype=np.float32)
        res_diff = np.zeros((im_h, im_w, 3), dtype=np.float32)
        num_frames = roi_rec['frame_seg_len']
        gop_id = cur_frame_id // GOP_SIZE
        pos_id = cur_frame_id % GOP_SIZE

        #assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        if cur_frame_id + 1 == num_frames:
            im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            image_dirs = roi_rec['image'].split('/')
            video_name = image_dirs[-2] + '.mp4'
            video_dir = os.path.join(image_dirs[0], image_dirs[1], image_dirs[2], image_dirs[3], image_dirs[4], 'mpeg4_snippets', image_dirs[5], video_name)
            assert os.path.exists(video_dir), '%s does not exists'.format(video_dir)
            im = coviar_py2.load(video_dir, gop_id, pos_id, 0, True) .astype(np.float32)  
            motion_vector = coviar_py2.load(video_dir, gop_id, pos_id, 1, True).astype(np.float32)
            motion_vector = - motion_vector
            res_diff = coviar_py2.load(video_dir, gop_id, pos_id, 2, True).astype(np.float32)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            motion_vector = motion_vector[:, ::-1]
            motion_vector[:, :, 0] = - motion_vector[:, :, 0]
            res_diff = res_diff[:, ::-1, :]
       
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS, config.network.PIXEL_SCALE)
        motion_vector_tensor, res_diff_tensor = transform_mv_res(motion_vector, res_diff, im_scale, config.network.PIXEL_MEANS, config.network.PIXEL_SCALE)
        processed_ims.append(im_tensor)
        processed_motion_vector.append(motion_vector_tensor)
        processed_res_diff.append(res_diff_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb, processed_motion_vector, processed_res_diff


def check_reconstruction(ref_im, im, motion_vector, res_diff, video_dir, cur_frame_id, ref_id, gop_id, pos_id, ref_gop_id, ref_pos_id):
    im_h, im_w, _ = im.shape
    for i in range(im_w):
        for j in range(im_h):
            mv_i_, mv_j_ = motion_vector[j, i]
            mv_i = i - mv_i_ 
            mv_j = j - mv_j_
            res = res_diff[j, i]
            if not (ref_im[mv_j, mv_i] + res == im[j, i]).all():
                import pdb;pdb.set_trace()
    return True


def get_pair_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ref_ims = []
    processed_old_ref_ims = []
    processed_eq_flags = []
    processed_eq_flags_old = []
    processed_roidb = []
    processed_motion_vector = []
    processed_res_diff = []
    for i in range(num_images):
        roi_rec = roidb[i]

        eq_flag = 0 # 0 for unequal, 1 for equal
        eq_flag_old = 0
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        #im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        im_h = int(roi_rec['height'])
        im_w = int(roi_rec['width'])
        motion_vector = np.zeros((im_h, im_w, 2))
        res_diff = np.zeros((im_h, im_w, 3))
        #import pdb;pdb.set_trace()
        if roi_rec.has_key('pattern'):
            ref_id = min(max(roi_rec['frame_seg_id'] + np.random.randint(config.TRAIN.MIN_OFFSET, config.TRAIN.MAX_OFFSET+1), 0),roi_rec['frame_seg_len']-1)
            cur_frame_id = roi_rec['frame_seg_id']
            gop_id = cur_frame_id // GOP_SIZE
            pos_id = cur_frame_id % GOP_SIZE
            image_dirs = roi_rec['image'].split('/')
            video_name = image_dirs[-2] + '.mp4'
            video_dir = os.path.join(image_dirs[0], image_dirs[1], image_dirs[2], image_dirs[3], image_dirs[4], 'mpeg4_snippets', image_dirs[5], image_dirs[6], video_name)
            assert os.path.exists(video_dir), '%s does not exists'.format(video_dir)
            num_frames = coviar_py2.get_num_frames(video_dir)
            if num_frames == cur_frame_id: #last video frame. coviar can not decode the last frame
                ref_id = cur_frame_id
                im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
                ref_im = im.copy()
                old_ref_im = im.copy()
            else:  #this frame is not the last frame, then load from coviar
                im = coviar_py2.load(video_dir, gop_id, pos_id, 0, True).astype(np.float32)
            if pos_id == 0 or ref_id == cur_frame_id: #key frame or last frame or just random key frame
                ref_id = cur_frame_id
                eq_flag = 1
                ref_im = im.copy()
                old_ref_im = im.copy()
            else:
                ref_id = gop_id * GOP_SIZE
                #ref_image = roi_rec['pattern'] % ref_id
                #assert os.path.exists(ref_image), '%s does not exist'.format(ref_image)
                #ref_im = cv2.imread(ref_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
                ref_gop_id = ref_id // GOP_SIZE
                ref_pos_id = ref_id % GOP_SIZE
                old_ref_gop_id = ref_gop_id - 1 if ref_gop_id > 0 else 0
                eq_flag_old = 1 if old_ref_gop_id == ref_gop_id else 0
                old_ref_im = coviar_py2.load(video_dir, old_ref_gop_id, ref_pos_id, 0, True).astype(np.float32)
                ref_im = coviar_py2.load(video_dir, ref_gop_id, ref_pos_id, 0, True).astype(np.float32)
                motion_vector = coviar_py2.load(video_dir, gop_id, pos_id, 1, True)
                motion_vector = - motion_vector
                res_diff = coviar_py2.load(video_dir, gop_id, pos_id, 2, True)
        else:
            im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            ref_im = im.copy()
            old_ref_im = im.copy()
            eq_flag = 1

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            ref_im = ref_im[:, ::-1, :]
            old_ref_im = old_ref_im[:, ::-1, :]
            motion_vector = motion_vector[:, ::-1]
            motion_vector[:, :, 0] = - motion_vector[:, :, 0]
            res_diff = res_diff[:, ::-1, :]
        #check motion vector and residual difference
        #if eq_flag == 0:
        #    print roidb[i]['flipped']
        #    check_reconstruction(ref_im, im, motion_vector, res_diff, video_dir, cur_frame_id, ref_id, gop_id, pos_id, ref_gop_id, ref_pos_id)

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        ref_im, im_scale = resize(ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        old_ref_im, im_scale = resize(old_ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS, config.network.PIXEL_SCALE)
        ref_im_tensor = transform(ref_im, config.network.PIXEL_MEANS, config.network.PIXEL_SCALE)
        old_ref_im_tensor = transform(old_ref_im, config.network.PIXEL_MEANS, config.network.PIXEL_SCALE)
        motion_vector_tensor, res_diff_tensor = transform_mv_res(motion_vector, res_diff, im_scale, config.network.PIXEL_MEANS, config.network.PIXEL_SCALE)
        processed_ims.append(im_tensor)
        processed_ref_ims.append(ref_im_tensor)
        processed_old_ref_ims.append(old_ref_im_tensor)
        processed_eq_flags.append(eq_flag)
        processed_eq_flags_old.append(eq_flag_old)
        processed_motion_vector.append(motion_vector_tensor)
        processed_res_diff.append(res_diff_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_ref_ims, processed_old_ref_ims, processed_eq_flags, processed_eq_flags_old, processed_roidb, processed_motion_vector, processed_res_diff

def transform_mv_res(motion_vector, res_diff, im_scale, pixel_means, pixel_scale, rcnn_stride=16, interpolation = cv2.INTER_LINEAR):
     
    motion_vector = cv2.resize(motion_vector.astype(np.float32), None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)
    res_diff = cv2.resize(res_diff.astype(np.float32), None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)
    
    im_h, im_w, _ = res_diff.shape
    p_im_h = int(np.ceil(im_h / float(rcnn_stride)) * rcnn_stride)
    p_im_w = int(np.ceil(im_w / float(rcnn_stride)) * rcnn_stride)

    padded_motion_vector = np.zeros((p_im_h, p_im_w, 2))
    padded_res_diff = np.zeros((p_im_h, p_im_w, 3))
        
    padded_motion_vector[:im_h, :im_w] = motion_vector
    padded_res_diff[:im_h, :im_w] = res_diff

    for i in range(3):
        padded_res_diff[:, :, i] = (padded_res_diff[:, :, 2 - i] - pixel_means[2 - i]) * pixel_scale

    rcnn_scale = 1.0 / rcnn_stride
    resize_motion_vector = cv2.resize(padded_motion_vector, None, None, fx=rcnn_scale, fy=rcnn_scale, interpolation=interpolation)
    resize_res_diff = cv2.resize(padded_res_diff, None, None, fx=rcnn_scale, fy=rcnn_scale, interpolation=interpolation)
    
    scale = im_scale * rcnn_scale
    resize_motion_vector *= scale
    tensor_h, tensor_w, _ = resize_res_diff.shape
    motion_vector_tensor = resize_motion_vector.transpose((2, 0, 1)).reshape(1, 2, tensor_h, tensor_w)
    res_diff_tensor = resize_res_diff.transpose((2, 0, 1)).reshape(1, 3, tensor_h, tensor_w)
    #motion_vector_tensor[:] = np.random.randint(-10, 10)
    #motion_vector_tensor[:] = 0.0
    #res_diff_tensor[:] = 0.0
    '''
    motion_vector = cv2.resize(motion_vector.astype(np.float32), None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)
    res_diff = cv2.resize(res_diff.astype(np.float32), None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)
    
    im_h, im_w, _ = res_diff.shape
    p_im_h = int(np.ceil(im_h / float(rcnn_stride)) * rcnn_stride)
    p_im_w = int(np.ceil(im_w / float(rcnn_stride)) * rcnn_stride)

    padded_motion_vector = np.zeros((p_im_h, p_im_w, 2))
    padded_res_diff = np.zeros((p_im_h, p_im_w, 3))
        
    padded_motion_vector[:im_h, :im_w] = motion_vector
    padded_res_diff[:im_h, :im_w] = res_diff

    for i in range(3):
        padded_res_diff[:, :, i] = (padded_res_diff[:, :, 2 - i] - pixel_means[2 - i]) * pixel_scale

    rcnn_scale = 1.0 / rcnn_stride
    resize_motion_vector = cv2.resize(padded_motion_vector, None, None, fx=rcnn_scale, fy=rcnn_scale, interpolation=interpolation)
    resize_res_diff = cv2.resize(padded_res_diff, None, None, fx=rcnn_scale, fy=rcnn_scale, interpolation=interpolation)
    
    scale = im_scale * rcnn_scale
    resize_motion_vector *= scale
    tensor_h, tensor_w, _ = resize_res_diff.shape
    motion_vector_tensor = np.zeros((1, 2, tensor_h, tensor_w), dtype=np.float32)
    res_diff_tensor = np.zeros((1, 3, tensor_h, tensor_w), dtype=np.float32)
    for i in range(2):
        motion_vector_tensor[0, i] = resize_motion_vector[:, :, i]
    for i in range(3):
        res_diff_tensor[0, i] = resize_res_diff[:, :, i]
    '''    
    return motion_vector_tensor, res_diff_tensor


def resize(im, target_size, max_size, stride=0, interpolation = cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale

def transform(im, pixel_means, pixel_scale):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    im_tensor = im_tensor * pixel_scale
    return im_tensor

def transform_seg_gt(gt):
    """
    transform segmentation gt image into mxnet tensor
    :param gt: [height, width, channel = 1]
    :return: [batch, channel = 1, height, width]
    """
    gt_tensor = np.zeros((1, 1, gt.shape[0], gt.shape[1]))
    gt_tensor[0, 0, :, :] = gt[:, :]

    return gt_tensor

def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im

def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor
