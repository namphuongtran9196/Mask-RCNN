"""
Mask R-CNN
The Mask R-CNN model reimplementation base on https://github.com/matterport/Mask_RCNN.git.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Rewritten by Tran Phuong Nam
"""
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from . import layers as layersc
from .. import utils

def get_backbone_feature(input_image,backbone_type='ResNet50V2'):
    """extract backbone feature from input image
    For input 256x256x3
    Return shape:
    C1 = 128x128xf for MobileNetV2 and C1 = 64x64xf for ResNet50V2
    C2 = 64x64xf
    C3 = 32x32xf
    C4 = 16x16xf
    C5 = 8x8xf
    """
    if backbone_type.lower() == 'resnet50v2':
        base_model = ResNet50V2(input_shape=input_image.shape[1:], include_top=False,
                        weights='imagenet')
        model = Model(base_model.input, 
                      [base_model.get_layer('conv2_block1_preact_relu').output,
                       base_model.get_layer('conv2_block3_preact_relu').output,
                       base_model.get_layer('conv3_block4_preact_relu').output,
                       base_model.get_layer('conv4_block6_preact_relu').output,
                       base_model.get_layer('conv5_block3_preact_relu').output])
        return model(input_image)
    elif backbone_type.lower() == 'mobilenetv2':
        base_model = MobileNetV2(input_shape=input_image.shape[1:], include_top=False,
                            weights='imagenet')
        model = Model(base_model.input, 
                      [base_model.get_layer('block_1_expand_relu').output,
                       base_model.get_layer('block_3_expand_relu').output,
                       base_model.get_layer('block_6_expand_relu').output,
                       base_model.get_layer('block_13_expand_relu').output,
                       base_model.get_layer('block_16_expand_relu').output])
        return model(input_image)
    else:
        raise TypeError('backbone_type error!')

def MaskRCNN(config,training=False):
    """Build Mask R-CNN architecture"""
    # Image size must be dividable by 2 multiple times
    h, w = config.IMAGE_SHAPE[:2]
    if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                        "to avoid fractions when downscaling and upscaling."
                        "For example, use 256, 320, 384, 448, 512, ... etc. ")
        
    # Inputs
    input_image = layers.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    input_image_meta = layers.Input(shape=[config.IMAGE_META_SIZE],name="input_image_meta")
    if training:
        # Detection GT (class IDs, bounding boxes, and masks)
        # 1. GT Class IDs (zero padded)
        input_gt_class_ids = layers.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        # 2. GT Boxes in pixels (zero padded)
        # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        input_gt_boxes = layers.Input(shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
        # Normalize coordinates
        h, w = tf.shape(input_image)[1], tf.shape(input_image)[2]
        scale = tf.cast(tf.stack([h, w, h, w],axis=0),tf.float32) - tf.constant(1.0)
        shift = tf.constant([0., 0., 1., 1.])
        gt_boxes=  tf.divide(input_gt_boxes - shift,scale)
        # scale = tf.cast(tf.stack([h, w, h, w],axis=0),tf.float32)
        # gt_boxes = tf.divide(input_gt_boxes,scale)
        
        # 3. GT Masks (zero padded)
        
        # [batch, height, width, MAX_GT_INSTANCES]
        if config.USE_MINI_MASK:
            input_gt_masks = layers.Input(
                shape=[config.MINI_MASK_SHAPE[0],config.MINI_MASK_SHAPE[1], None],
                name="input_gt_masks", dtype=bool)
        else:
            input_gt_masks = layers.Input(
                shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                name="input_gt_masks", dtype=bool)
    else:
        # Anchors in normalized coordinates
        input_anchors = layers.Input(shape=[None, 4], name="input_anchors")
        
    _, C2, C3, C4, C5 = get_backbone_feature(input_image,backbone_type=config.BACKBONE)
    
    P5 = layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    P4 = layers.Add(name="fpn_p4add")([
        layers.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    P3 = layers.Add(name="fpn_p3add")([
        layers.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    P2 = layers.Add(name="fpn_p2add")([
        layers.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = layers.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = layers.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    
    # Note that P6 is used in RPN, but not in the classifier heads.
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    mrcnn_feature_maps = [P2, P3, P4, P5]
    
    # Anchors
    if training:
        anchors = utils.get_anchors(config,config.IMAGE_SHAPE)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
        # A hack to get around Keras's bad support for constants
        anchors = layersc.AnchorsLayer(anchors, name="anchors")(input_image)
    else:
        anchors = input_anchors
        
    # RPN Model
    rpn = layersc.build_rpn_model(config.RPN_ANCHOR_STRIDE,
                            len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
    
    # Loop through pyramid layers
    layer_outputs = []  # list of lists
    for p in rpn_feature_maps:
        layer_outputs.append(rpn([p]))
    # Concatenate layer outputs
    # Convert from list of lists of level outputs to list of lists
    # of outputs across levels.
    # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
    outputs = list(zip(*layer_outputs))
    outputs = [layers.Concatenate(axis=1, name=n)(list(o))
                for o, n in zip(outputs, output_names)]

    rpn_class_logits, rpn_class, rpn_bbox = outputs
    
    # Generate proposals
    # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
    # and zero padded.
    proposal_count = config.POST_NMS_ROIS_TRAINING if training else config.POST_NMS_ROIS_INFERENCE
    rpn_rois = layersc.ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name="ROI",
        config=config)([rpn_class, rpn_bbox, anchors])
    
    if training:
        # Class ID mask to mark class IDs supported by the dataset the image
        # came from.
        if not config.USE_RPN_ROIS:
            # Ignore predicted ROIs and use ROIs provided as an input.
            input_rois = layers.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                    name="input_roi", dtype=np.int32)
            # Normalize coordinates
            target_rois = layers.Lambda(lambda x: utils.norm_boxes_graph(
                x, tf.shape(input_image)[1:3]))(input_rois)
        else:
            target_rois = rpn_rois
        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_class_ids, target_bbox, target_mask =\
            layersc.DetectionTargetLayer(config, name="proposal_targets")([
                target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

        # Network Heads
        # TODO: verify that this handles zero padded ROIs
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
            layersc.fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                    config.POOL_SIZE, config.NUM_CLASSES,
                                    train_bn=config.TRAIN_BN,
                                    fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

        mrcnn_mask = layersc.build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                            input_image_meta,
                                            config.MASK_POOL_SIZE,
                                            config.NUM_CLASSES,
                                            train_bn=config.TRAIN_BN)

        output_rois = layers.Lambda(lambda x: x * 1, name="output_rois")(rois)
        # Model
        inputs = {"input_image": input_image,
                  "input_image_meta": input_image_meta,
                  "input_gt_class_ids": input_gt_class_ids,
                  "input_gt_boxes": input_gt_boxes,
                  "input_gt_masks": input_gt_masks}
        if not config.USE_RPN_ROIS:
            inputs.update({"input_rois": input_rois})
        outputs = {"rpn_class_logits": rpn_class_logits,
                   "rpn_class": rpn_class,
                   "rpn_bbox": rpn_bbox,
                    "mrcnn_class_logits": mrcnn_class_logits,
                    "mrcnn_class": mrcnn_class,
                    "mrcnn_bbox": mrcnn_bbox,
                    "mrcnn_mask": mrcnn_mask,
                    "rpn_rois": rpn_rois,
                    "output_rois": output_rois,
                    "target_class_ids": target_class_ids,
                    "target_bbox": target_bbox,
                    "target_mask": target_mask,
                    # testing
                    "input_gt_class_ids": input_gt_class_ids,
                    "input_gt_boxes": input_gt_boxes,
                    "gt_boxes": gt_boxes,
                    "input_gt_masks" : input_gt_masks,
                    "target_rois" : target_rois
                    }
                #    rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
        model = Model(inputs, outputs, name='mask_rcnn')
    else:
        # Network Heads
        # Proposal classifier and BBox regressor heads
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
            layersc.fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                    config.POOL_SIZE, config.NUM_CLASSES,
                                    train_bn=config.TRAIN_BN,
                                    fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

        # Detections
        # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
        # normalized coordinates
        detections = layersc.DetectionLayer(config, name="mrcnn_detection")(
            [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

        # Create masks for detections
        detection_boxes = layers.Lambda(lambda x: x[..., :4])(detections)
        mrcnn_mask = layersc.build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                            input_image_meta,
                                            config.MASK_POOL_SIZE,
                                            config.NUM_CLASSES,
                                            train_bn=config.TRAIN_BN)
        inputs = {"input_image": input_image,
                  "input_image_meta": input_image_meta,
                  "input_anchors": input_anchors}
        outputs = {"detections": detections,
                   "mrcnn_class": mrcnn_class,
                   "mrcnn_bbox": mrcnn_bbox,
                    "mrcnn_mask": mrcnn_mask,
                    "rpn_rois": rpn_rois,
                    "rpn_class": rpn_class,
                    "rpn_bbox": rpn_bbox}
        model = Model(inputs,outputs,name='mask_rcnn')

    # # Add multi-GPU support.
    # if config.GPU_COUNT > 1:
    #     from mrcnn.parallel_model import ParallelModel
    #     model = ParallelModel(model, config.GPU_COUNT)

    return model
    
if __name__ == '__main__':
    from ...configs.config import Config
    model = MaskRCNN(Config(),training=True)
    model.summary()