import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.dataset.dataset import Dataset
from configs.config import Config
from src import utils
from src.models.model import MaskRCNN
import tensorflow as tf
import cv2
import numpy as np
from src.models import layers as layersc
config = Config()

# coco config
config.NUM_CLASSES = 80
config.IMAGES_PER_GPU = 2
config.IMAGE_MIN_DIM = 224
config.IMAGE_MAX_DIM = 448
config.STEPS = 100000
config.STEPS_PER_EPOCH = 500
train_dataset = Dataset(config, './data/testing',classes_id=[1])

# initialize model
model = MaskRCNN(config,training=True)
# model.summary()
model.load_weights('/home/minami/Code/Git-repo/Mask-RCNN/data/mask_rcnn_6000.h5', by_name=True)
data = train_dataset.get_random_batch()

inputs = {"input_image": data["batch_images"],
                    "input_image_meta": data["batch_image_meta"],
                    "input_gt_class_ids": data["batch_gt_class_ids"],
                    "input_gt_boxes": data["batch_gt_boxes"],
                    "input_gt_masks": data["batch_gt_masks"]}
# print("Input data:",data["batch_gt_class_ids"])
# print("Gt boxes:", data["batch_gt_boxes"])
# print(data["batch_gt_boxes"])

outputs = model(inputs, training=True)

print(outputs["target_bbox"])
print(outputs["target_class_ids"])
# detection = tf.concat([tf.cast(outputs["target_bbox"],tf.float32),
#                       tf.cast(outputs["target_class_ids"][...,tf.newaxis],tf.float32),
#                       tf.ones([2,200,1],dtype=tf.float32)],axis = -1)
# for i in outputs["target_class_ids"][0].numpy():
#     print(i)
# decode = utils.test_dataset(detection.numpy(), outputs["target_mask"][...,tf.newaxis].numpy(),
#                             data["batch_images"], config)


# anchors = utils.get_anchors(config,config.IMAGE_SHAPE)
# print(anchors)
print(
    decode["rois"].shape,
    decode["class_ids"].shape,
    decode["scores"].shape,
    decode["masks"].shape )
for i in range(len(data["batch_images"])):
    image = utils.unmold_image(data["batch_images"][i], config)
    r = data["batch_gt_boxes"][i]
    r = r[~np.all(r == 0, axis=1)]
    # mask = data["batch_gt_masks"][i] * 255
    # mask = mask.astype(np.uint8)
    
    for bbox in r:
        cv2.rectangle(image, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), (0, 255, 0), 2)
    # cv2.imshow('img',image)
    for proposal in outputs["rpn_rois"][i]:
        img = image.copy()
        cv2.rectangle(img, (int(proposal[1]*image.shape[0]), int(proposal[0]*image.shape[1])), 
                      (int(proposal[3]*image.shape[0]), int(proposal[2]*image.shape[1])), (0, 0, 255), 2)
        cv2.imshow('img', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()