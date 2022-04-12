from src.dataset.dataset import Dataset
from configs.config import Config
from src import utils
from src.models.model import MaskRCNN
import tensorflow as tf
import cv2
config = Config()

# coco config
config.NUM_CLASSES = 1
config.IMAGES_PER_GPU = 1
config.IMAGE_MIN_DIM = 224
config.IMAGE_MAX_DIM = 448
config.STEPS = 1000
config.STEPS_PER_EPOCH = 500
train_dataset = Dataset(config, './data/train',classes_id=[1])

# initialize model
model = MaskRCNN(config,training=True)
data = train_dataset.get_random_batch()

inputs = {"input_image": data["batch_images"],
                    "input_image_meta": data["batch_image_meta"],
                    "input_gt_class_ids": data["batch_gt_class_ids"],
                    "input_gt_boxes": data["batch_gt_boxes"],
                    "input_gt_masks": data["batch_gt_masks"]}
print(data["batch_gt_class_ids"])
outputs = model(inputs, training=True)
detection = tf.concat([tf.cast(outputs["target_bbox"],tf.float32),
                      tf.cast(outputs["target_class_ids"][...,tf.newaxis],tf.float32),
                      tf.ones([1,200,1],dtype=tf.float32)],axis = -1)
# for i in outputs["target_class_ids"][0].numpy():
#     print(i)
decode = utils.test_dataset(detection.numpy(), outputs["target_mask"][...,tf.newaxis].numpy(),data["batch_images"], config)
image = utils.unmold_image(data["batch_images"][0], config)

print(
decode["rois"].shape,
decode["class_ids"].shape,
decode["scores"].shape,
decode["masks"].shape )
cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()