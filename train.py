import os
import sys
import argparse

# Teminal arguments
parser = argparse.ArgumentParser(description='Convert dataset to tfrecord')
parser.add_argument('-d','--dataset',type=str,help='path to dataset',default='./data/train')
parser.add_argument('-dv','--val_dataset',type=str,help='path to val dataset',default=None)
parser.add_argument('--gpu', type=int,help='GPU ID, default is 0, -1 for using CPU', default=0)
args = parser.parse_args()

# Remove logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Set using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import tensorflow as tf

from datetime import datetime
from configs.config import Config
from src import utils
from src.models.model import MaskRCNN
from src.dataset.dataset import Dataset
from src.loss_function.loss import (rpn_class_loss_graph, rpn_bbox_loss_graph, mrcnn_class_loss_graph, 
                                    mrcnn_bbox_loss_graph, mrcnn_mask_loss_graph)
def main(args):
    # init
    utils.set_memory_growth()
    config = Config()
    
    # coco config
    config.NUM_CLASSES = 1
    config.IMAGES_PER_GPU = 1
    config.IMAGE_MIN_DIM = 224
    config.IMAGE_MAX_DIM = 448
    config.STEPS = 1000
    config.STEPS_PER_EPOCH = 500
    
    
    # load dataset
    train_dataset = Dataset(config, args.dataset,classes_id=[1])
    train_dataset = iter(train_dataset)
    if args.val_dataset is not None:
        val_dataset = Dataset(config, args.val_dataset,classes_id=[1])
        val_dataset = iter(val_dataset)
        min_val_total_loss = sys.float_info.max
        
    # initialize model
    model = MaskRCNN(config,training=True)

    # create logs, weights and samples prediction folder
    save_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(config.CHECKPOINT_DIR, save_time,'logs')
    weights_dir = os.path.join(config.CHECKPOINT_DIR, save_time,'weights')
    samples = os.path.join(config.CHECKPOINT_DIR, save_time,'samples')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(samples, exist_ok=True)
    
    # log writer
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM)
    
    # define training tf function for faster training
    @tf.function
    def train_step(data):
        with tf.GradientTape() as tape:
            active_class_ids = data["batch_image_meta"][:, 12:]
            inputs = {"input_image": data["batch_images"],
                    "input_image_meta": data["batch_image_meta"],
                    "input_gt_class_ids": data["batch_gt_class_ids"],
                    "input_gt_boxes": data["batch_gt_boxes"],
                    "input_gt_masks": data["batch_gt_masks"]}
            outputs = model(inputs, training=True)
            
            # Losses
            rpn_class_loss = rpn_class_loss_graph(data["batch_rpn_match"], outputs["rpn_class_logits"])
            rpn_bbox_loss = rpn_bbox_loss_graph(config,data["batch_rpn_bbox"], data["batch_rpn_match"], outputs["rpn_bbox"])
            class_loss = mrcnn_class_loss_graph(outputs["target_class_ids"], outputs["mrcnn_class_logits"], active_class_ids)
            bbox_loss = mrcnn_bbox_loss_graph(outputs["target_bbox"], outputs["target_class_ids"], outputs["mrcnn_bbox"])
            mask_loss = mrcnn_mask_loss_graph(outputs["target_mask"], outputs["target_class_ids"], outputs["mrcnn_mask"])
            # Add L2 Regularization
            # Skip gamma and beta weights of batch normalization layers.
            reg_loss = tf.add_n([tf.keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                        for w in model.trainable_weights
                        if 'gamma' not in w.name and 'beta' not in w.name])
            losses = {"rpn_class_loss": rpn_class_loss,
                    "rpn_bbox_loss": rpn_bbox_loss,
                    "class_loss": class_loss,
                    "bbox_loss": bbox_loss,
                    "mask_loss": mask_loss,
                    "reg_loss": reg_loss}
            total_loss = tf.add_n([l for l in losses.values()])
        # Backpropagation
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return total_loss, losses
    
    # define testing step tf function for faster inference
    @tf.function
    def test_step(data):
        active_class_ids = data["batch_image_meta"][:, 12:]
        inputs = {"input_image": data["batch_images"],
                "input_image_meta": data["batch_image_meta"],
                "input_gt_class_ids": data["batch_gt_class_ids"],
                "input_gt_boxes": data["batch_gt_boxes"],
                "input_gt_masks": data["batch_gt_masks"]}
        outputs = model(inputs, training=True)
        
        # Losses
        rpn_class_loss = rpn_class_loss_graph(data["batch_rpn_match"], outputs["rpn_class_logits"])
        rpn_bbox_loss = rpn_bbox_loss_graph(config,data["batch_rpn_bbox"], data["batch_rpn_match"], outputs["rpn_bbox"])
        class_loss = mrcnn_class_loss_graph(outputs["target_class_ids"], outputs["mrcnn_class_logits"], active_class_ids)
        bbox_loss = mrcnn_bbox_loss_graph(outputs["target_bbox"], outputs["target_class_ids"], outputs["mrcnn_bbox"])
        mask_loss = mrcnn_mask_loss_graph(outputs["target_mask"], outputs["target_class_ids"], outputs["mrcnn_mask"])
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_loss = tf.add_n([tf.keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                    for w in model.trainable_weights
                    if 'gamma' not in w.name and 'beta' not in w.name])
        losses = {"rpn_class_loss": rpn_class_loss,
                "rpn_bbox_loss": rpn_bbox_loss,
                "class_loss": class_loss,
                "bbox_loss": bbox_loss,
                "mask_loss": mask_loss,
                "reg_loss": reg_loss}
        total_loss = tf.add_n([l for l in losses.values()])
        return total_loss,losses
    
    # Visualize the training process
    prog_bar = utils.ProgressBar(config.STEPS,0,bar_width=2)
    
    # Start training
    for step in range(1,config.STEPS+1):
        # load a batch of data
        data = next(train_dataset,None)
        # check if the batch is end of dataset
        if data is None:
            train_dataset.on_epoch_end()
            train_dataset = iter(train_dataset)
            data = next(train_dataset)
        # train 1 step
        total_loss,losses = train_step(data)
        # visualize the loss
        prog_bar.update("Step={}/{}, l={:.4f},rpnc={:.4f},rpnb={:.4f},cl={:.4f},bbl={:.4f},ml={:.4f},rl={:.4f}".format(
            step, config.STEPS,total_loss.numpy(),
            losses["rpn_class_loss"].numpy(),losses["rpn_bbox_loss"].numpy(),
            losses["class_loss"].numpy(),losses["bbox_loss"].numpy(),
            losses["mask_loss"].numpy(),losses["reg_loss"].numpy()))
        # write log after every 10 steps
        if step % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar(
                    'loss/total_loss', total_loss, step=step)
                for k, l in losses.items():
                    tf.summary.scalar('loss/{}'.format(k), l, step=step)
        
        # save weights using validation dataset
        if args.val_dataset is not None and step % config.VALIDATION_STEPS == 0:
            batch_val_loss = []
            while True:
                val_data = next(val_dataset,None)
                if val_data is None:
                    # reset the validation dataset after each epoch
                    val_dataset = iter(val_dataset)
                    break
                val_total_loss, val_losses = test_step(val_data)
                batch_val_loss.append(val_total_loss.numpy())
            batch_val_loss = tf.reduce_mean(batch_val_loss).numpy()
            print("\nValidation loss: {:.4f}".format(batch_val_loss))
            
            with summary_writer.as_default():
                tf.summary.scalar(
                    'val_loss/total_loss', total_loss, step=step)
                for k, l in val_losses.items():
                    tf.summary.scalar('val_loss/{}'.format(k), l, step=step)
                    
            if config.SAVE_BEST_ONLY:
                if min_val_total_loss >= batch_val_loss:
                    min_val_total_loss = batch_val_loss
                    model.save_weights(os.path.join(weights_dir, 'mask_rcnn_best_{}.h5'.format(step)))
            
        # save weights after every epoch
        elif step % config.STEPS_PER_EPOCH== 0:
            model.save_weights(os.path.join(weights_dir, 'mask_rcnn_{}.h5'.format(step)))


if __name__ == '__main__':
    main(args)