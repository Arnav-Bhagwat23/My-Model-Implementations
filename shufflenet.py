#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: customized_shufflenet.py

import argparse
import math
import numpy as np
import os
import cv2
from tensorpack import tfv1 as tf

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, SmartInit, model_utils
from tensorpack.tfutils.scope_utils import within_name_scope
from tensorpack.utils import logger
from tensorpack.utils.gpu import count_gpus

from imagenet_utils import ImageNetBaseModel, run_evaluation, prepare_imagenet_data


@layer_register(log_shape=True)
def DepthwiseConv(inputs, filters, kernel_size, padding='SAME', stride=1,
                  initializer=None, activation=tf.identity):
    input_shape = inputs.get_shape().as_list()
    input_channels = input_shape[1]
    assert filters % input_channels == 0, (filters, input_channels)
    multiplier = filters // input_channels

    if initializer is None:
        initializer = tf.variance_scaling_initializer(2.0)
    kernel_dims = [kernel_size, kernel_size]
    filter_dims = kernel_dims + [input_channels, multiplier]

    weights = tf.get_variable('weights', filter_dims, initializer=initializer)
    conv_result = tf.nn.depthwise_conv2d(inputs, weights, [1, 1, stride, stride], padding=padding, data_format='NCHW')
    return activation(conv_result, name='output')


@within_name_scope()
def shuffle_channels(tensor, groups):
    tensor_shape = tensor.get_shape().as_list()
    num_channels = tensor_shape[1]
    assert num_channels % groups == 0, num_channels
    tensor = tf.reshape(tensor, [-1, num_channels // groups, groups] + tensor_shape[-2:])
    tensor = tf.transpose(tensor, [0, 2, 1, 3, 4])
    tensor = tf.reshape(tensor, [-1, num_channels] + tensor_shape[-2:])
    return tensor


@layer_register()
def shufflenet_block(input_tensor, out_channels, groups, stride):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[1]
    skip_connection = input_tensor

    first_split = groups if input_channels > 24 else 1
    intermediate = Conv2D('conv1', input_tensor, out_channels // 4, 1, split=first_split, activation=BNReLU)
    intermediate = shuffle_channels(intermediate, groups)
    intermediate = DepthwiseConv('depthwise_conv', intermediate, out_channels // 4, 3, stride=stride)
    intermediate = BatchNorm('depthwise_bn', intermediate)

    intermediate = Conv2D('conv2', intermediate,
                          out_channels if stride == 1 else out_channels - input_channels,
                          1, split=groups)
    intermediate = BatchNorm('conv2_bn', intermediate)
    if stride == 1:
        output = tf.nn.relu(skip_connection + intermediate)
    else:
        skip_connection = AvgPooling('avg_pool', skip_connection, 3, 2, padding='SAME')
        output = tf.concat([skip_connection, tf.nn.relu(intermediate)], axis=1)
    return output


@layer_register()
def advanced_shufflenet_block(input_tensor, out_channels, stride):
    if stride == 1:
        skip_tensor, input_tensor = tf.split(input_tensor, 2, axis=1)
    else:
        skip_tensor, input_tensor = input_tensor, input_tensor
    skip_channels = int(skip_tensor.shape[1])

    intermediate = Conv2D('conv1', input_tensor, out_channels // 2, 1, activation=BNReLU)
    intermediate = DepthwiseConv('dconv', intermediate, out_channels // 2, 3, stride=stride)
    intermediate = BatchNorm('dconv_bn', intermediate)
    intermediate = Conv2D('conv2', intermediate, out_channels - skip_channels, 1, activation=BNReLU)

    if stride == 2:
        skip_tensor = DepthwiseConv('shortcut_dconv', skip_tensor, skip_channels, 3, stride=2)
        skip_tensor = BatchNorm('shortcut_dconv_bn', skip_tensor)
        skip_tensor = Conv2D('shortcut_conv', skip_tensor, skip_channels, 1, activation=BNReLU)
    output = tf.concat([skip_tensor, intermediate], axis=1)
    output = shuffle_channels(output, 2)
    return output


@layer_register(log_shape=True)
def shufflenet_stage(input_tensor, output_channels, num_blocks, groups):
    intermediate = input_tensor
    for block_index in range(num_blocks):
        block_name = 'block{}'.format(block_index)
        if args.version2:
            intermediate = advanced_shufflenet_block(block_name, intermediate, output_channels, 2 if block_index == 0 else 1)
        else:
            intermediate = shufflenet_block(block_name, intermediate, output_channels, groups, 2 if block_index == 0 else 1)
    return intermediate


class CustomModel(ImageNetBaseModel):
    regularization_weight = 4e-5

    def compute_logits(self, input_image):

        with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='channels_first'), \
                argscope(Conv2D, use_bias=False):

            groups = args.groups
            if not args.version2:
                channels = {
                    3: [240, 480, 960],
                    4: [272, 544, 1088],
                    8: [384, 768, 1536]
                }
                multiplier = groups * 4
                channels = [int(math.ceil(x * args.scaling / multiplier) * multiplier)
                            for x in channels[groups]]
                first_channel = int(math.ceil(24 * args.scaling / groups) * groups)
            else:
                channels = {
                    0.5: [48, 96, 192],
                    1.: [116, 232, 464]
                }[args.scaling]
                first_channel = 24

            logger.info("Channels: " + str([first_channel] + channels))

            layer = Conv2D('conv1', input_image, first_channel, 3, strides=2, activation=BNReLU)
            layer = MaxPooling('maxpool1', layer, 3, 2, padding='SAME')

            layer = shufflenet_stage('stage2', layer, channels[0], 4, groups)
            layer = shufflenet_stage('stage3', layer, channels[1], 8, groups)
            layer = shufflenet_stage('stage4', layer, channels[2], 4, groups)

            if args.version2:
                layer = Conv2D('conv5', layer, 1024, 1, activation=BNReLU)

            layer = GlobalAvgPooling('gap', layer)
            final_logits = FullyConnected('final_fc', layer, 1000)
            return final_logits


def load_data(dataset_name, batch_size):
    is_training = dataset_name == 'train'

    if is_training:
        augmentations = [
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.49 if args.scaling < 1 else 0.08, 1.)),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentations = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return prepare_imagenet_data(
        args.dataset, dataset_name, batch_size, augmentations)


def prepare_training(model, num_towers):
    batch_per_tower = TOTAL_BATCH_SIZE // num_towers

    logger.info("Running with {} towers. Batch per tower: {}".format(num_towers, batch_per_tower))
    train_data = load_data('train', batch_per_tower)
    val_data = load_data('val', batch_per_tower)

    step_size = 1280000 // TOTAL_BATCH_SIZE
    max_iterations = 3 * 10**5
    max_epochs = (max_iterations // step_size) + 1
    training_callbacks = [
        ModelSaver(),
        ScheduledHyperParamSetter('learning_rate',
                                  [(0, 0.5), (max_iterations, 0)],
                                  interp='linear', step_based=True),
        EstimatedTimeLeft()
    ]
    metrics = [ClassificationError('top1_err', 'val_top1_err'),
               ClassificationError('top5_err', 'val_top5_err')]
    if num_towers == 1:
        training_callbacks.append(InferenceRunner(QueueInput(val_data),
                                                  ScalarStats(metrics)))
    return TrainConfig(
        data=train_data,
        model=model,
        callbacks=training_callbacks,
        steps_per_epoch=step_size,
        max_epoch=max_epochs
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaling', type=float, default=1.0)
    parser.add_argument('--groups', type=int, default=3, choices=[3, 4, 8])
    parser.add_argument('--version2', action='store_true', help='Use ShuffleNetV2')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()

    num_towers = max(1, count_gpus())
    global TOTAL_BATCH_SIZE
    TOTAL_BATCH_SIZE = args.batch

    model = CustomModel()
    logger.info("Model scaling: {} | Groups: {} | Version 2: {}".format(args.scaling, args.groups, args.version2))

    if args.evaluate:
        run_evaluation(model, args.dataset)
    else:
        train_config = prepare_training(model, num_towers)
        launch_train_with_config(train_config, SyncMultiGPUTrainerReplicated(num_towers))


if __name__ == '__main__':
    main()
