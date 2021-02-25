import tensorflow as tf
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from utils import tf_util
from utils.octant_utils import octant_cnn_module, subsample_module, pointnet_fp_module


def get_transform(point_cloud, is_training, bn_decay=None, K=3):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1, K], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        weights = tf.get_variable('weights', [128, K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K], initializer=tf.constant_initializer(0.0), dtype=tf.float32) + tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """
    with tf.variable_scope('transform_net') as sc:
        transform = get_transform(point_cloud, is_training, bn_decay, K=9)
    point_cloud = tf.matmul(point_cloud, transform)
    """

    l0_xyz = point_cloud[:, :, :3]
    l0_points = point_cloud[:, :, 3:]

    l0_points = octant_cnn_module(l0_xyz, l0_points, [32, 32, 64], is_training=is_training,
                                  bn_decay=bn_decay, scope='conv1')
    l1_xyz, l1_points = subsample_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32,
                                         group_all=False, scope='pool1')

    l1_points = octant_cnn_module(l1_xyz, l1_points, [64, 64, 128], is_training=is_training,
                                  bn_decay=bn_decay, scope='conv2')
    l2_xyz, l2_points = subsample_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=32,
                                         group_all=False, scope='pool2')

    l2_points = octant_cnn_module(l2_xyz, l2_points, [128, 128, 256], is_training=is_training,
                                  bn_decay=bn_decay, scope='conv3')
    l3_xyz, l3_points = subsample_module(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=32,
                                         group_all=False, scope='pool3')

    l3_points = octant_cnn_module(l3_xyz, l3_points, [256, 256, 512], is_training=is_training,
                                  bn_decay=bn_decay, scope='conv4')
    l4_xyz, l4_points = subsample_module(l3_xyz, l3_points, npoint=16, radius=0.8, nsample=32,
                                         group_all=False, scope='pool4')

    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, tf.concat([l3_xyz, l3_points], axis=-1), l4_points, [256, 256],
                                   is_training, bn_decay, scope='sem_fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, tf.concat([l2_xyz, l2_points], axis=-1), l3_points, [256, 256],
                                   is_training, bn_decay, scope='sem_fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, tf.concat([l1_xyz, l1_points], axis=-1), l2_points, [256, 128],
                                   is_training, bn_decay, scope='sem_fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz, l0_points], axis=-1), l1_points, [128, 128, 128],
                                   is_training, bn_decay, scope='sem_fa_layer4')

    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training,
                         scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 13, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net


def get_loss(pred, label):
    loss = tf.losses.sparse_softmax_cross_entropy(logits=pred, labels=label)
    return tf.reduce_mean(loss)
