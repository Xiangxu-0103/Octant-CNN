import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
from utils import tf_util
from utils.octant_utils import octant_cnn_module, subsample_module, pointnet_fp_module


def get_transform(point_cloud, is_training, bn_decay=None, K=3):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1, 3], padding='VALID', stride=[1, 1],
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
        assert K == 3
        weights = tf.get_variable('weights', [128, 3 * K], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        biases = tf.get_variable('biases', [3 * K], initializer=tf.constant_initializer(0.0), dtype=tf.float32) + tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


def get_model(point_cloud, input_label, is_training, cat_num, part_num,
              batch_size, num_point, weight_decay, bn_decay=None):
    end_points = {}

    with tf.variable_scope('transform_net') as sc:
        K = 3
        transform = get_transform(point_cloud, is_training, bn_decay, K)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    l0_xyz = point_cloud_transformed
    l0_points = None

    l0_points = octant_cnn_module(l0_xyz, l0_points, [64, 64, 128], is_training=is_training,
                                  bn_decay=bn_decay, scope='conv1')
    l1_xyz, l1_points = subsample_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32,
                                         group_all=False, scope='pool1')

    l1_points = octant_cnn_module(l1_xyz, l1_points, [128, 128, 256], is_training=is_training,
                                  bn_decay=bn_decay, scope='conv2')
    l2_xyz, l2_points = subsample_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64,
                                         group_all=False, scope='pool2')

    l2_points = octant_cnn_module(l2_xyz, l2_points, [256, 512, 1024], is_training=is_training,
                                  bn_decay=bn_decay, scope='conv3')
    l3_xyz, l3_points = subsample_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
                                         group_all=True, scope='pool3')

    # classification network
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='cla/fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='cla/fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='cla/dp1')
    net = tf_util.fully_connected(net, cat_num, activation_fn=None, scope='cla/fc3')

    # segmentation network
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, tf.concat([l2_xyz, l2_points], axis=-1), l3_points, [256, 256],
                                   is_training, bn_decay, scope='fa_layer1', weight_decay=weight_decay)
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, tf.concat([l1_xyz, l1_points], axis=-1), l2_points, [256, 128],
                                   is_training, bn_decay, scope='fa_layer2', weight_decay=weight_decay)

    one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, cat_num])
    one_hot_label_expand = tf.tile(one_hot_label_expand, [1, num_point, 1])
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([one_hot_label_expand, l0_xyz, l0_points], axis=-1), l1_points, [128, 128],
                                   is_training, bn_decay, scope='fa_layer3', weight_decay=weight_decay)

    net2 = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1',
                          bn_decay=bn_decay, weight_decay=weight_decay)
    net2 = tf_util.dropout(net2, keep_prob=0.5, is_training=is_training, scope='seg/dp1')
    net2 = tf_util.conv1d(net2, part_num, 1, padding='VALID', activation_fn=None,
                          bn=False, scope='fc2', weight_decay=weight_decay)
    return net, net2, end_points


def get_loss(l_pred, seg_pred, label, seg, weight, end_points):
    per_instance_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l_pred, labels=label)
    label_loss = tf.reduce_mean(per_instance_label_loss)

    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)

    per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

    total_loss = weight * seg_loss + (1 - weight) * label_loss
    return total_loss, label_loss, per_instance_label_loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res
