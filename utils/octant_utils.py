import tensorflow as tf
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from utils import tf_util
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from tf_ops.grouping.tf_grouping import group_point, query_ball_point, knn_point
from tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate
from tf_ops.octant.tf_octant import octant_select


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=False):
    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))  # (batch_size, npoint, 3)
    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nsample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group_all(xyz, points, use_xyz=False):
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0, 0, 0]).reshape((1, 1, 3)), (batch_size, 1, 1)),
                          dtype=tf.float32)  # (batch_size, 1, 3)
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def octant_cnn_module(xyz, points, out_channels, is_training, bn_decay, scope, bn=True, use_xyz=True, weight_decay=None):
    concat_points = []
    with tf.variable_scope(scope) as sc:
        idx = octant_select(xyz)
        grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8, 3)
        grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 8, 1])  # translation normalization
        grouped_xyz = tf.concat([grouped_xyz, tf.tile(tf.expand_dims(xyz, 2), [1, 1, 8, 1])], axis=-1)  # (batch_size, npoint, 8, 6)
        for i, out_channel in enumerate(out_channels):
            if points is not None:
                grouped_points = group_point(points, idx)  # (batch_size, npoint, 8, channel)
                if use_xyz:
                    new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, 8, 6+channel)
                else:
                    new_points = grouped_points
            else:
                new_points = grouped_xyz
            points = tf_util.conv2d(new_points, out_channel, [1, 8],
                                    padding='VALID', stride=[1, 1],
                                    bn=bn, is_training=is_training,
                                    scope='conv_{}'.format(i), bn_decay=bn_decay,
                                    weight_decay=weight_decay)
            points = tf.squeeze(points, [2])  # (batch_size, npoint, out_channel)
            concat_points.append(points)
        
        fuse_points = tf.concat(concat_points, axis=-1)
        fuse_points = tf_util.conv1d(fuse_points, out_channels[-1], 1, padding='VALID', bn=True,
                                     is_training=is_training, scope='concat_conv', bn_decay=bn_decay)
        points = fuse_points + concat_points[-1]
        return points


def subsample_module(xyz, points, npoint, radius, nsample, group_all, scope, pooling='max', knn=False, use_xyz=False):
    with tf.variable_scope(scope) as sc:
        # Sampling and Grouping
        if group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        if pooling == 'max':
            new_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')
        elif pooling == 'avg':
            new_points = tf.reduce_mean(new_points, axis=[2], keepdims=True, name='avgpool')
        elif pooling == 'max_and_avg':
            max_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')
            avg_points = tf.reduce_mean(new_points, axis=[2], keepdims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        new_points = tf.squeeze(new_points, [2])
        return new_xyz, new_points


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True, weight_decay=None):
    """PointNet Feature Propogation (FP) Module
    Input:
        xyz1: (batch_size, ndataset1, 3) TF tensor
        xyz2: (batch_size, ndataset2, 3) TF tensor
        points1: (batch_size, ndataset1, nchannel1) TF tensor
        points2: (batch_size, ndataset2, nchannel2) TF tensor
        mlp: list of int32 -- output size for MLP on each point
    Return:
        new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    """
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])  # B, ndataset1, nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_{}'.format(i), bn_decay=bn_decay,
                                         weight_decay=weight_decay)
        new_points1 = tf.squeeze(new_points1, [2])
        return new_points1
