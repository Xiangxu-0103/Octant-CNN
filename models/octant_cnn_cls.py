import os
import sys
import tensorflow as tf
BASR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASR_DIR)
sys.path.append(ROOT_DIR)
from utils import tf_util
from utils.octant_utils import octant_cnn_module, subsample_module
from models.transform_nets import input_transform_net


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output is Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud = tf.matmul(point_cloud, transform)
    l0_xyz = point_cloud
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

    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')
    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B,
    """
    labels = tf.one_hot(indices=label, depth=40)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__ == '__main__':
    with tf.Graph().as_default() as graph:
        inputs = tf.zeros((16, 1024, 3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
