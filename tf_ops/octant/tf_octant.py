"""
Author: Jiang Mingyang
octant-cnn module op, modified by Xiang Xu !!!
email: xiangxu0103@gmail.com
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

octant_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_octant_so.so'))


def octant_select(xyz):
    """
    :param xyz: (b, n, 3) float
    :param radius: float
    :return: (b, n, 8) int
    """
    idx = octant_module.cube_select(xyz)
    return idx


ops.NoGradient('CubeSelect')
