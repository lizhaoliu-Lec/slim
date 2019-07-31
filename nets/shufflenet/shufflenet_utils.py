# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


@slim.add_arg_scope
def group_conv2d(inputs,
                 num_outputs,
                 kernel_size,
                 num_groups=1,
                 stride=1,
                 rate=1,
                 padding='SAME',
                 activation_fn=tf.nn.relu,
                 normalizer_fn=None,
                 normalizer_params=None,
                 biases_initializer=tf.zeros_initializer(),
                 scope=None):
    with tf.variable_scope(scope, 'Group_Cnv2d', [inputs]):
        biases_initializer = biases_initializer if normalizer_fn is None else None
        if num_groups == 1:
            return slim.conv2d(inputs, num_outputs, kernel_size,
                               stride=stride, rate=rate,
                               padding=padding,
                               activation_fn=activation_fn,
                               normalizer_fn=normalizer_fn,
                               normalizer_params=normalizer_params,
                               biases_initializer=biases_initializer,
                               scope=scope)
        else:
            depth_in = slim.utils.last_dimension(
                inputs.get_shape(), min_rank=4)

            assert num_outputs % num_groups == 0, (
                    "num_outputs=%d is not divisible by num_groups=%d" %
                    (num_outputs, num_groups))
            assert depth_in % num_groups == 0, (
                    "depth_in=%d is not divisible by num_groups=%d" %
                    (depth_in, num_groups))

            group_size_out = num_outputs // num_groups
            input_slices = tf.split(inputs, num_groups, axis=-1)
            output_slices = [slim.conv2d(inputs=input_slice,
                                         num_outputs=group_size_out,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         rate=rate,
                                         padding=padding,
                                         activation_fn=None,
                                         normalizer_fn=None,
                                         biases_initializer=biases_initializer,
                                         scope='group_%d' % idx)
                             for idx, input_slice in enumerate(input_slices)]
            net = tf.concat(output_slices, axis=-1)

            if normalizer_fn is not None:
                normalizer_params = normalizer_params or {}
                net = normalizer_fn(net, **normalizer_params)
            if activation_fn is not None:
                net = activation_fn(net)
            return net


def _channel_shuffle(inputs,
                     num_groups,
                     scope=None):
    if num_groups == 1:
        return inputs
    with tf.variable_scope(scope, 'Channel_Shuffle', [inputs]):
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        assert depth_in % num_groups == 0, (
                "depth_in=%d is not divisible by num_groups=%d" %
                (depth_in, num_groups))
        # group size, depth = g * n
        group_size = depth_in // num_groups
        net = inputs
        net_shape = net.shape.as_list()
        # reshape to (b, h, w, g, n)
        net = tf.reshape(net, net_shape[:3] + [num_groups, group_size])
        # transpose to (b, h, w, n, g)
        net = tf.transpose(net, [0, 1, 2, 4, 3])
        # reshape back to (b, h, w, depth)
        net = tf.reshape(net, net_shape)
        return net
