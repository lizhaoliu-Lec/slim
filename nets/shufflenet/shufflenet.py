# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.shufflenet.shufflenet_utils import group_conv2d, _channel_shuffle


@slim.add_arg_scope
def shufflenet_unit(inputs,
                    depth,
                    depth_bottleneck,
                    stride,
                    groups,
                    rate=1,
                    outputs_collections=None,
                    scope=None,
                    use_bounded_activations=False):
    with tf.variable_scope(scope, 'shufflenet', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if stride == 2:
            ratio = depth // depth_bottleneck
            depth -= depth_in
            depth_bottleneck = depth // ratio
            depth = depth_bottleneck * ratio

        # 1x1 group conv
        residual = group_conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                                num_groups=groups, scope='group_conv1')
        # channel shuffle
        residual = _channel_shuffle(residual, groups, 'channel_shuffle')
        # 3x3 depthwise conv. By passing filters=None
        # separable_conv2d produces only a depthwise convolution layer
        residual = slim.separable_conv2d(residual, None, [3, 3],
                                         depth_multiplier=1,
                                         stride=stride,
                                         rate=rate,
                                         activation_fn=None,
                                         scope='depthwise_conv')
        residual = group_conv2d(residual, depth, [1, 1], stride=1,
                                num_groups=groups, activation_fn=None,
                                scope='group_conv2')
        if stride == 1:
            shortcut = inputs
            output = tf.nn.relu(shortcut + residual)
        else:
            shortcut = slim.avg_pool2d(inputs, [3, 3], stride=2, scope='pool1',
                                       padding='SAME')
            output = tf.nn.relu(tf.concat([shortcut, residual], axis=3))

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.original_name_scope,
                                                output)


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ShuffleNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ShuffleNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride, groups) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def shufflenet_block(scope, base_depth, num_units, stride, groups, groups_in=None):
    """Helper function for creating a shufflenet bottleneck block.

    Args:
      scope: The scope of the block.
      base_depth: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the block.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.
      groups_in: number of groups for first unit.
      groups: number of groups for each unit except the first unit.

    Returns:
      A shufflenet bottleneck block.
    """
    return Block(scope, shufflenet_unit, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride,
        'groups': groups if groups_in is None else groups_in
    }] + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1,
        'groups': groups
    }] * (num_units - 1))


@slim.add_arg_scope
def stack_blocks_dense(net,
                       blocks,
                       output_stride=None,
                       outputs_collections=None):
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride larger
    # than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError(
                        'The target output_stride cannot be reached.')

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(
                            net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)

                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
            net = slim.utils.collect_named_outputs(
                outputs_collections, sc.name, net)

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net


def shufflenet_base(inputs,
                    is_training=True,
                    dropout_keep_prob=None,
                    depth_multiplier=1.0,
                    min_depth=8,
                    groups=3,
                    bottlenet_compact_rate=4,
                    output_stride=None,
                    include_root_block=True,
                    reuse=None,
                    scope=None):
    """Shufflenet base."""
    Depth_Channels = {'1': [144, 288, 576], '2': [200, 400, 800], '3': [240, 480, 960],
                      '4': [272, 544, 1088], '8': [384, 768, 1536], }
    groups_str = str(groups)
    assert groups_str in ['1', '2', '3', '4', '8'], (
            'groups must be one of [1, 2, 3, 4, 8], your groups=%d' % groups)

    def depth_multi(d):
        return max(int(d * depth_multiplier), min_depth)

    depths = [depth_multi(depth) for depth in Depth_Channels[groups_str]]
    base_depths = [depth // bottlenet_compact_rate for depth in depths]
    blocks = [
        shufflenet_block('block1', base_depth=base_depths[0], num_units=4,
                         stride=2, groups=groups, groups_in=1),
        shufflenet_block('block2', base_depth=base_depths[1], num_units=8,
                         stride=2, groups=groups),
        shufflenet_block('block3', base_depth=base_depths[2], num_units=4,
                         stride=2, groups=groups),
    ]

    with tf.variable_scope(scope, 'Shufflenet', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, shufflenet_unit, stack_blocks_dense,
                             slim.separable_conv2d],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training,
                                dropout_keep_prop=dropout_keep_prob):
                end_points = {}
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError(
                                'The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    net = slim.conv2d(inputs, 24, 3, stride=2,
                                      padding='SAME', scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                else:
                    height = inputs.get_shape()[1]
                    stride = 2 if height > 32 else 1
                    net = slim.conv2d(net, 24, 3, stride=stride, scope='conv1')

                net = stack_blocks_dense(net, blocks, output_stride)

                # Convert end_points_collection into a dictionary of end_points.
                end_points.update(slim.utils.convert_collection_to_dict(
                    end_points_collection))

                return net, end_points


shufflenet_base.default_image_size = 224


def shufflenet_v1(inputs,
                  num_classes=1001,
                  prediction_fn=slim.softmax,
                  dropout_keep_prob=None,
                  is_training=False,
                  output_stride=None,
                  include_root_block=True,
                  reuse=None,
                  scope='Shufflenet'):
    """Shufflenet base."""
    with tf.variable_scope(scope, 'Shufflenet', reuse=reuse) as scope:
        net, end_points = shufflenet_base(inputs,
                                          is_training,
                                          dropout_keep_prob,
                                          output_stride=output_stride,
                                          include_root_block=include_root_block,
                                          reuse=reuse,
                                          scope=scope)

        with tf.variable_scope('Logits'):
            # Global average pooling.
            net = tf.reduce_mean(
                net, [1, 2], name='pool5', keepdims=True)

            end_points['global_pool'] = net
            if not num_classes:
                return net, end_points

            # 1 x 1 x num_classes
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                              normalizer_fn=None, scope='Conv2d_xxxxx_1x1')

            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

        end_points['Logits'] = net

        if prediction_fn:
            end_points['Predictions'] = prediction_fn(net, scope='Predictions')

        return net, end_points


shufflenet_v1.default_image_size = shufflenet_base.default_image_size


def shufflenet_v1_arg_scope(is_training=True,
                            weight_decay=0.00004,
                            stddev=0.09,
                            regularize_depthwise=False):
    """Defines the default Shufflenet_V1 arg scope.

    Args:
      is_training: Whether or not we're training the model.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.

    Returns:
      An `arg_scope` to use for the mobilenet v1 model.
    """
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.9997,
        'epsilon': 0.001,
    }

    if stddev < 0:
        weight_intitializer = slim.initializers.xavier_initializer()
    else:
        weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)

    # Set weight_decay for weights in Conv and DepthSepConv layers.

    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weight_intitializer,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm):
        with slim.arg_scope([group_conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm):
            with slim.arg_scope([slim.batch_norm],
                                **batch_norm_params):
                with slim.arg_scope([slim.conv2d],
                                    weights_regularizer=regularizer):
                    with slim.arg_scope([slim.separable_conv2d],
                                        weights_regularizer=depthwise_regularizer) as sc:
                        return sc
