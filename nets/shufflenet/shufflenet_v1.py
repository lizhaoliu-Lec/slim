"""ShuffleNet v1.

ShuffleNet, which is designed specially
for mobile devices with very limited computing power (e.g.,10-150 MFLOPs)

As described in https://arxiv.org/pdf/1707.01083.pdf.

  ShuffleNet: An Extremely Efficient Convolutional Neural Network
                    for Mobile Devices
  Xiangyu Zhang∗ Xinyu Zhou∗ Mengxiao Lin Jian Sun

100% Shufflenet V1 (group=3, base) with input size 224x224:

See shufflenet_v1()
+------------------------------------------+------------------+
|        Layer                             |      params      |
+------------------------------------------+------------------+
|        ShufflenetV1/Conv2d_0             |        696       |
|        ShufflenetV1/MaxPool2d_0          |          0       |
|        ShufflenetV1/Stage_0/Unit_0       |       6.97k      |
|        ShufflenetV1/Stage_0/Unit_1       |      10.86k      |
|        ShufflenetV1/Stage_0/Unit_2       |      10.86k      |
|        ShufflenetV1/Stage_0/Unit_3       |      10.86k      |
|        ShufflenetV1/Stage_1/Unit_0       |      21.24k      |
|        ShufflenetV1/Stage_1/Unit_1       |      40.92k      |
|        ShufflenetV1/Stage_1/Unit_2       |      40.92k      |
|        ShufflenetV1/Stage_1/Unit_3       |      40.92k      |
|        ShufflenetV1/Stage_1/Unit_4       |      40.92k      |
|        ShufflenetV1/Stage_1/Unit_5       |      40.92k      |
|        ShufflenetV1/Stage_1/Unit_6       |      40.92k      |
|        ShufflenetV1/Stage_1/Unit_7       |      40.92k      |
|        ShufflenetV1/Stage_2/Unit_0       |      80.88k      |
|        ShufflenetV1/Stage_2/Unit_1       |     158.64k      |
|        ShufflenetV1/Stage_2/Unit_2       |     158.64k      |
|        ShufflenetV1/Stage_2/Unit_3       |     158.64k      |
+------------------------------------------+------------------+
|               Total:                     |     904.73k      |
+------------------------------------------+------------------+
|               Flops:                     |   1,385,158,059  |
+------------------------------------------+------------------+


200% Shufflenet V1 (group=3, base) with input size 224x224:

See shufflenet_v1()
+------------------------------------------+------------------+
|        Layer                             |      params      |
+------------------------------------------+------------------+
|        ShufflenetV1/Conv2d_0             |        696       |
|        ShufflenetV1/MaxPool2d_0          |          0       |
|        ShufflenetV1/Stage_0/Unit_0       |      23.59k      |
|        ShufflenetV1/Stage_0/Unit_1       |      40.92k      |
|        ShufflenetV1/Stage_0/Unit_2       |      40.92k      |
|        ShufflenetV1/Stage_0/Unit_3       |      40.92k      |
|        ShufflenetV1/Stage_1/Unit_0       |      80.88k      |
|        ShufflenetV1/Stage_1/Unit_1       |     158.64k      |
|        ShufflenetV1/Stage_1/Unit_2       |     158.64k      |
|        ShufflenetV1/Stage_1/Unit_3       |     158.64k      |
|        ShufflenetV1/Stage_1/Unit_4       |     158.64k      |
|        ShufflenetV1/Stage_1/Unit_5       |     158.64k      |
|        ShufflenetV1/Stage_1/Unit_6       |     158.64k      |
|        ShufflenetV1/Stage_1/Unit_7       |     158.64k      |
|        ShufflenetV1/Stage_2/Unit_0       |       2.19m      |
|        ShufflenetV1/Stage_2/Unit_1       |     315.36k      |
|        ShufflenetV1/Stage_2/Unit_2       |     624.48k      |
|        ShufflenetV1/Stage_2/Unit_3       |     624.48k      |
+------------------------------------------+------------------+
|               Total:                     |       3.53m      |
+------------------------------------------+------------------+
|               Flops:                     |   5,100,328,779  |
+------------------------------------------+------------------+

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.shufflenet.shufflenet_utils import group_conv2d, \
    _channel_shuffle, _reduced_kernel_size_for_small_input

DEPTH_CHANNELS_DEFS = {
    '1': [144, 288, 576],
    '2': [200, 400, 800],
    '3': [240, 480, 960],
    '4': [272, 544, 1088],
    '8': [384, 768, 1536],
}


@slim.add_arg_scope
def unit_fn(inputs,
            depth,
            depth_bottleneck,
            stride,
            num_groups,
            rate=1,
            spatial_down=False,
            first_stage_first_unit=False):
    """

    Args:
        inputs:
        depth:
        depth_bottleneck:
        stride:
        num_groups:
        rate:
        spatial_down:
        first_stage_first_unit:

    Returns:

    """
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if spatial_down:
        depth -= depth_in

    # we round down the depth of bottlenet to be adaptive with group conv.
    if depth_bottleneck % num_groups != 0:
        depth_bottleneck = depth_bottleneck - depth_bottleneck % num_groups

    residual = group_conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                            num_groups=num_groups if not first_stage_first_unit else 1)

    # channel shuffle
    if not first_stage_first_unit:
        residual = _channel_shuffle(residual, num_groups)

    # 3 x 3 depthwise conv. By passing filters=None
    # separable_conv2d produces only a depthwise convolution layer
    residual = slim.separable_conv2d(residual,
                                     num_outputs=None,
                                     kernel_size=[3, 3],
                                     depth_multiplier=1,
                                     stride=stride,
                                     padding='SAME',
                                     rate=rate,
                                     activation_fn=None)

    residual = group_conv2d(residual, depth, [1, 1], stride=1,
                            num_groups=num_groups, activation_fn=None)

    if spatial_down:
        shortcut = slim.avg_pool2d(inputs, [3, 3], stride=stride,
                                   padding='SAME')
        output = tf.nn.relu(tf.concat([shortcut, residual], axis=3))

    else:
        shortcut = inputs
        output = tf.nn.relu(shortcut + residual)

    return output


# Shufflenet Stage.
# scope: The scope of the `Stage`.
# unit_fn: The ShuffleNet unit function which takes as input a `Tensor` and
#   returns another `Tensor` with the output of the ShuffleNet unit.
# args: A list of length equal to the number of units in the `Stage`. The list
#   contains one (depth, depth_bottleneck, stride, groups) tuple for each unit in the
#   Stage to serve as argument to unit_fn.
Stage = namedtuple('Stage', ['scope', 'unit_fn', 'args'])


def stage(scope,
          depth,
          bottlenet_compact_ratio,
          num_units,
          num_groups,
          stride,
          min_depth=None):
    """Helper function for creating a shufflenet.

    Args:
      scope:
      depth: The depth of the bottleneck layer for each unit.
      bottlenet_compact_ratio: float,
      num_units: The number of units in the stage.
      num_groups: number of groups for each unit except the first unit.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride = 1.
      min_depth: int,

    Returns:
      A shufflenet bottleneck Stage.
    """

    args = []

    for i in range(num_units):
        depth_bottleneck = max(int(bottlenet_compact_ratio * depth), min_depth) if min_depth else int(
            bottlenet_compact_ratio * depth)
        arg = dict(
            scope='Unit_%d' % i,
            depth=depth,
            depth_bottleneck=depth_bottleneck,
            num_groups=num_groups,
        )
        if i == 0:
            arg.update({
                'stride': stride,
                'spatial_down': True
            })
        else:
            arg.update({
                'stride': 1,
            })

        args.append(arg)

    return Stage(scope=scope, unit_fn=unit_fn, args=args)


@slim.add_arg_scope
def stack_stages(inputs,
                 stages,
                 output_stride=None,
                 final_endpoint='Stage_2/Unit_3'):
    """

    Args:
        inputs:
        stages:
        output_stride:
        final_endpoint:

    Returns:

    """
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride larger
    # than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    end_points = {}
    net = inputs
    for stage_idx, stage in enumerate(stages):
        with tf.variable_scope(stage.scope, 'Stage', [net]):

            for unit_idx, unit_args in enumerate(stage.args):
                with tf.variable_scope(unit_args['scope'], 'Unit', [net]):
                    unit_args.pop('scope')

                    end_point = 'Stage_%d/Unit_%d' % (stage_idx, unit_idx)
                    if output_stride is not None and current_stride == output_stride:
                        # If we have reached the target output_stride, then we need to employ
                        # atrous convolution with stride=1 and multiply the atrous rate by the
                        # current unit's stride for use in subsequent layers.
                        unit_stride = 1
                        unit_rate = rate
                        rate *= unit_args['stride']

                    else:
                        unit_stride = unit_args['stride']
                        unit_rate = 1
                        current_stride *= unit_stride

                    if not stage_idx and not unit_idx:
                        unit_args['first_stage_first_unit'] = True

                    unit_args['stride'] = unit_stride
                    net = stage.unit_fn(net, rate=unit_rate, **unit_args)

                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

    return net, end_points


def shufflenet_v1_base(inputs,
                       final_endpoint='Stage_2/Unit_3',
                       min_depth=8,
                       min_depth_constraint_bottlenet=True,
                       depth_multiplier=1.0,
                       depth_channels_defs=None,
                       num_groups=3,
                       bottlenet_compact_ratio=0.25,
                       output_stride=None,
                       scope=None):
    """

    Args:
        inputs:
        final_endpoint:
        min_depth:
        min_depth_constraint_bottlenet:
        depth_multiplier:
        depth_channels_defs:
        num_groups:
        bottlenet_compact_ratio:
        output_stride:
        scope:

    Returns:

    """

    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if depth_channels_defs is None:
        depth_channels_defs = DEPTH_CHANNELS_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('expect output_stride values: 8, 16, 32, but got `%d`' % output_stride)

    if str(num_groups) not in depth_channels_defs.keys():
        raise ValueError('expect num_groups in `%s`, but got `%s`' % (
            str(depth_channels_defs.keys()), str(num_groups)))

    depths = [depth(d) for d in depth_channels_defs[str(num_groups)]]

    if not _valid_depth(depths, num_groups):
        raise ValueError('depths `%s` is not a valid depths for group conv.' % str(depths))

    stages = [
        stage(scope='Stage_0',
              depth=depths[0],
              bottlenet_compact_ratio=bottlenet_compact_ratio,
              num_units=4,
              num_groups=num_groups,
              stride=2,
              min_depth=min_depth if min_depth_constraint_bottlenet else None),
        stage(scope='Stage_1',
              depth=depths[1],
              bottlenet_compact_ratio=bottlenet_compact_ratio,
              num_units=8,
              num_groups=num_groups,
              stride=2,
              min_depth=min_depth if min_depth_constraint_bottlenet else None),
        stage(scope='Stage_2',
              depth=depths[2],
              bottlenet_compact_ratio=bottlenet_compact_ratio,
              num_units=4,
              num_groups=num_groups,
              stride=2,
              min_depth=min_depth if min_depth_constraint_bottlenet else None),
    ]

    with tf.variable_scope(scope, 'ShufflenetV1', [inputs]):
        net = inputs

        end_point = 'Conv2d_0'
        net = slim.conv2d(net, 24, [3, 3], stride=2, padding='SAME',
                          biases_initializer=None,
                          scope='Conv2d_0')
        end_points[end_point] = net

        if final_endpoint == end_point:
            return net, end_points

        end_point = 'MaxPool2d_0'
        net = slim.max_pool2d(net, [3, 3], stride=2, padding='SAME',
                              scope='MaxPool2d_0')
        end_points[end_point] = net

        if final_endpoint == end_point:
            return net, end_points

        if output_stride:
            output_stride = output_stride // 4

        net, stage_end_points = stack_stages(net, stages, output_stride, final_endpoint)

        end_points.update(stage_end_points)

        return net, end_points


def shufflenet_v1(inputs,
                  num_classes=1000,
                  dropout_keep_prob=0.999,
                  is_training=True,
                  min_depth=8,
                  min_depth_constraint_bottlenet=True,
                  depth_multiplier=1.0,
                  depth_channels_defs=None,
                  num_groups=3,
                  prediction_fn=slim.softmax,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='ShufflenetV1',
                  global_pool=False):
    """

    Args:
        inputs:
        num_classes:
        dropout_keep_prob:
        is_training:
        min_depth:
        min_depth_constraint_bottlenet:
        depth_multiplier:
        depth_channels_defs:
        num_groups:
        prediction_fn:
        spatial_squeeze:
        reuse:
        scope:
        global_pool:

    Returns:

    """

    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))
    with tf.variable_scope(scope, 'ShufflenetV1', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):

            net, end_points = shufflenet_v1_base(inputs, scope=scope,
                                                 min_depth=min_depth,
                                                 min_depth_constraint_bottlenet=min_depth_constraint_bottlenet,
                                                 depth_multiplier=depth_multiplier,
                                                 depth_channels_defs=depth_channels_defs, num_groups=num_groups)

            with tf.variable_scope('Logits'):
                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
                    end_points['global_pool'] = net
                else:
                    # Pooling with fixed kernel size
                    kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
                    net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                          scope='AvgPool_1a')
                    end_points['AvgPool_1a'] = net

                if not num_classes:
                    return net, end_points
                # 1 x 1 x (576 | 800 | 960 | 1088 | 1536)
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')

                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            end_points['Logits'] = logits

            if prediction_fn:
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

        return logits, end_points


shufflenet_v1.default_image_size = 224


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


shufflenet_v1_025_1 = wrapped_partial(shufflenet_v1, depth_multiplier=0.25, num_groups=1)
shufflenet_v1_025_2 = wrapped_partial(shufflenet_v1, depth_multiplier=0.25, num_groups=2)
shufflenet_v1_025_3 = wrapped_partial(shufflenet_v1, depth_multiplier=0.25, num_groups=3)
shufflenet_v1_025_4 = wrapped_partial(shufflenet_v1, depth_multiplier=0.25, num_groups=4)
shufflenet_v1_025_8 = wrapped_partial(shufflenet_v1, depth_multiplier=0.25, num_groups=8)

shufflenet_v1_050_1 = wrapped_partial(shufflenet_v1, depth_multiplier=0.5, num_groups=1)
shufflenet_v1_050_2 = wrapped_partial(shufflenet_v1, depth_multiplier=0.5, num_groups=2)
shufflenet_v1_050_3 = wrapped_partial(shufflenet_v1, depth_multiplier=0.5, num_groups=3)
shufflenet_v1_050_4 = wrapped_partial(shufflenet_v1, depth_multiplier=0.5, num_groups=4)
shufflenet_v1_050_8 = wrapped_partial(shufflenet_v1, depth_multiplier=0.5, num_groups=8)

shufflenet_v1_100_1 = wrapped_partial(shufflenet_v1, depth_multiplier=1.0, num_groups=1)
shufflenet_v1_100_2 = wrapped_partial(shufflenet_v1, depth_multiplier=1.0, num_groups=2)
shufflenet_v1_100_3 = wrapped_partial(shufflenet_v1, depth_multiplier=1.0, num_groups=3)
shufflenet_v1_100_4 = wrapped_partial(shufflenet_v1, depth_multiplier=1.0, num_groups=4)
shufflenet_v1_100_8 = wrapped_partial(shufflenet_v1, depth_multiplier=1.0, num_groups=8)

shufflenet_v1_150_1 = wrapped_partial(shufflenet_v1, depth_multiplier=1.5, num_groups=1)
shufflenet_v1_150_2 = wrapped_partial(shufflenet_v1, depth_multiplier=1.5, num_groups=2)
shufflenet_v1_150_3 = wrapped_partial(shufflenet_v1, depth_multiplier=1.5, num_groups=3)
shufflenet_v1_150_4 = wrapped_partial(shufflenet_v1, depth_multiplier=1.5, num_groups=4)
shufflenet_v1_150_8 = wrapped_partial(shufflenet_v1, depth_multiplier=1.5, num_groups=8)

shufflenet_v1_200_1 = wrapped_partial(shufflenet_v1, depth_multiplier=2.0, num_groups=1)
shufflenet_v1_200_2 = wrapped_partial(shufflenet_v1, depth_multiplier=2.0, num_groups=2)
shufflenet_v1_200_3 = wrapped_partial(shufflenet_v1, depth_multiplier=2.0, num_groups=3)
shufflenet_v1_200_4 = wrapped_partial(shufflenet_v1, depth_multiplier=2.0, num_groups=4)
shufflenet_v1_200_8 = wrapped_partial(shufflenet_v1, depth_multiplier=2.0, num_groups=8)


def _valid_depth(depths, num_groups):
    """

    Args:
        depths:
        num_groups:

    Returns:

    """
    depths_length = len(depths)
    if depths_length != 3:
        raise ValueError('expect depths length of `3`, but got `%d`' % depths_length)
    for i in range(depths_length):
        depth = depths[i]
        if depth % num_groups != 0:
            print('depth % num_groups != 0')
            return False
        if not i:
            if (depths[i] - depths[i - 1]) % num_groups != 0:
                return False
    return True


def shufflenet_v1_arg_scope(
        is_training=True,
        weight_decay=0.00004,
        stddev=0.09,
        regularize_depthwise=False,
        regularize_group_conv=True,
        batch_norm_decay=0.9997,
        batch_norm_epsilon=0.001,
        batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
        normalizer_fn=slim.batch_norm):
    """Defines the default ShufflenetV1 arg scope.

    Args:
      is_training: Whether or not we're training the model. If this is set to
        None, the parameter is not added to the batch_norm arg_scope.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.
      regularize_group_conv: Whether or not apply regularization on group conv.
      batch_norm_decay: Decay for batch norm moving average.
      batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm.
      batch_norm_updates_collections: Collection for the update ops for
        batch norm.
      normalizer_fn: Normalization function to apply after convolution.

    Returns:
      An `arg_scope` to use for the shufflenet v1 model.
    """
    batch_norm_params = {
        'center': True,
        'scale': True,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'updates_collections': batch_norm_updates_collections,
    }
    if is_training is not None:
        batch_norm_params['is_training'] = is_training

    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None

    if regularize_group_conv:
        group_conv_regularizer = regularizer
    else:
        group_conv_regularizer = None

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, group_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6, normalizer_fn=normalizer_fn):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer):
                    with slim.arg_scope([group_conv2d],
                                        weights_regularizer=group_conv_regularizer) as sc:
                        return sc
