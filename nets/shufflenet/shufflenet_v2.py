"""ShuffleNet v2.

ShuffleNet v2, which is designed specially
for mobile devices with very limited computing power (e.g.,10-150 MFLOPs)

As described in https://arxiv.org/pdf/1807.11164.pdf.

    ShuffleNet V2: Practical Guidelines for
                        Efficient CNN Architecture Design
    Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun

100% Shufflenet V2 (base) with input size 224x224:

See shufflenet_v2()

+------------------------------------------+------------------+
|        Layer                             |      params      |
+------------------------------------------+------------------+
|        ShufflenetV2/Conv2d_0             |         696      |
|        ShufflenetV2/MaxPool2d_0          |           0      |
|        ShufflenetV2/Stage_0/Unit_0       |       7.40k      |
|        ShufflenetV2/Stage_0/Unit_1       |       7.60k      |
|        ShufflenetV2/Stage_0/Unit_2       |       7.60k      |
|        ShufflenetV2/Stage_0/Unit_3       |       7.60k      |
|        ShufflenetV2/Stage_1/Unit_0       |      43.62k      |
|        ShufflenetV2/Stage_1/Unit_1       |      28.65k      |
|        ShufflenetV2/Stage_1/Unit_2       |      28.65k      |
|        ShufflenetV2/Stage_1/Unit_3       |      28.65k      |
|        ShufflenetV2/Stage_1/Unit_4       |      28.65k      |
|        ShufflenetV2/Stage_1/Unit_5       |      28.65k      |
|        ShufflenetV2/Stage_1/Unit_6       |      28.65k      |
|        ShufflenetV2/Stage_1/Unit_7       |      28.65k      |
|        ShufflenetV2/Stage_2/Unit_0       |     167.97k      |
|        ShufflenetV2/Stage_2/Unit_1       |     111.13k      |
|        ShufflenetV2/Stage_2/Unit_2       |     111.13k      |
|        ShufflenetV2/Stage_2/Unit_3       |     111.13k      |
|        ShufflenetV2/Conv2d_1             |     477.18k      |
+------------------------------------------+------------------+
|               Total:                     |       1.25m      |
+------------------------------------------+------------------+
|               Flops:                     |   1,449,901,433  |
+------------------------------------------+------------------+


50% Shufflenet V2 (base) with input size 224x224:

See shufflenet_v2()

+------------------------------------------+------------------+
|        Layer                             |      params      |
+------------------------------------------+------------------+
|        ShufflenetV2/Conv2d_0             |         696      |
|        ShufflenetV2/MaxPool2d_0          |           0      |
|        ShufflenetV2/Stage_0/Unit_0       |       2.40k      |
|        ShufflenetV2/Stage_0/Unit_1       |       1.51k      |
|        ShufflenetV2/Stage_0/Unit_2       |       1.51k      |
|        ShufflenetV2/Stage_0/Unit_3       |       1.51k      |
|        ShufflenetV2/Stage_1/Unit_0       |       8.26k      |
|        ShufflenetV2/Stage_1/Unit_1       |       5.33k      |
|        ShufflenetV2/Stage_1/Unit_2       |       5.33k      |
|        ShufflenetV2/Stage_1/Unit_3       |       5.33k      |
|        ShufflenetV2/Stage_1/Unit_4       |       5.33k      |
|        ShufflenetV2/Stage_1/Unit_5       |       5.33k      |
|        ShufflenetV2/Stage_1/Unit_6       |       5.33k      |
|        ShufflenetV2/Stage_1/Unit_7       |       5.33k      |
|        ShufflenetV2/Stage_2/Unit_0       |      30.34k      |
|        ShufflenetV2/Stage_2/Unit_1       |      19.87k      |
|        ShufflenetV2/Stage_2/Unit_2       |      19.87k      |
|        ShufflenetV2/Stage_2/Unit_3       |      19.87k      |
|        ShufflenetV2/Conv2d_1             |     198.66k      |
+------------------------------------------+------------------+
|               Total:                     |     341.79k      |
+------------------------------------------+------------------+
|               Flops:                     |    401,081,945   |
+------------------------------------------+------------------+


150% Shufflenet V2 (base) with input size 224x224:

See shufflenet_v2()

+------------------------------------------+------------------+
|        Layer                             |      params      |
+------------------------------------------+------------------+
|        ShufflenetV2/Conv2d_0             |         696      |
|        ShufflenetV2/MaxPool2d_0          |           0      |
|        ShufflenetV2/Stage_0/Unit_0       |      13.73k      |
|        ShufflenetV2/Stage_0/Unit_1       |      16.81k      |
|        ShufflenetV2/Stage_0/Unit_2       |      16.81k      |
|        ShufflenetV2/Stage_0/Unit_3       |      16.81k      |
|        ShufflenetV2/Stage_1/Unit_0       |      97.86k      |
|        ShufflenetV2/Stage_1/Unit_1       |      64.59k      |
|        ShufflenetV2/Stage_1/Unit_2       |      64.59k      |
|        ShufflenetV2/Stage_1/Unit_3       |      64.59k      |
|        ShufflenetV2/Stage_1/Unit_4       |      64.59k      |
|        ShufflenetV2/Stage_1/Unit_5       |      64.59k      |
|        ShufflenetV2/Stage_1/Unit_6       |      64.59k      |
|        ShufflenetV2/Stage_1/Unit_7       |      64.59k      |
|        ShufflenetV2/Stage_2/Unit_0       |      30.34k      |
|        ShufflenetV2/Stage_2/Unit_1       |      19.87k      |
|        ShufflenetV2/Stage_2/Unit_2       |      19.87k      |
|        ShufflenetV2/Stage_2/Unit_3       |      19.87k      |
|        ShufflenetV2/Conv2d_1             |     722.94k      |
+------------------------------------------+------------------+
|               Total:                     |       2.48m      |
+------------------------------------------+------------------+
|               Flops:                     |   2,964,491,993  |
+------------------------------------------+------------------+


200% Shufflenet V2 (base) with input size 224x224:

See shufflenet_v2()

+------------------------------------------+------------------+
|        Layer                             |      params      |
+------------------------------------------+------------------+
|        ShufflenetV2/Conv2d_0             |         696      |
|        ShufflenetV2/MaxPool2d_0          |           0      |
|        ShufflenetV2/Stage_0/Unit_0       |      23.08k      |
|        ShufflenetV2/Stage_0/Unit_1       |      31.60k      |
|        ShufflenetV2/Stage_0/Unit_2       |      31.60k      |
|        ShufflenetV2/Stage_0/Unit_3       |      31.60k      |
|        ShufflenetV2/Stage_1/Unit_0       |     185.44k      |
|        ShufflenetV2/Stage_1/Unit_1       |     122.73k      |
|        ShufflenetV2/Stage_1/Unit_2       |     122.73k      |
|        ShufflenetV2/Stage_1/Unit_3       |     122.73k      |
|        ShufflenetV2/Stage_1/Unit_4       |     122.73k      |
|        ShufflenetV2/Stage_1/Unit_5       |     122.73k      |
|        ShufflenetV2/Stage_1/Unit_6       |     122.73k      |
|        ShufflenetV2/Stage_1/Unit_7       |     122.73k      |
|        ShufflenetV2/Stage_2/Unit_0       |     728.10k      |
|        ShufflenetV2/Stage_2/Unit_1       |     483.61k      |
|        ShufflenetV2/Stage_2/Unit_2       |     483.61k      |
|        ShufflenetV2/Stage_2/Unit_3       |     483.61k      |
|        ShufflenetV2/Conv2d_1             |       2.00m      |
+------------------------------------------+------------------+
|               Total:                     |       5.34m      |
+------------------------------------------+------------------+
|               Flops:                     |   5,843,465,465  |
+------------------------------------------+------------------+

"""

# Tensorflow mandates this
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from collections import namedtuple
import tensorflow as tf

# slim = tf.contrib.slim
import tensorflow.contrib.slim as slim
from nets.shufflenet.shufflenet_utils import _channel_shuffle, _reduced_kernel_size_for_small_input

Stage = namedtuple('Stage', ['scope', 'unit_fn', 'args'])

# shufflenet v2 depths dictionary
depths_dict = {0.5: (48, 96, 192, 1024),
               1.0: (116, 232, 464, 1024),
               1.5: (176, 352, 704, 1024),
               2.0: (244, 488, 976, 2048)}


@slim.add_arg_scope
def unit_fn(inputs, depth, stride,
            groups, left_ratio,
            spatial_down=False,
            rate=1):
    """

    Args:
        inputs:
        depth:
        stride:
        groups:
        left_ratio:
        spatial_down:
        rate:

    Returns:

    """
    if spatial_down:
        if depth % 2 != 0:
            raise ValueError("depth must be divided by 2, but found '%d'." % depth)
        depth //= 2

        # construct left branch
        # 3x3 depthwise conv. By passing filters=None
        # separable_conv2d produces only a depthwise convolution layer
        shortcut = slim.separable_conv2d(inputs, None, [3, 3],
                                         depth_multiplier=1,
                                         stride=stride,
                                         rate=rate,
                                         padding='SAME',
                                         activation_fn=None)
        # 1x1 conv
        shortcut = slim.conv2d(
            shortcut, depth, 1, stride=1, padding='SAME')
        residual = inputs
    else:
        depth_left = int(depth * left_ratio)  # left branch depth
        depth = depth - depth_left  # right branch depth
        shortcut, residual = tf.split(inputs, [depth_left, depth], axis=-1)
    # construct right branch
    # 1x1 conv
    residual = slim.conv2d(residual, depth, 1, stride=1, padding='SAME')
    # 3x3 depthwise conv. By passing filters=None
    # separable_conv2d produces only a depthwise convolution layer
    residual = slim.separable_conv2d(residual, None, [3, 3],
                                     depth_multiplier=1,
                                     stride=stride,
                                     rate=rate,
                                     activation_fn=None)
    # 1x1 conv
    residual = slim.conv2d(residual, depth, 1, stride=1, padding='SAME')
    # concat two branches
    output = tf.concat([shortcut, residual], axis=-1)
    # channel shuflle
    output = _channel_shuffle(output, num_groups=groups)
    return output


def stage(scope, depth, num_units, stride, groups=2, left_ratio=0.5):
    """Helper function for creating a shufflenet v2 bottleneck block.

        Args:
          scope: The scope of the block.
          depth: The depth of the bottleneck layer for each unit.
          num_units: The number of units in the block.
          stride: The stride of the block, implemented as a stride in the last unit.
            All other units have stride=1.
          groups: number of groups for each unit except the first unit.
          left_ratio: split channel ratio for left branch.

        Returns:
          A shufflenet v2 bottleneck block.
    """
    args = []

    for i in range(num_units):
        arg = dict(
            scope='Unit_%d' % i,
            depth=depth,
            groups=groups,
            left_ratio=left_ratio,
        )
        if i == 0:
            arg.update({
                'stride': stride,
                'spatial_down': True
            })
        else:
            arg.update({
                'stride': 1,
                'spatial_down': False
            })
        args.append(arg)
    return Stage(scope, unit_fn, args)


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

                    unit_args['stride'] = unit_stride
                    net = stage.unit_fn(net, rate=unit_rate, **unit_args)

                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

    return net, end_points


def shufflenet_v2_base(inputs,
                       final_endpoint='',
                       depth_multiplier=1.0,
                       output_stride=None,
                       include_root_block=True,
                       reuse=None,
                       scope=None):
    """

    Args:
        inputs:
        final_endpoint:
        depth_multiplier:
        output_stride:
        include_root_block:
        reuse:
        scope:

    Returns:

    """
    if depth_multiplier not in list(depths_dict.keys()):
        raise ValueError("depth_multiplier must be one of 0.5, 1.0, 1.5, 2.0, but found '%d'." % depth_multiplier)

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError("Only allowed output_stride values are 8, 16, 32, but found '%d'" % output_stride)

    depths = depths_dict[depth_multiplier]
    end_points = {}
    stages = [
        stage('Stage_0', depth=depths[0], num_units=4,
              stride=2, groups=2, left_ratio=0.5),
        stage('Stage_1', depth=depths[1], num_units=8,
              stride=2, groups=2, left_ratio=0.5),
        stage('Stage_2', depth=depths[2], num_units=4,
              stride=2, groups=2, left_ratio=0.5),
    ]
    with tf.variable_scope(scope, 'ShufflenetV2', [inputs], reuse=reuse) as sc:
        net = inputs
        if include_root_block:
            if output_stride is not None:
                if output_stride % 4 != 0:
                    raise ValueError(
                        'The output_stride needs to be a multiple of 4.')
                output_stride /= 4

            end_point = 'Conv2d_0'
            net = slim.conv2d(net, 24, [3, 3], stride=2,
                              padding='SAME', scope='Conv2d_0')
            end_points[end_point] = net
            if final_endpoint == end_point:
                return net, end_points

            end_point = 'MaxPool2d_0'
            net = slim.max_pool2d(net, [3, 3], stride=2,
                                  padding='SAME', scope='MaxPool2d_0')
            end_points[end_point] = net
            if final_endpoint == end_point:
                return net, end_points
        else:
            height = inputs.get_shape()[1]
            stride = 2 if height > 32 else 1

            end_point = 'Conv2d_0'
            net = slim.conv2d(net, 24, [3, 3], stride=stride,
                              padding='SAME', scope='Conv2d_0')
            end_points[end_point] = net
            if final_endpoint == end_point:
                return net, end_points
        net, stage_end_points = stack_stages(net, stages, output_stride, final_endpoint)
        end_points.update(stage_end_points)
        # Return before Conv2d_1
        if "Stage" in final_endpoint:
            return net, end_points

        # final 1x1 conv
        end_point = 'Conv2d_1'
        net = slim.conv2d(net, depths[-1], 1,
                          stride=1, scope='Conv2d_1')
        end_points[end_point] = net
        if final_endpoint == end_point:
            return net, end_points

        return net, end_points


shufflenet_v2_base.default_image_size = 224


def shufflenet_v2(inputs,
                  num_classes=1000,
                  dropout_keep_prob=0.999,
                  is_training=True,
                  depth_multiplier=1.0,
                  prediction_fn=slim.softmax,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='ShufflenetV2',
                  global_pool=False, ):
    """Shufflenet v2 model for classification.

    Args:
        inputs:
        num_classes:
        dropout_keep_prob:
        is_training:
        depth_multiplier:
        prediction_fn:
        spatial_squeeze:
        reuse:
        scope:
        global_pool:

    Returns:

    """

    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError("Invalid input tensor rank, expected 4, was: '%d'." %
                         len(input_shape))
    with tf.variable_scope(scope, 'ShufflenetV2', [inputs], reuse=reuse)as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            net, end_points = shufflenet_v2_base(inputs,
                                                 depth_multiplier=depth_multiplier,
                                                 include_root_block=True,
                                                 reuse=reuse,
                                                 scope=scope)
            with tf.variable_scope('Logits'):
                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(
                        net, [1, 2], name='global_pool', keepdims=True)
                    end_points['global_pool'] = net
                else:
                    # Pooling with fixed kernel size
                    kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
                    net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                          scope='AvgPool_1a')
                    end_points['AvgPool_1a'] = net

                if not num_classes:
                    return net, end_points

                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout')
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            end_points['Logits'] = logits

            if prediction_fn:
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

        return logits, end_points


shufflenet_v2.default_image_size = shufflenet_v2_base.default_image_size


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func


shufflenet_v2_050 = wrapped_partial(shufflenet_v2, depth_multiplier=0.5)
shufflenet_v2_100 = wrapped_partial(shufflenet_v2, depth_multiplier=1.0)
shufflenet_v2_150 = wrapped_partial(shufflenet_v2, depth_multiplier=1.5)
shufflenet_v2_200 = wrapped_partial(shufflenet_v2, depth_multiplier=2.0)


def shufflenet_v2_arg_scope(is_training=True,
                            weight_decay=0.00004,
                            stddev=0.09,
                            regularize_depthwise=False,
                            batch_norm_decay=0.9997,
                            batch_norm_epsilon=0.001,
                            batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
                            normalizer_fn=slim.batch_norm):
    """Defines the default ShufflenetV2 arg scope.

    Args:
      is_training: Whether or not we're training the model. If this is set to
        None, the parameter is not added to the batch_norm arg_scope.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.
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

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=normalizer_fn):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer) as sc:
                    return sc
