# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.shufflenet.shufflenet_utils import group_conv2d, _channel_shuffle

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
            reached_output_stride=False):
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if stride == 2 or reached_output_stride:
        ratio = depth // depth_bottleneck
        depth -= depth_in
        depth_bottleneck = depth // ratio
        depth = depth_bottleneck * ratio

    # 1 x 1 group conv
    residual = group_conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                            num_groups=num_groups)
    # channel shuffle
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

    if stride == 2 or reached_output_stride:
        if reached_output_stride:
            pool_stride = 1
        else:
            pool_stride = 2
        shortcut = slim.avg_pool2d(inputs, [3, 3], stride=pool_stride,
                                   padding='SAME')
        output = tf.nn.relu(tf.concat([shortcut, residual], axis=3))

    elif stride == 1:
        shortcut = inputs
        output = tf.nn.relu(shortcut + residual)

    else:
        raise ValueError('expect stride of 1 or 2, but got `%d`' % stride)

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
          depth_bottleneck,
          num_units,
          num_groups,
          stride,
          num_groups_in=None):
    """Helper function for creating a shufflenet.

    Args:
      scope:
      depth_bottleneck: The depth of the bottleneck layer for each unit.
      num_units: The number of units in the stage.
      num_groups: number of groups for each unit except the first unit.
      stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride = 1.
      num_groups_in: number of groups for first unit.

    Returns:
      A shufflenet bottleneck Stage.
    """

    args = []

    for i in range(num_units):
        arg = dict(
            scope='Unit_%d' % i,
            depth=depth_bottleneck * 4,
            depth_bottleneck=depth_bottleneck,
        )
        if i == 0:
            arg.update({
                'stride': stride,
                'num_groups': num_groups if num_groups_in is None else num_groups_in
            })
        else:
            arg.update({
                'stride': 1,
                'num_groups': num_groups
            })

        args.append(arg)

    return Stage(scope=scope,
                 unit_fn=unit_fn,
                 args=args)


@slim.add_arg_scope
def stack_stages(inputs,
                 stages,
                 output_stride=None,
                 # TODO (lizhao liu)
                 # configure endpoint correctly
                 final_endpoint='xxxxxxxxxx'):
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

                        if unit_args['stride'] == 2:
                            unit_args['reached_output_stride'] = True

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


def shufflenet_base(inputs,
                    # TODO (lizhao liu)
                    # configure endpoint correctly
                    final_endpoint='xxxxxxxxxxxx',
                    min_depth=8,
                    depth_multiplier=1.0,
                    depth_channels_defs=None,
                    num_groups=3,
                    bottlenet_compact_ratio=0.25,
                    output_stride=None,
                    scope=None):
    """Shufflenet base."""
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if depth_channels_defs is None:
        depth_channels_defs = DEPTH_CHANNELS_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    if str(num_groups) not in depth_channels_defs.keys():
        raise ValueError('expect num_groups in `%s`, but got `%s`' % (
            str(depth_channels_defs.keys()), str(num_groups)))

    depths = [depth(d) for d in depth_channels_defs[str(num_groups)]]

    if len(depths) != 3:
        raise ValueError('expect `3` depths for `%d` groups, but got `%s`' % (
            num_groups, str(depth_channels_defs[str(num_groups)])))

    bottleneck_depths = [int(d * bottlenet_compact_ratio) for d in depths]

    stages = [
        stage(scope='Stage_0',
              depth_bottleneck=bottleneck_depths[0],
              num_units=4,
              num_groups=num_groups,
              stride=2,
              num_groups_in=1),
        stage(scope='Stage_1',
              depth_bottleneck=bottleneck_depths[1],
              num_units=8,
              stride=2,
              num_groups=num_groups),
        stage(scope='Stage_2',
              depth_bottleneck=bottleneck_depths[2],
              num_units=4,
              stride=2,
              num_groups=num_groups),
    ]

    with tf.variable_scope(scope, 'ShufflenetV1', [inputs]):
        net = inputs

        end_point = 'Conv2d_0'
        net = slim.conv2d(net, 24, [3, 3], stride=2, padding='SAME',
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
                  depth_multiplier=1.0,
                  depth_channels_defs=None,
                  prediction_fn=slim.softmax,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='ShufflenetV1',
                  global_pool=False):
    """Shufflenet v1 model for classification."""

    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))
    with tf.variable_scope(scope, 'ShufflenetV1', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):

            net, end_points = shufflenet_base(inputs, scope=scope,
                                              min_depth=min_depth,
                                              depth_multiplier=depth_multiplier,
                                              depth_channels_defs=depth_channels_defs)

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


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.

    If the shape of the input images is unknown at graph construction time this
    function assumes that the input images are large enough.

    Args:
      input_tensor: input tensor of size [batch_size, height, width, channels].
      kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

    Returns:
      a tensor with the kernel size.
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]

    return kernel_size_out


def shufflenet_v1_arg_scope(
        is_training=True,
        weight_decay=0.00004,
        stddev=0.09,
        regularize_depthwise=False,
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

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d, group_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer) as sc:
                    return sc
