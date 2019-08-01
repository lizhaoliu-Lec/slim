# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.nets.shufflenet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from nets.shufflenet import shufflenet
from nets.shufflenet.shufflenet_utils import group_conv2d


class ShuffleNetV1Test(tf.test.TestCase):

    def testBuild(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            logits, _ = shufflenet.shufflenet_v1(inputs, num_classes)
            self.assertListEqual(logits.get_shape().as_list(),
                                 [batch_size, num_classes])

    def testBuildClassificationNetwork(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = shufflenet.shufflenet_v1(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith(
            'ShufflenetV1/Logits/SpatialSqueeze'))
        self.assertListEqual(logits.get_shape().as_list(),
                             [batch_size, num_classes])
        self.assertTrue('Predictions' in end_points)
        self.assertListEqual(end_points['Predictions'].get_shape().as_list(),
                             [batch_size, num_classes])

    def testBuildPreLogitsNetwork(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = None

        inputs = tf.random_uniform((batch_size, height, width, 3))
        net, end_points = shufflenet.shufflenet_v1(inputs, num_classes)
        self.assertTrue(net.op.name.startswith('ShufflenetV1/Logits/AvgPool'))
        self.assertListEqual(net.get_shape().as_list(), [batch_size, 1, 1, 960])
        self.assertFalse('Logits' in end_points)
        self.assertFalse('Predictions' in end_points)

    def testBuildBaseNetwork(self):
        batch_size = 5
        height, width = 224, 224

        inputs = tf.random_uniform((batch_size, height, width, 3))
        net, endpoints = shufflenet.shufflenet_base(inputs)
        print(net.op.name)
        # self.assertTrue(net.op.name.startswith('ShufflenetV1/stage3/shufflenet_unit_3'))
        self.assertListEqual(net.get_shape().as_list(),
                             [batch_size, 7, 7, 960])

        expected_endpoints = [
            # regular conv and pool
            'Conv2d_0', 'MaxPool2d_0',
            # Stage 1 with 4 units
            'Stage_0/Unit_0', 'Stage_0/Unit_1',
            'Stage_0/Unit_2', 'Stage_0/Unit_3',
            # Stage 2 with 8 units
            'Stage_1/Unit_0', 'Stage_1/Unit_1',
            'Stage_1/Unit_2', 'Stage_1/Unit_3',
            'Stage_1/Unit_4', 'Stage_1/Unit_5',
            'Stage_1/Unit_6', 'Stage_1/Unit_7',
            # Stage 3 with 4 units
            'Stage_2/Unit_0', 'Stage_2/Unit_1',
            'Stage_2/Unit_2', 'Stage_2/Unit_3',
        ]
        self.maxDiff = None
        self.assertItemsEqual(endpoints.keys(), expected_endpoints)

    def testBuildOnlyUptoFinalEndpoint(self):
        batch_size = 5
        height, width = 224, 224
        endpoints = [
            # regular conv and pool
            'Conv2d_0', 'MaxPool2d_0',
            # Stage 1 with 4 units
            'Stage_0/Unit_0', 'Stage_0/Unit_1',
            'Stage_0/Unit_2', 'Stage_0/Unit_3',
            # Stage 2 with 8 units
            'Stage_1/Unit_0', 'Stage_1/Unit_1',
            'Stage_1/Unit_2', 'Stage_1/Unit_3',
            'Stage_1/Unit_4', 'Stage_1/Unit_5',
            'Stage_1/Unit_6', 'Stage_1/Unit_7',
            # Stage 3 with 4 units
            'Stage_2/Unit_0', 'Stage_2/Unit_1',
            'Stage_2/Unit_2', 'Stage_2/Unit_3',
        ]
        for index, endpoint in enumerate(endpoints):
            with tf.Graph().as_default():
                inputs = tf.random_uniform((batch_size, height, width, 3))
                out_tensor, end_points = shufflenet.shufflenet_base(inputs, final_endpoint=endpoint)
                print(out_tensor.op.name)
                self.assertTrue(out_tensor.op.name.startswith(
                    'ShufflenetV1/' + endpoint))
                self.assertItemsEqual(endpoints[:index + 1], end_points.keys())

    def testBuildCustomNetworkUsingDepthChannelsDefs(self):
        batch_size = 5
        height, width = 224, 224
        depth_channels_defs = {
            '1': [144, 288, 576],
            '2': [200, 400, 800],
            '3': [240, 960, 1024],
            '4': [272, 544, 1088],
            '8': [384, 768, 1536],
        }

        inputs = tf.random_uniform((batch_size, height, width, 3))
        net, end_points = shufflenet.shufflenet_base(
            inputs, final_endpoint='Stage_1/Unit_7', depth_channels_defs=depth_channels_defs)
        self.assertTrue(net.op.name.startswith('ShufflenetV1/Stage_1/Unit_7'))
        self.assertListEqual(net.get_shape().as_list(),
                             [batch_size, 14, 14, 960])
        expected_endpoints = [
            # regular conv and pool
            'Conv2d_0', 'MaxPool2d_0',
            # Stage 1 with 4 units
            'Stage_0/Unit_0', 'Stage_0/Unit_1',
            'Stage_0/Unit_2', 'Stage_0/Unit_3',
            # Stage 2 with 8 units
            'Stage_1/Unit_0', 'Stage_1/Unit_1',
            'Stage_1/Unit_2', 'Stage_1/Unit_3',
            'Stage_1/Unit_4', 'Stage_1/Unit_5',
            'Stage_1/Unit_6', 'Stage_1/Unit_7', ]
        self.assertItemsEqual(end_points.keys(), expected_endpoints)

    def testBuildAndCheckAllEndPointsUptoStage_2_Sub_Unit_3(self):
        batch_size = 5
        height, width = 224, 224

        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            _, end_points = shufflenet.shufflenet_base(
                inputs, final_endpoint='Stage_2/Unit_3')

        endpoints_shapes = {
            # regular conv and pool
            'Conv2d_0': [batch_size, 112, 112, 24],
            'MaxPool2d_0': [batch_size, 56, 56, 24],
            # Stage 1 with 4 units
            'Stage_0/Unit_0': [batch_size, 28, 28, 240],
            'Stage_0/Unit_1': [batch_size, 28, 28, 240],
            'Stage_0/Unit_2': [batch_size, 28, 28, 240],
            'Stage_0/Unit_3': [batch_size, 28, 28, 240],
            # Stage 2 with 8 units
            'Stage_1/Unit_0': [batch_size, 14, 14, 480],
            'Stage_1/Unit_1': [batch_size, 14, 14, 480],
            'Stage_1/Unit_2': [batch_size, 14, 14, 480],
            'Stage_1/Unit_3': [batch_size, 14, 14, 480],
            'Stage_1/Unit_4': [batch_size, 14, 14, 480],
            'Stage_1/Unit_5': [batch_size, 14, 14, 480],
            'Stage_1/Unit_6': [batch_size, 14, 14, 480],
            'Stage_1/Unit_7': [batch_size, 14, 14, 480],
            # Stage 3 with 4 units
            'Stage_2/Unit_0': [batch_size, 7, 7, 960],
            'Stage_2/Unit_1': [batch_size, 7, 7, 960],
            'Stage_2/Unit_2': [batch_size, 7, 7, 960],
            'Stage_2/Unit_3': [batch_size, 7, 7, 960],
        }

        self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
        for endpoint_name, expected_shape in endpoints_shapes.items():
            self.assertTrue(endpoint_name in end_points)
            self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                                 expected_shape)

    def testOutputStride16BuildAndCheckAllEndPointsUptoStage_2_Sub_Unit_3(self):
        batch_size = 5
        height, width = 224, 224
        output_stride = 16

        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            _, end_points = shufflenet.shufflenet_base(
                inputs, output_stride=output_stride,
                final_endpoint='Stage_2/Unit_3')

        endpoints_shapes = {
            # regular conv and pool
            'Conv2d_0': [batch_size, 112, 112, 24],
            'MaxPool2d_0': [batch_size, 56, 56, 24],
            # Stage 1 with 4 units
            'Stage_0/Unit_0': [batch_size, 28, 28, 240],
            'Stage_0/Unit_1': [batch_size, 28, 28, 240],
            'Stage_0/Unit_2': [batch_size, 28, 28, 240],
            'Stage_0/Unit_3': [batch_size, 28, 28, 240],
            # Stage 2 with 8 units
            'Stage_1/Unit_0': [batch_size, 14, 14, 480],
            'Stage_1/Unit_1': [batch_size, 14, 14, 480],
            'Stage_1/Unit_2': [batch_size, 14, 14, 480],
            'Stage_1/Unit_3': [batch_size, 14, 14, 480],
            'Stage_1/Unit_4': [batch_size, 14, 14, 480],
            'Stage_1/Unit_5': [batch_size, 14, 14, 480],
            'Stage_1/Unit_6': [batch_size, 14, 14, 480],
            'Stage_1/Unit_7': [batch_size, 14, 14, 480],
            # Stage 3 with 4 units
            'Stage_2/Unit_0': [batch_size, 14, 14, 960],
            'Stage_2/Unit_1': [batch_size, 14, 14, 960],
            'Stage_2/Unit_2': [batch_size, 14, 14, 960],
            'Stage_2/Unit_3': [batch_size, 14, 14, 960],
        }

        self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
        for endpoint_name, expected_shape in endpoints_shapes.items():
            self.assertTrue(endpoint_name in end_points)
            self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                                 expected_shape)

    def testOutputStride8BuildAndCheckAllEndPointsUptoStage_2_Sub_Unit_3(self):
        batch_size = 5
        height, width = 224, 224
        output_stride = 8

        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            _, end_points = shufflenet.shufflenet_base(
                inputs, output_stride=output_stride,
                final_endpoint='Stage_2/Unit_3')

        endpoints_shapes = {
            # regular conv and pool
            'Conv2d_0': [batch_size, 112, 112, 24],
            'MaxPool2d_0': [batch_size, 56, 56, 24],
            # Stage 1 with 4 units
            'Stage_0/Unit_0': [batch_size, 28, 28, 240],
            'Stage_0/Unit_1': [batch_size, 28, 28, 240],
            'Stage_0/Unit_2': [batch_size, 28, 28, 240],
            'Stage_0/Unit_3': [batch_size, 28, 28, 240],
            # Stage 2 with 8 units
            'Stage_1/Unit_0': [batch_size, 28, 28, 480],
            'Stage_1/Unit_1': [batch_size, 28, 28, 480],
            'Stage_1/Unit_2': [batch_size, 28, 28, 480],
            'Stage_1/Unit_3': [batch_size, 28, 28, 480],
            'Stage_1/Unit_4': [batch_size, 28, 28, 480],
            'Stage_1/Unit_5': [batch_size, 28, 28, 480],
            'Stage_1/Unit_6': [batch_size, 28, 28, 480],
            'Stage_1/Unit_7': [batch_size, 28, 28, 480],
            # Stage 3 with 4 units
            'Stage_2/Unit_0': [batch_size, 28, 28, 960],
            'Stage_2/Unit_1': [batch_size, 28, 28, 960],
            'Stage_2/Unit_2': [batch_size, 28, 28, 960],
            'Stage_2/Unit_3': [batch_size, 28, 28, 960],
        }
        self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
        for endpoint_name, expected_shape in endpoints_shapes.items():
            self.assertTrue(endpoint_name in end_points)
            self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                                 expected_shape)

    def testBuildAndCheckAllEndPointsApproximateFaceNet(self):
        batch_size = 5
        height, width = 128, 128

        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            _, end_points = shufflenet.shufflenet_base(
                inputs, final_endpoint='Stage_2/Unit_3', depth_multiplier=0.75)

        # For the Conv2d_0 layer FaceNet has depth=16
        endpoints_shapes = {
            # regular conv and pool
            'Conv2d_0': [batch_size, 64, 64, 24],
            'MaxPool2d_0': [batch_size, 32, 32, 24],
            # Stage 1 with 4 units
            'Stage_0/Unit_0': [batch_size, 16, 16, 180],
            'Stage_0/Unit_1': [batch_size, 16, 16, 180],
            'Stage_0/Unit_2': [batch_size, 16, 16, 180],
            'Stage_0/Unit_3': [batch_size, 16, 16, 180],
            # Stage 2 with 8 units
            'Stage_1/Unit_0': [batch_size, 8, 8, 360],
            'Stage_1/Unit_1': [batch_size, 8, 8, 360],
            'Stage_1/Unit_2': [batch_size, 8, 8, 360],
            'Stage_1/Unit_3': [batch_size, 8, 8, 360],
            'Stage_1/Unit_4': [batch_size, 8, 8, 360],
            'Stage_1/Unit_5': [batch_size, 8, 8, 360],
            'Stage_1/Unit_6': [batch_size, 8, 8, 360],
            'Stage_1/Unit_7': [batch_size, 8, 8, 360],
            # Stage 3 with 4 units
            'Stage_2/Unit_0': [batch_size, 4, 4, 720],
            'Stage_2/Unit_1': [batch_size, 4, 4, 720],
            'Stage_2/Unit_2': [batch_size, 4, 4, 720],
            'Stage_2/Unit_3': [batch_size, 4, 4, 720],
        }
        self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
        for endpoint_name, expected_shape in endpoints_shapes.items():
            self.assertTrue(endpoint_name in end_points)
            self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                                 expected_shape)

    def testModelHasExpectedNumberOfParameters(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))

        endpoints_num_params = {
            # regular conv and pool
            'Conv2d_0': 720, 'MaxPool2d_0': 720,
            # Stage 1 with 4 units
            'Stage_0/Unit_0': 6498, 'Stage_0/Unit_1': 17718,
            'Stage_0/Unit_2': 28938, 'Stage_0/Unit_3': 40158,
            # Stage 2 with 8 units
            'Stage_1/Unit_0': 51378, 'Stage_1/Unit_1': 93018,
            'Stage_1/Unit_2': 134658, 'Stage_1/Unit_3': 176298,
            'Stage_1/Unit_4': 217938, 'Stage_1/Unit_5': 259578,
            'Stage_1/Unit_6': 301218, 'Stage_1/Unit_7': 342858,
            # Stage 3 with 4 units
            'Stage_2/Unit_0': 384498, 'Stage_2/Unit_1': 544578,
            'Stage_2/Unit_2': 704658, 'Stage_2/Unit_3': 864738,
        }

        for scope, end_point in enumerate(endpoints_num_params):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d, group_conv2d],
                                normalizer_fn=slim.batch_norm):
                shufflenet.shufflenet_base(inputs, final_endpoint=end_point, scope=str(scope))
                total_params, _ = slim.model_analyzer.analyze_vars(
                    slim.get_model_variables(scope=str(scope)))

                self.assertAlmostEqual(endpoints_num_params[end_point], total_params)

    def testBuildEndPointsWithDepthMultiplierLessThanOne(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = shufflenet.shufflenet_v1(inputs, num_classes)

        endpoint_keys = [key for key in end_points.keys() if key.startswith('Stage')]

        _, end_points_with_multiplier = shufflenet.shufflenet_v1(
            inputs, num_classes, scope='depth_multiplied_net',
            depth_multiplier=0.5)

        for key in endpoint_keys:
            original_depth = end_points[key].get_shape().as_list()[3]
            new_depth = end_points_with_multiplier[key].get_shape().as_list()[3]
            self.assertEqual(0.5 * original_depth, new_depth)

    def testBuildEndPointsWithDepthMultiplierGreaterThanOne(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = shufflenet.shufflenet_v1(inputs, num_classes)

        endpoint_keys = [key for key in end_points.keys()
                         if key.startswith('Stage')]

        _, end_points_with_multiplier = shufflenet.shufflenet_v1(
            inputs, num_classes, scope='depth_multiplied_net',
            depth_multiplier=2.0)

        for key in endpoint_keys:
            original_depth = end_points[key].get_shape().as_list()[3]
            new_depth = end_points_with_multiplier[key].get_shape().as_list()[3]
            self.assertEqual(2.0 * original_depth, new_depth)

    def testRaiseValueErrorWithInvalidDepthMultiplier(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        with self.assertRaises(ValueError):
            _ = shufflenet.shufflenet_v1(
                inputs, num_classes, depth_multiplier=-0.1)
        with self.assertRaises(ValueError):
            _ = shufflenet.shufflenet_v1(
                inputs, num_classes, depth_multiplier=0.0)

    def testHalfSizeImages(self):
        batch_size = 5
        height, width = 112, 112
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = shufflenet.shufflenet_v1(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith('ShufflenetV1/Logits'))
        self.assertListEqual(logits.get_shape().as_list(),
                             [batch_size, num_classes])
        pre_pool = end_points['Stage_2/Unit_3']
        self.assertListEqual(pre_pool.get_shape().as_list(),
                             [batch_size, 4, 4, 960])

    def testUnknownImageShape(self):
        tf.reset_default_graph()
        batch_size = 2
        height, width = 224, 224
        num_classes = 1000
        input_np = np.random.uniform(0, 1, (batch_size, height, width, 3))
        with self.test_session() as sess:
            inputs = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3))
            logits, end_points = shufflenet.shufflenet_v1(inputs, num_classes)
            self.assertTrue(logits.op.name.startswith('ShufflenetV1/Logits'))
            self.assertListEqual(logits.get_shape().as_list(),
                                 [batch_size, num_classes])
            pre_pool = end_points['Stage_2/Unit_3']
            feed_dict = {inputs: input_np}
            tf.global_variables_initializer().run()
            pre_pool_out = sess.run(pre_pool, feed_dict=feed_dict)
            self.assertListEqual(list(pre_pool_out.shape), [batch_size, 7, 7, 960])


if __name__ == '__main__':
    tf.test.main()
