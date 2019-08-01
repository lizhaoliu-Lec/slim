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
            '3': [480, 960, 1024],
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
        endpoints_groups = {
            '1': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 6804, 'Stage_0/Unit_1': 18144,
                'Stage_0/Unit_2': 29484, 'Stage_0/Unit_3': 40824,
                'Stage_1/Unit_0': 63072, 'Stage_1/Unit_1': 106488,
                'Stage_1/Unit_2': 149904, 'Stage_1/Unit_3': 193320,
                'Stage_1/Unit_4': 236736, 'Stage_1/Unit_5': 280152,
                'Stage_1/Unit_6': 323568, 'Stage_1/Unit_7': 366984,
                'Stage_2/Unit_0': 452952, 'Stage_2/Unit_1': 622728,
                'Stage_2/Unit_2': 792504, 'Stage_2/Unit_3': 962280
            },
            '2': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 7598, 'Stage_0/Unit_1': 18948,
                'Stage_0/Unit_2': 30298, 'Stage_0/Unit_3': 41648,
                'Stage_1/Unit_0': 63748, 'Stage_1/Unit_1': 106448,
                'Stage_1/Unit_2': 149148, 'Stage_1/Unit_3': 191848,
                'Stage_1/Unit_4': 234548, 'Stage_1/Unit_5': 277248,
                'Stage_1/Unit_6': 319948, 'Stage_1/Unit_7': 362648,
                'Stage_2/Unit_0': 446848, 'Stage_2/Unit_1': 612248,
                'Stage_2/Unit_2': 777648, 'Stage_2/Unit_3': 943048
            },
            '3': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 8028, 'Stage_0/Unit_1': 19248,
                'Stage_0/Unit_2': 30468, 'Stage_0/Unit_3': 41688,
                'Stage_1/Unit_0': 63408, 'Stage_1/Unit_1': 105048,
                'Stage_1/Unit_2': 146688, 'Stage_1/Unit_3': 188328,
                'Stage_1/Unit_4': 229968, 'Stage_1/Unit_5': 271608,
                'Stage_1/Unit_6': 313248, 'Stage_1/Unit_7': 354888,
                'Stage_2/Unit_0': 436728, 'Stage_2/Unit_1': 596808,
                'Stage_2/Unit_2': 756888, 'Stage_2/Unit_3': 916968
            },
            '4': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 8332, 'Stage_0/Unit_1': 19416,
                'Stage_0/Unit_2': 30500, 'Stage_0/Unit_3': 41584,
                'Stage_1/Unit_0': 62936, 'Stage_1/Unit_1': 103600,
                'Stage_1/Unit_2': 144264, 'Stage_1/Unit_3': 184928,
                'Stage_1/Unit_4': 225592, 'Stage_1/Unit_5': 266256,
                'Stage_1/Unit_6': 306920, 'Stage_1/Unit_7': 347584,
                'Stage_2/Unit_0': 427280, 'Stage_2/Unit_1': 582592,
                'Stage_2/Unit_2': 737904, 'Stage_2/Unit_3': 893216
            },
            '8': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 9864, 'Stage_0/Unit_1': 21672,
                'Stage_0/Unit_2': 33480, 'Stage_0/Unit_3': 45288,
                'Stage_1/Unit_0': 67752, 'Stage_1/Unit_1': 109800,
                'Stage_1/Unit_2': 151848, 'Stage_1/Unit_3': 193896,
                'Stage_1/Unit_4': 235944, 'Stage_1/Unit_5': 277992,
                'Stage_1/Unit_6': 320040, 'Stage_1/Unit_7': 362088,
                'Stage_2/Unit_0': 443880, 'Stage_2/Unit_1': 601704,
                'Stage_2/Unit_2': 759528, 'Stage_2/Unit_3': 917352
            },
        }

        for num_groups in endpoints_groups:
            for scope, end_point in enumerate(endpoints_groups[num_groups]):
                print(num_groups, end_point)
                with slim.arg_scope([slim.conv2d, slim.separable_conv2d, group_conv2d],
                                    normalizer_fn=slim.batch_norm):
                    shufflenet.shufflenet_base(inputs, final_endpoint=end_point,
                                               num_groups=int(num_groups),
                                               scope=num_groups + '/' + str(scope))
                    total_params, _ = slim.model_analyzer.analyze_vars(
                        slim.get_model_variables(scope=num_groups + '/' + str(scope)))

                    self.assertAlmostEqual(endpoints_groups[num_groups][end_point], total_params)

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

    def testUnknownBatchSize(self):
        batch_size = 1
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.placeholder(tf.float32, (None, height, width, 3))
        logits, _ = shufflenet.shufflenet_v1(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith('ShufflenetV1/Logits'))
        self.assertListEqual(logits.get_shape().as_list(),
                             [None, num_classes])
        images = tf.random_uniform((batch_size, height, width, 3))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(logits, {inputs: images.eval()})
            self.assertEquals(output.shape, (batch_size, num_classes))

    def testEvaluation(self):
        batch_size = 2
        height, width = 224, 224
        num_classes = 1000

        eval_inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, _ = shufflenet.shufflenet_v1(eval_inputs, num_classes,
                                             is_training=False)
        predictions = tf.argmax(logits, 1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(predictions)
            self.assertEquals(output.shape, (batch_size,))

    def testTrainEvalWithReuse(self):
        train_batch_size = 5
        eval_batch_size = 2
        height, width = 150, 150
        num_classes = 1000

        train_inputs = tf.random_uniform((train_batch_size, height, width, 3))
        shufflenet.shufflenet_v1(train_inputs, num_classes)
        eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
        logits, _ = shufflenet.shufflenet_v1(eval_inputs, num_classes,
                                             reuse=True)
        predictions = tf.argmax(logits, 1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(predictions)
            self.assertEquals(output.shape, (eval_batch_size,))

    def testLogitsNotSqueezed(self):
        num_classes = 25
        images = tf.random_uniform([1, 224, 224, 3])
        logits, _ = shufflenet.shufflenet_v1(images,
                                             num_classes=num_classes,
                                             spatial_squeeze=False)

        with self.test_session() as sess:
            tf.global_variables_initializer().run()
            logits_out = sess.run(logits)
            self.assertListEqual(list(logits_out.shape), [1, 1, 1, num_classes])

    def testBatchNormScopeDoesNotHaveIsTrainingWhenItsSetToNone(self):
        sc = shufflenet.shufflenet_v1_arg_scope(is_training=None)
        self.assertNotIn('is_training', sc[slim.arg_scope_func_key(
            slim.batch_norm)])

    def testBatchNormScopeDoesHasIsTrainingWhenItsNotNone(self):
        sc = shufflenet.shufflenet_v1_arg_scope(is_training=True)
        self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])
        sc = shufflenet.shufflenet_v1_arg_scope(is_training=False)
        self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])
        sc = shufflenet.shufflenet_v1_arg_scope()
        self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])


if __name__ == '__main__':
    tf.test.main()
