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

from nets.shufflenet import shufflenet_v1
from nets.shufflenet.shufflenet_utils import group_conv2d


class ShuffleNetV1Test(tf.test.TestCase):

    def testBuild(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            logits, _ = shufflenet_v1.shufflenet_v1(inputs, num_classes)
            self.assertListEqual(logits.get_shape().as_list(),
                                 [batch_size, num_classes])

    def testBuildClassificationNetwork(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = shufflenet_v1.shufflenet_v1(inputs, num_classes)
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
        net, end_points = shufflenet_v1.shufflenet_v1(inputs, num_classes)
        self.assertTrue(net.op.name.startswith('ShufflenetV1/Logits/AvgPool'))
        self.assertListEqual(net.get_shape().as_list(), [batch_size, 1, 1, 960])
        self.assertFalse('Logits' in end_points)
        self.assertFalse('Predictions' in end_points)

    def testBuildBaseNetwork(self):
        batch_size = 5
        height, width = 224, 224

        inputs = tf.random_uniform((batch_size, height, width, 3))
        net, endpoints = shufflenet_v1.shufflenet_v1_base(inputs)
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
                out_tensor, end_points = shufflenet_v1.shufflenet_v1_base(inputs, final_endpoint=endpoint)
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
            '3': [480, 960, 1440],
            '4': [272, 544, 1088],
            '8': [384, 768, 1536],
        }

        inputs = tf.random_uniform((batch_size, height, width, 3))
        net, end_points = shufflenet_v1.shufflenet_v1_base(
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
            _, end_points = shufflenet_v1.shufflenet_v1_base(
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
            _, end_points = shufflenet_v1.shufflenet_v1_base(
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
            _, end_points = shufflenet_v1.shufflenet_v1_base(
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
            _, end_points = shufflenet_v1.shufflenet_v1_base(
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
                    shufflenet_v1.shufflenet_v1_base(inputs, final_endpoint=end_point,
                                                     num_groups=int(num_groups),
                                                     scope=num_groups + '/' + str(scope))
                    total_params, _ = slim.model_analyzer.analyze_vars(
                        slim.get_model_variables(scope=num_groups + '/' + str(scope)))

                    self.assertAlmostEqual(endpoints_groups[num_groups][end_point], total_params)

    def testModelHasExpectedNumberOfParametersWithRate0_25(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))
        endpoints_groups_0_25 = {
            '1': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 1215, 'Stage_0/Unit_1': 2106,
                'Stage_0/Unit_2': 2997, 'Stage_0/Unit_3': 3888,
                'Stage_1/Unit_0': 5562, 'Stage_1/Unit_1': 8640,
                'Stage_1/Unit_2': 11718, 'Stage_1/Unit_3': 14796,
                'Stage_1/Unit_4': 17874, 'Stage_1/Unit_5': 20952,
                'Stage_1/Unit_6': 24030, 'Stage_1/Unit_7': 27108,
                'Stage_2/Unit_0': 33048, 'Stage_2/Unit_1': 44388,
                'Stage_2/Unit_2': 55728, 'Stage_2/Unit_3': 67068
            },
            '2': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 1422, 'Stage_0/Unit_1': 2352,
                'Stage_0/Unit_2': 3282, 'Stage_0/Unit_3': 4212,
                'Stage_1/Unit_0': 5922, 'Stage_1/Unit_1': 8982,
                'Stage_1/Unit_2': 12042, 'Stage_1/Unit_3': 15102,
                'Stage_1/Unit_4': 18162, 'Stage_1/Unit_5': 21222,
                'Stage_1/Unit_6': 24282, 'Stage_1/Unit_7': 27342,
                'Stage_2/Unit_0': 33392, 'Stage_2/Unit_1': 44742,
                'Stage_2/Unit_2': 56092, 'Stage_2/Unit_3': 67442
            },
            '3': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 1593, 'Stage_0/Unit_1': 2598,
                'Stage_0/Unit_2': 3603, 'Stage_0/Unit_3': 4608,
                'Stage_1/Unit_0': 6438, 'Stage_1/Unit_1': 9648,
                'Stage_1/Unit_2': 12858, 'Stage_1/Unit_3': 16068,
                'Stage_1/Unit_4': 19278, 'Stage_1/Unit_5': 22488,
                'Stage_1/Unit_6': 25698, 'Stage_1/Unit_7': 28908,
                'Stage_2/Unit_0': 34968, 'Stage_2/Unit_1': 46188,
                'Stage_2/Unit_2': 57408, 'Stage_2/Unit_3': 68628
            },
            '4': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 1652, 'Stage_0/Unit_1': 2640,
                'Stage_0/Unit_2': 3628, 'Stage_0/Unit_3': 4616,
                'Stage_1/Unit_0': 6388, 'Stage_1/Unit_1': 9452,
                'Stage_1/Unit_2': 12516, 'Stage_1/Unit_3': 15580,
                'Stage_1/Unit_4': 18644, 'Stage_1/Unit_5': 21708,
                'Stage_1/Unit_6': 24772, 'Stage_1/Unit_7': 27836,
                'Stage_2/Unit_0': 33888, 'Stage_2/Unit_1': 44972,
                'Stage_2/Unit_2': 56056, 'Stage_2/Unit_3': 67140
            },
            '8': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 2088, 'Stage_0/Unit_1': 3312,
                'Stage_0/Unit_2': 4536, 'Stage_0/Unit_3': 5760,
                'Stage_1/Unit_0': 7920, 'Stage_1/Unit_1': 11520,
                'Stage_1/Unit_2': 15120, 'Stage_1/Unit_3': 18720,
                'Stage_1/Unit_4': 22320, 'Stage_1/Unit_5': 25920,
                'Stage_1/Unit_6': 29520, 'Stage_1/Unit_7': 33120,
                'Stage_2/Unit_0': 39744, 'Stage_2/Unit_1': 51552,
                'Stage_2/Unit_2': 63360, 'Stage_2/Unit_3': 75168
            },
        }

        for num_groups in endpoints_groups_0_25:
            for scope, end_point in enumerate(endpoints_groups_0_25[num_groups]):
                print(num_groups, end_point)
                with slim.arg_scope([slim.conv2d, slim.separable_conv2d, group_conv2d],
                                    normalizer_fn=slim.batch_norm):
                    shufflenet_v1.shufflenet_v1_base(inputs, final_endpoint=end_point,
                                                     depth_multiplier=0.25,
                                                     num_groups=int(num_groups),
                                                     scope=num_groups + '/' + str(scope))
                    total_params, _ = slim.model_analyzer.analyze_vars(
                        slim.get_model_variables(scope=num_groups + '/' + str(scope)))

                    self.assertAlmostEqual(endpoints_groups_0_25[num_groups][end_point], total_params)

    def testModelHasExpectedNumberOfParametersWithRate0_5(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))
        endpoints_groups_0_5 = {
            '1': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 2430, 'Stage_0/Unit_1': 5508,
                'Stage_0/Unit_2': 8586, 'Stage_0/Unit_3': 11664,
                'Stage_1/Unit_0': 17604, 'Stage_1/Unit_1': 28944,
                'Stage_1/Unit_2': 40284, 'Stage_1/Unit_3': 51624,
                'Stage_1/Unit_4': 62964, 'Stage_1/Unit_5': 74304,
                'Stage_1/Unit_6': 85644, 'Stage_1/Unit_7': 96984,
                'Stage_2/Unit_0': 119232, 'Stage_2/Unit_1': 162648,
                'Stage_2/Unit_2': 206064, 'Stage_2/Unit_3': 249480
            },
            '2': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 2796, 'Stage_0/Unit_1': 5856,
                'Stage_0/Unit_2': 8916, 'Stage_0/Unit_3': 11976,
                'Stage_1/Unit_0': 18026, 'Stage_1/Unit_1': 29376,
                'Stage_1/Unit_2': 40726, 'Stage_1/Unit_3': 52076,
                'Stage_1/Unit_4': 63426, 'Stage_1/Unit_5': 74776,
                'Stage_1/Unit_6': 86126, 'Stage_1/Unit_7': 97476,
                'Stage_2/Unit_0': 119576, 'Stage_2/Unit_1': 162276,
                'Stage_2/Unit_2': 204976, 'Stage_2/Unit_3': 247676
            },
            '3': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 3138, 'Stage_0/Unit_1': 6348,
                'Stage_0/Unit_2': 9558, 'Stage_0/Unit_3': 12768,
                'Stage_1/Unit_0': 18828, 'Stage_1/Unit_1': 30048,
                'Stage_1/Unit_2': 41268, 'Stage_1/Unit_3': 52488,
                'Stage_1/Unit_4': 63708, 'Stage_1/Unit_5': 74928,
                'Stage_1/Unit_6': 86148, 'Stage_1/Unit_7': 97368,
                'Stage_2/Unit_0': 119088, 'Stage_2/Unit_1': 160728,
                'Stage_2/Unit_2': 202368, 'Stage_2/Unit_3': 244008
            },
            '4': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 3200, 'Stage_0/Unit_1': 6264,
                'Stage_0/Unit_2': 9328, 'Stage_0/Unit_3': 12392,
                'Stage_1/Unit_0': 18444, 'Stage_1/Unit_1': 29528,
                'Stage_1/Unit_2': 40612, 'Stage_1/Unit_3': 51696,
                'Stage_1/Unit_4': 62780, 'Stage_1/Unit_5': 73864,
                'Stage_1/Unit_6': 84948, 'Stage_1/Unit_7': 96032,
                'Stage_2/Unit_0': 117384, 'Stage_2/Unit_1': 158048,
                'Stage_2/Unit_2': 198712, 'Stage_2/Unit_3': 239376
            },
            '8': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 4104, 'Stage_0/Unit_1': 7704,
                'Stage_0/Unit_2': 11304, 'Stage_0/Unit_3': 14904,
                'Stage_1/Unit_0': 21528, 'Stage_1/Unit_1': 33336,
                'Stage_1/Unit_2': 45144, 'Stage_1/Unit_3': 56952,
                'Stage_1/Unit_4': 68760, 'Stage_1/Unit_5': 80568,
                'Stage_1/Unit_6': 92376, 'Stage_1/Unit_7': 104184,
                'Stage_2/Unit_0': 126648, 'Stage_2/Unit_1': 168696,
                'Stage_2/Unit_2': 210744, 'Stage_2/Unit_3': 252792
            },
        }

        for num_groups in endpoints_groups_0_5:
            for scope, end_point in enumerate(endpoints_groups_0_5[num_groups]):
                print(num_groups, end_point)
                with slim.arg_scope([slim.conv2d, slim.separable_conv2d, group_conv2d],
                                    normalizer_fn=slim.batch_norm):
                    shufflenet_v1.shufflenet_v1_base(inputs, final_endpoint=end_point,
                                                     depth_multiplier=0.5,
                                                     num_groups=int(num_groups),
                                                     scope=num_groups + '/' + str(scope))
                    total_params, _ = slim.model_analyzer.analyze_vars(
                        slim.get_model_variables(scope=num_groups + '/' + str(scope)))

                    self.assertAlmostEqual(endpoints_groups_0_5[num_groups][end_point], total_params)

    def testModelHasExpectedNumberOfParametersWithRate1_5(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))
        endpoints_groups_1_5 = {
            '1': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 13770, 'Stage_0/Unit_1': 38556,
                'Stage_0/Unit_2': 63342, 'Stage_0/Unit_3': 88128,
                'Stage_1/Unit_0': 137052, 'Stage_1/Unit_1': 233280,
                'Stage_1/Unit_2': 329508, 'Stage_1/Unit_3': 425736,
                'Stage_1/Unit_4': 521964, 'Stage_1/Unit_5': 618192,
                'Stage_1/Unit_6': 714420, 'Stage_1/Unit_7': 810648,
                'Stage_2/Unit_0': 1001808, 'Stage_2/Unit_1': 1380888,
                'Stage_2/Unit_2': 1759968, 'Stage_2/Unit_3': 2139048
            },
            '2': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 14646, 'Stage_0/Unit_1': 38856,
                'Stage_0/Unit_2': 63066, 'Stage_0/Unit_3': 87276,
                'Stage_1/Unit_0': 135426, 'Stage_1/Unit_1': 229476,
                'Stage_1/Unit_2': 323526, 'Stage_1/Unit_3': 417576,
                'Stage_1/Unit_4': 511626, 'Stage_1/Unit_5': 605676,
                'Stage_1/Unit_6': 699726, 'Stage_1/Unit_7': 793776,
                'Stage_2/Unit_0': 980076, 'Stage_2/Unit_1': 1348176,
                'Stage_2/Unit_2': 1716276, 'Stage_2/Unit_3': 2084376
            },
            '3': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 15318, 'Stage_0/Unit_1': 39348,
                'Stage_0/Unit_2': 63378, 'Stage_0/Unit_3': 87408,
                'Stage_1/Unit_0': 134388, 'Stage_1/Unit_1': 225648,
                'Stage_1/Unit_2': 316908, 'Stage_1/Unit_3': 408168,
                'Stage_1/Unit_4': 499428, 'Stage_1/Unit_5': 590688,
                'Stage_1/Unit_6': 681948, 'Stage_1/Unit_7': 773208,
                'Stage_2/Unit_0': 953568, 'Stage_2/Unit_1': 1308888,
                'Stage_2/Unit_2': 1664208, 'Stage_2/Unit_3': 2019528
            },
            '4': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 15372, 'Stage_0/Unit_1': 38496,
                'Stage_0/Unit_2': 61620, 'Stage_0/Unit_3': 84744,
                'Stage_1/Unit_0': 130644, 'Stage_1/Unit_1': 219384,
                'Stage_1/Unit_2': 308124, 'Stage_1/Unit_3': 396864,
                'Stage_1/Unit_4': 485604, 'Stage_1/Unit_5': 574344,
                'Stage_1/Unit_6': 663084, 'Stage_1/Unit_7': 751824,
                'Stage_2/Unit_0': 926856, 'Stage_2/Unit_1': 1270800,
                'Stage_2/Unit_2': 1614744, 'Stage_2/Unit_3': 1958688
            },
            '8': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 17928, 'Stage_0/Unit_1': 42552,
                'Stage_0/Unit_2': 67176, 'Stage_0/Unit_3': 91800,
                'Stage_1/Unit_0': 139320, 'Stage_1/Unit_1': 230040,
                'Stage_1/Unit_2': 320760, 'Stage_1/Unit_3': 411480,
                'Stage_1/Unit_4': 502200, 'Stage_1/Unit_5': 592920,
                'Stage_1/Unit_6': 683640, 'Stage_1/Unit_7': 774360,
                'Stage_2/Unit_0': 952344, 'Stage_2/Unit_1': 1299672,
                'Stage_2/Unit_2': 1647000, 'Stage_2/Unit_3': 1994328
            },
        }

        for num_groups in endpoints_groups_1_5:
            for scope, end_point in enumerate(endpoints_groups_1_5[num_groups]):
                print(num_groups, end_point)
                with slim.arg_scope([slim.conv2d, slim.separable_conv2d, group_conv2d],
                                    normalizer_fn=slim.batch_norm):
                    shufflenet_v1.shufflenet_v1_base(inputs, final_endpoint=end_point,
                                                     depth_multiplier=1.5,
                                                     num_groups=int(num_groups),
                                                     scope=num_groups + '/' + str(scope))
                    total_params, _ = slim.model_analyzer.analyze_vars(
                        slim.get_model_variables(scope=num_groups + '/' + str(scope)))

                    self.assertAlmostEqual(endpoints_groups_1_5[num_groups][end_point], total_params)

    def testModelHasExpectedNumberOfParametersWithRate2_0(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))
        endpoints_groups_2_0 = {
            '1': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 23328, 'Stage_0/Unit_1': 66744,
                'Stage_0/Unit_2': 110160, 'Stage_0/Unit_3': 153576,
                'Stage_1/Unit_0': 239544, 'Stage_1/Unit_1': 409320,
                'Stage_1/Unit_2': 579096, 'Stage_1/Unit_3': 748872,
                'Stage_1/Unit_4': 918648, 'Stage_1/Unit_5': 1088424,
                'Stage_1/Unit_6': 1258200, 'Stage_1/Unit_7': 1427976,
                'Stage_2/Unit_0': 1765800, 'Stage_2/Unit_1': 2437128,
                'Stage_2/Unit_2': 3108456, 'Stage_2/Unit_3': 3779784
            },
            '2': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 24548, 'Stage_0/Unit_1': 67248,
                'Stage_0/Unit_2': 109948, 'Stage_0/Unit_3': 152648,
                'Stage_1/Unit_0': 236848, 'Stage_1/Unit_1': 402248,
                'Stage_1/Unit_2': 567648, 'Stage_1/Unit_3': 733048,
                'Stage_1/Unit_4': 898448, 'Stage_1/Unit_5': 1063848,
                'Stage_1/Unit_6': 1229248, 'Stage_1/Unit_7': 1394648,
                'Stage_2/Unit_0': 1723048, 'Stage_2/Unit_1': 2373848,
                'Stage_2/Unit_2': 3024648, 'Stage_2/Unit_3': 3675448
            },
            '3': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 25008, 'Stage_0/Unit_1': 66648,
                'Stage_0/Unit_2': 108288, 'Stage_0/Unit_3': 149928,
                'Stage_1/Unit_0': 231768, 'Stage_1/Unit_1': 391848,
                'Stage_1/Unit_2': 551928, 'Stage_1/Unit_3': 712008,
                'Stage_1/Unit_4': 872088, 'Stage_1/Unit_5': 1032168,
                'Stage_1/Unit_6': 1192248, 'Stage_1/Unit_7': 1352328,
                'Stage_2/Unit_0': 1669608, 'Stage_2/Unit_1': 2296968,
                'Stage_2/Unit_2': 2924328, 'Stage_2/Unit_3': 3551688
            },
            '4': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 25264, 'Stage_0/Unit_1': 65928,
                'Stage_0/Unit_2': 106592, 'Stage_0/Unit_3': 147256,
                'Stage_1/Unit_0': 226952, 'Stage_1/Unit_1': 382264,
                'Stage_1/Unit_2': 537576, 'Stage_1/Unit_3': 692888,
                'Stage_1/Unit_4': 848200, 'Stage_1/Unit_5': 1003512,
                'Stage_1/Unit_6': 1158824, 'Stage_1/Unit_7': 1314136,
                'Stage_2/Unit_0': 1621496, 'Stage_2/Unit_1': 2228056,
                'Stage_2/Unit_2': 2834616, 'Stage_2/Unit_3': 3441176
            },
            '8': {
                'Conv2d_0': 720, 'MaxPool2d_0': 720,
                'Stage_0/Unit_0': 28296, 'Stage_0/Unit_1': 70344,
                'Stage_0/Unit_2': 112392, 'Stage_0/Unit_3': 154440,
                'Stage_1/Unit_0': 236232, 'Stage_1/Unit_1': 394056,
                'Stage_1/Unit_2': 551880, 'Stage_1/Unit_3': 709704,
                'Stage_1/Unit_4': 867528, 'Stage_1/Unit_5': 1025352,
                'Stage_1/Unit_6': 1183176, 'Stage_1/Unit_7': 1341000,
                'Stage_2/Unit_0': 1652040, 'Stage_2/Unit_1': 2262600,
                'Stage_2/Unit_2': 2873160, 'Stage_2/Unit_3': 3483720
            },
        }

        for num_groups in endpoints_groups_2_0:
            for scope, end_point in enumerate(endpoints_groups_2_0[num_groups]):
                print(num_groups, end_point)
                with slim.arg_scope([slim.conv2d, slim.separable_conv2d, group_conv2d],
                                    normalizer_fn=slim.batch_norm):
                    shufflenet_v1.shufflenet_v1_base(inputs, final_endpoint=end_point,
                                                     num_groups=int(num_groups),
                                                     depth_multiplier=2.0,
                                                     scope=num_groups + '/' + str(scope))
                    total_params, _ = slim.model_analyzer.analyze_vars(
                        slim.get_model_variables(scope=num_groups + '/' + str(scope)))

                    self.assertAlmostEqual(endpoints_groups_2_0[num_groups][end_point], total_params)

    def testBuildEndPointsWithDepthMultiplierLessThanOne(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        _, end_points = shufflenet_v1.shufflenet_v1(inputs, num_classes)

        endpoint_keys = [key for key in end_points.keys() if key.startswith('Stage')]

        _, end_points_with_multiplier = shufflenet_v1.shufflenet_v1(
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
        _, end_points = shufflenet_v1.shufflenet_v1(inputs, num_classes)

        endpoint_keys = [key for key in end_points.keys()
                         if key.startswith('Stage')]

        _, end_points_with_multiplier = shufflenet_v1.shufflenet_v1(
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
            _ = shufflenet_v1.shufflenet_v1(
                inputs, num_classes, depth_multiplier=-0.1)
        with self.assertRaises(ValueError):
            _ = shufflenet_v1.shufflenet_v1(
                inputs, num_classes, depth_multiplier=0.0)

    def testHalfSizeImages(self):
        batch_size = 5
        height, width = 112, 112
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = shufflenet_v1.shufflenet_v1(inputs, num_classes)
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
            logits, end_points = shufflenet_v1.shufflenet_v1(inputs, num_classes)
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
        logits, _ = shufflenet_v1.shufflenet_v1(inputs, num_classes)
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
        logits, _ = shufflenet_v1.shufflenet_v1(eval_inputs, num_classes,
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
        shufflenet_v1.shufflenet_v1(train_inputs, num_classes)
        eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
        logits, _ = shufflenet_v1.shufflenet_v1(eval_inputs, num_classes,
                                                reuse=True)
        predictions = tf.argmax(logits, 1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(predictions)
            self.assertEquals(output.shape, (eval_batch_size,))

    def testLogitsNotSqueezed(self):
        num_classes = 25
        images = tf.random_uniform([1, 224, 224, 3])
        logits, _ = shufflenet_v1.shufflenet_v1(images,
                                                num_classes=num_classes,
                                                spatial_squeeze=False)

        with self.test_session() as sess:
            tf.global_variables_initializer().run()
            logits_out = sess.run(logits)
            self.assertListEqual(list(logits_out.shape), [1, 1, 1, num_classes])

    def testBatchNormScopeDoesNotHaveIsTrainingWhenItsSetToNone(self):
        sc = shufflenet_v1.shufflenet_v1_arg_scope(is_training=None)
        self.assertNotIn('is_training', sc[slim.arg_scope_func_key(
            slim.batch_norm)])

    def testBatchNormScopeDoesHasIsTrainingWhenItsNotNone(self):
        sc = shufflenet_v1.shufflenet_v1_arg_scope(is_training=True)
        self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])
        sc = shufflenet_v1.shufflenet_v1_arg_scope(is_training=False)
        self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])
        sc = shufflenet_v1.shufflenet_v1_arg_scope()
        self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])

    def testFlops(self):
        def stats_graph(graph):
            flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
            params = tf.profiler.profile(graph,
                                         options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
            print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))

        with slim.arg_scope(shufflenet_v1.shufflenet_v1_arg_scope()):
            shufflenet_v1.shufflenet_v1_base(inputs, depth_multiplier=2.0)
            graph = tf.get_default_graph()
            stats_graph(graph)
        self.assertTrue(False)


if __name__ == '__main__':
    tf.test.main()
