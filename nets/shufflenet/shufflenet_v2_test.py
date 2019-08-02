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
import numpy as np
from nets.shufflenet import shufflenet_v2

# slim = tf.contrib.slim
import tensorflow.contrib.slim as slim


class ShuffleNetV2Test(tf.test.TestCase):

    # 1st test
    def testBuild(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            logits, _ = shufflenet_v2.shufflenet_v2(inputs, num_classes)
            self.assertListEqual(logits.get_shape().as_list(),
                                 [batch_size, num_classes])

    # 2nd test
    def testBuildClassificationNetwork(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = shufflenet_v2.shufflenet_v2(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith(
            'ShufflenetV2/Logits/SpatialSqueeze'
        ))
        self.assertListEqual(logits.get_shape().as_list(),
                             [batch_size, num_classes])

        self.assertTrue('Predictions' in end_points)
        self.assertListEqual(end_points['Predictions'].get_shape().as_list(),
                             [batch_size, num_classes])

    # 3rd test
    def testBuildPreLogitsNetwork(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = None

        inputs = tf.random_uniform((batch_size, height, width, 3))
        net, end_points = shufflenet_v2.shufflenet_v2(inputs, num_classes)
        self.assertTrue(net.op.name.startswith('ShufflenetV2/Logits/AvgPool'))
        self.assertListEqual(net.get_shape().as_list(), [batch_size, 1, 1, 1024])
        self.assertFalse('Logits' in end_points)
        self.assertFalse('Predictions' in end_points)

    # 4th test
    def testBuildBaseNetwork(self):
        batch_size = 5
        height, width = 224, 224

        inputs = tf.random_uniform((batch_size, height, width, 3))
        net, endpoints = shufflenet_v2.shufflenet_v2_base(inputs)
        self.assertListEqual(net.get_shape().as_list(), [batch_size, 7, 7, 1024])
        self.maxDiff = None

        expected_endpoints = [
            # Conv1 & MaxPool
            'Conv2d_0', 'MaxPool2d_0',
            # Stage 1 with 4 units
            'Stage_0/Unit_0',
            'Stage_0/Unit_1',
            'Stage_0/Unit_2',
            'Stage_0/Unit_3',
            # Stage 2 with 8 units
            'Stage_1/Unit_0',
            'Stage_1/Unit_1',
            'Stage_1/Unit_2',
            'Stage_1/Unit_3',
            'Stage_1/Unit_4',
            'Stage_1/Unit_5',
            'Stage_1/Unit_6',
            'Stage_1/Unit_7',
            # Stage 3 with 4 units
            'Stage_2/Unit_0',
            'Stage_2/Unit_1',
            'Stage_2/Unit_2',
            'Stage_2/Unit_3',
            # Conv2 to mix up features
            'Conv2d_1',
        ]

        self.assertItemsEqual(endpoints.keys(), expected_endpoints)

    # 5th test
    def testBuildOnlyUptoFinalEndpoint(self):
        batch_size = 5
        height, width = 224, 224
        endpoints = [
            # Conv1 & MaxPool
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
            # Conv2 to mix up features
            'Conv2d_1',
        ]
        for index, endpoint in enumerate(endpoints):
            with tf.Graph().as_default():
                inputs = tf.random_uniform((batch_size, height, width, 3))
                out_tensor, end_points = shufflenet_v2.shufflenet_v2_base(
                    inputs, final_endpoint=endpoint)
                print(out_tensor.op.name)
                self.assertTrue(out_tensor.op.name.startswith(
                    'ShufflenetV2/' + endpoint))
                self.assertItemsEqual(endpoints[:index + 1], end_points.keys())

    # 6th test
    def testBuildAndCheckAllEndPointsUptoConv2d_1(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            _, end_points = shufflenet_v2.shufflenet_v2_base(
                inputs, final_endpoint='Conv2d_1')
        endpoints_shapes = {
            # Conv1 & MaxPool
            'Conv2d_0': [batch_size, 112, 112, 24],
            'MaxPool2d_0': [batch_size, 56, 56, 24],
            # Stage 1 with 4 units
            'Stage_0/Unit_0': [batch_size, 28, 28, 116],
            'Stage_0/Unit_1': [batch_size, 28, 28, 116],
            'Stage_0/Unit_2': [batch_size, 28, 28, 116],
            'Stage_0/Unit_3': [batch_size, 28, 28, 116],
            # Stage 2 with 8 units
            'Stage_1/Unit_0': [batch_size, 14, 14, 232],
            'Stage_1/Unit_1': [batch_size, 14, 14, 232],
            'Stage_1/Unit_2': [batch_size, 14, 14, 232],
            'Stage_1/Unit_3': [batch_size, 14, 14, 232],
            'Stage_1/Unit_4': [batch_size, 14, 14, 232],
            'Stage_1/Unit_5': [batch_size, 14, 14, 232],
            'Stage_1/Unit_6': [batch_size, 14, 14, 232],
            'Stage_1/Unit_7': [batch_size, 14, 14, 232],
            # Stage 3 with 4 units
            'Stage_2/Unit_0': [batch_size, 7, 7, 464],
            'Stage_2/Unit_1': [batch_size, 7, 7, 464],
            'Stage_2/Unit_2': [batch_size, 7, 7, 464],
            'Stage_2/Unit_3': [batch_size, 7, 7, 464],
            # Conv2 to mix up features
            'Conv2d_1': [batch_size, 7, 7, 1024]
        }

        self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
        for endpoint_name, expected_shape in endpoints_shapes.items():
            self.assertTrue(endpoint_name in end_points)
            self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                                 expected_shape)

    # 7th test
    def testOutputStride16BuildAndCheckAllEndPointsUptoConv2d_1(self):
        batch_size = 5
        height, width = 224, 224
        output_stride = 16

        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            _, end_points = shufflenet_v2.shufflenet_v2_base(
                inputs, output_stride=output_stride,
                final_endpoint='Conv2d_1')
        endpoints_shapes = {
            # Conv1 & MaxPool
            'Conv2d_0': [batch_size, 112, 112, 24],
            'MaxPool2d_0': [batch_size, 56, 56, 24],
            # Stage 1 with 4 units
            'Stage_0/Unit_0': [batch_size, 28, 28, 116],
            'Stage_0/Unit_1': [batch_size, 28, 28, 116],
            'Stage_0/Unit_2': [batch_size, 28, 28, 116],
            'Stage_0/Unit_3': [batch_size, 28, 28, 116],
            # Stage 2 with 8 units
            'Stage_1/Unit_0': [batch_size, 14, 14, 232],
            'Stage_1/Unit_1': [batch_size, 14, 14, 232],
            'Stage_1/Unit_2': [batch_size, 14, 14, 232],
            'Stage_1/Unit_3': [batch_size, 14, 14, 232],
            'Stage_1/Unit_4': [batch_size, 14, 14, 232],
            'Stage_1/Unit_5': [batch_size, 14, 14, 232],
            'Stage_1/Unit_6': [batch_size, 14, 14, 232],
            'Stage_1/Unit_7': [batch_size, 14, 14, 232],
            # Stage 3 with 4 units
            'Stage_2/Unit_0': [batch_size, 14, 14, 464],
            'Stage_2/Unit_1': [batch_size, 14, 14, 464],
            'Stage_2/Unit_2': [batch_size, 14, 14, 464],
            'Stage_2/Unit_3': [batch_size, 14, 14, 464],
            # Conv2 to mix up features
            'Conv2d_1': [batch_size, 14, 14, 1024]
        }

        self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
        for endpoint_name, expected_shape in endpoints_shapes.items():
            self.assertTrue(endpoint_name in end_points)
            self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                                 expected_shape)

    # 8th test
    def testOutputStride8BuildAndCheckAllEndPointsUptoConv2d_1(self):
        batch_size = 5
        height, width = 224, 224
        output_stride = 8

        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            _, end_points = shufflenet_v2.shufflenet_v2_base(
                inputs, output_stride=output_stride,
                final_endpoint='Conv2d_1')
        endpoints_shapes = {
            # Conv1 & MaxPool
            'Conv2d_0': [batch_size, 112, 112, 24],
            'MaxPool2d_0': [batch_size, 56, 56, 24],
            # Stage 1 with 4 units
            'Stage_0/Unit_0': [batch_size, 28, 28, 116],
            'Stage_0/Unit_1': [batch_size, 28, 28, 116],
            'Stage_0/Unit_2': [batch_size, 28, 28, 116],
            'Stage_0/Unit_3': [batch_size, 28, 28, 116],
            # Stage 2 with 8 units
            'Stage_1/Unit_0': [batch_size, 28, 28, 232],
            'Stage_1/Unit_1': [batch_size, 28, 28, 232],
            'Stage_1/Unit_2': [batch_size, 28, 28, 232],
            'Stage_1/Unit_3': [batch_size, 28, 28, 232],
            'Stage_1/Unit_4': [batch_size, 28, 28, 232],
            'Stage_1/Unit_5': [batch_size, 28, 28, 232],
            'Stage_1/Unit_6': [batch_size, 28, 28, 232],
            'Stage_1/Unit_7': [batch_size, 28, 28, 232],
            # Stage 3 with 4 units
            'Stage_2/Unit_0': [batch_size, 28, 28, 464],
            'Stage_2/Unit_1': [batch_size, 28, 28, 464],
            'Stage_2/Unit_2': [batch_size, 28, 28, 464],
            'Stage_2/Unit_3': [batch_size, 28, 28, 464],
            # Conv2 to mix up features
            'Conv2d_1': [batch_size, 28, 28, 1024]
        }

        self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
        for endpoint_name, expected_shape in endpoints_shapes.items():
            self.assertTrue(endpoint_name in end_points)
            self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                                 expected_shape)

    # 9th test
    def testBuild2_0xBasicNetworkAndCheckAllEndPointsUptoConv2d_1(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            _, end_points = shufflenet_v2.shufflenet_v2_base(
                inputs, depth_multiplier=2.0, final_endpoint='Conv2d_1')
        endpoints_shapes = {
            # Conv1 & MaxPool
            'Conv2d_0': [batch_size, 112, 112, 24],
            'MaxPool2d_0': [batch_size, 56, 56, 24],
            # Stage 1 with 4 units
            'Stage_0/Unit_0': [batch_size, 28, 28, 244],
            'Stage_0/Unit_1': [batch_size, 28, 28, 244],
            'Stage_0/Unit_2': [batch_size, 28, 28, 244],
            'Stage_0/Unit_3': [batch_size, 28, 28, 244],
            # Stage 2 with 8 units
            'Stage_1/Unit_0': [batch_size, 14, 14, 488],
            'Stage_1/Unit_1': [batch_size, 14, 14, 488],
            'Stage_1/Unit_2': [batch_size, 14, 14, 488],
            'Stage_1/Unit_3': [batch_size, 14, 14, 488],
            'Stage_1/Unit_4': [batch_size, 14, 14, 488],
            'Stage_1/Unit_5': [batch_size, 14, 14, 488],
            'Stage_1/Unit_6': [batch_size, 14, 14, 488],
            'Stage_1/Unit_7': [batch_size, 14, 14, 488],
            # Stage 3 with 4 units
            'Stage_2/Unit_0': [batch_size, 7, 7, 976],
            'Stage_2/Unit_1': [batch_size, 7, 7, 976],
            'Stage_2/Unit_2': [batch_size, 7, 7, 976],
            'Stage_2/Unit_3': [batch_size, 7, 7, 976],
            # Conv2 to mix up features
            'Conv2d_1': [batch_size, 7, 7, 2048]
        }

        self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
        for endpoint_name, expected_shape in endpoints_shapes.items():
            self.assertTrue(endpoint_name in end_points)
            self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                                 expected_shape)

    # 10th test
    def testBuild1_5xBasicNetworkAndCheckAllEndPointsUptoConv2d_1(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            _, end_points = shufflenet_v2.shufflenet_v2_base(
                inputs, depth_multiplier=1.5, final_endpoint='Conv2d_1')
        endpoints_shapes = {
            # Conv1 & MaxPool
            'Conv2d_0': [batch_size, 112, 112, 24],
            'MaxPool2d_0': [batch_size, 56, 56, 24],
            # Stage 1 with 4 units
            'Stage_0/Unit_0': [batch_size, 28, 28, 176],
            'Stage_0/Unit_1': [batch_size, 28, 28, 176],
            'Stage_0/Unit_2': [batch_size, 28, 28, 176],
            'Stage_0/Unit_3': [batch_size, 28, 28, 176],
            # Stage 2 with 8 units
            'Stage_1/Unit_0': [batch_size, 14, 14, 352],
            'Stage_1/Unit_1': [batch_size, 14, 14, 352],
            'Stage_1/Unit_2': [batch_size, 14, 14, 352],
            'Stage_1/Unit_3': [batch_size, 14, 14, 352],
            'Stage_1/Unit_4': [batch_size, 14, 14, 352],
            'Stage_1/Unit_5': [batch_size, 14, 14, 352],
            'Stage_1/Unit_6': [batch_size, 14, 14, 352],
            'Stage_1/Unit_7': [batch_size, 14, 14, 352],
            # Stage 3 with 4 units
            'Stage_2/Unit_0': [batch_size, 7, 7, 704],
            'Stage_2/Unit_1': [batch_size, 7, 7, 704],
            'Stage_2/Unit_2': [batch_size, 7, 7, 704],
            'Stage_2/Unit_3': [batch_size, 7, 7, 704],
            # Conv2 to mix up features
            'Conv2d_1': [batch_size, 7, 7, 1024]
        }

        self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
        for endpoint_name, expected_shape in endpoints_shapes.items():
            self.assertTrue(endpoint_name in end_points)
            self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                                 expected_shape)

    # 11th test
    def testBuild0_5xBasicNetworkAndCheckAllEndPointsUptoConv2d_1(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                            normalizer_fn=slim.batch_norm):
            _, end_points = shufflenet_v2.shufflenet_v2_base(
                inputs, depth_multiplier=0.5, final_endpoint='Conv2d_1')
        endpoints_shapes = {
            # Conv1 & MaxPool
            'Conv2d_0': [batch_size, 112, 112, 24],
            'MaxPool2d_0': [batch_size, 56, 56, 24],
            # Stage 1 with 4 units
            'Stage_0/Unit_0': [batch_size, 28, 28, 48],
            'Stage_0/Unit_1': [batch_size, 28, 28, 48],
            'Stage_0/Unit_2': [batch_size, 28, 28, 48],
            'Stage_0/Unit_3': [batch_size, 28, 28, 48],
            # Stage 2 with 8 units
            'Stage_1/Unit_0': [batch_size, 14, 14, 96],
            'Stage_1/Unit_1': [batch_size, 14, 14, 96],
            'Stage_1/Unit_2': [batch_size, 14, 14, 96],
            'Stage_1/Unit_3': [batch_size, 14, 14, 96],
            'Stage_1/Unit_4': [batch_size, 14, 14, 96],
            'Stage_1/Unit_5': [batch_size, 14, 14, 96],
            'Stage_1/Unit_6': [batch_size, 14, 14, 96],
            'Stage_1/Unit_7': [batch_size, 14, 14, 96],
            # Stage 3 with 4 units
            'Stage_2/Unit_0': [batch_size, 7, 7, 192],
            'Stage_2/Unit_1': [batch_size, 7, 7, 192],
            'Stage_2/Unit_2': [batch_size, 7, 7, 192],
            'Stage_2/Unit_3': [batch_size, 7, 7, 192],
            # Conv2 to mix up features
            'Conv2d_1': [batch_size, 7, 7, 1024]
        }

        self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
        for endpoint_name, expected_shape in endpoints_shapes.items():
            self.assertTrue(endpoint_name in end_points)
            self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                                 expected_shape)

    # 12th test
    def testRaiseValueErrorWithInvalidDepthMultiplier(self):
        batch_size = 5
        height, width = 224, 224
        num_classes = 1000
        inputs = tf.random_uniform((batch_size, height, width, 3))
        with self.assertRaises(ValueError):
            _ = shufflenet_v2.shufflenet_v2_base(
                inputs, num_classes, depth_multiplier=-0.1)

        with self.assertRaises(ValueError):
            _ = shufflenet_v2.shufflenet_v2_base(
                inputs, num_classes, depth_multiplier=0.75)

    # 13th test
    def testHalfSizeImages(self):
        batch_size = 5
        height, width = 112, 112
        num_classes = 1000

        inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, end_points = shufflenet_v2.shufflenet_v2(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith('ShufflenetV2/Logits'))
        self.assertListEqual(logits.get_shape().as_list(),
                             [batch_size, num_classes])
        pre_pool = end_points['Conv2d_1']
        self.assertListEqual(pre_pool.get_shape().as_list(),
                             [batch_size, 4, 4, 1024])

    # 14th test
    def testUnknownImageShape(self):
        tf.reset_default_graph()
        batch_size = 2
        height, width = 224, 224
        num_classes = 1000
        input_np = np.random.uniform(0, 1, (batch_size, height, width, 3))
        with self.test_session() as sess:
            inputs = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3))
            logits, end_points = shufflenet_v2.shufflenet_v2(inputs, num_classes)
            self.assertTrue(logits.op.name.startswith('ShufflenetV2/Logits'))
            self.assertListEqual(logits.get_shape().as_list(),
                                 [batch_size, num_classes])
            pre_pool = end_points['Conv2d_1']
            feed_dict = {inputs: input_np}
            tf.global_variables_initializer().run()
            pre_pool_out = sess.run(pre_pool, feed_dict=feed_dict)
            self.assertListEqual(list(pre_pool_out.shape), [batch_size, 7, 7, 1024])

    # 15th test
    def testGlobalPoolUnknownImageShape(self):
        tf.reset_default_graph()
        batch_size = 1
        height, width = 250, 300
        num_classes = 1000
        input_np = np.random.uniform(0, 1, (batch_size, height, width, 3))
        with self.test_session() as sess:
            inputs = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3))
            logits, end_points = shufflenet_v2.shufflenet_v2(inputs, num_classes,
                                                             global_pool=True)
            self.assertTrue(logits.op.name.startswith('ShufflenetV2/Logits'))
            self.assertListEqual(logits.get_shape().as_list(),
                                 [batch_size, num_classes])
            pre_pool = end_points['Conv2d_1']
            feed_dict = {inputs: input_np}
            tf.global_variables_initializer().run()
            pre_pool_out = sess.run(pre_pool, feed_dict=feed_dict)
            self.assertListEqual(list(pre_pool_out.shape), [batch_size, 8, 10, 1024])

    # 16th test
    def testUnknowBatchSize(self):
        batch_size = 1
        height, width = 224, 224
        num_classes = 1000

        inputs = tf.placeholder(tf.float32, (None, height, width, 3))
        logits, _ = shufflenet_v2.shufflenet_v2(inputs, num_classes)
        self.assertTrue(logits.op.name.startswith('ShufflenetV2/Logits'))
        self.assertListEqual(logits.get_shape().as_list(),
                             [None, num_classes])
        images = tf.random_uniform((batch_size, height, width, 3))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(logits, {inputs: images.eval()})
            self.assertEquals(output.shape, (batch_size, num_classes))

    # 17th test
    def testEvaluation(self):
        batch_size = 2
        height, width = 224, 224
        num_classes = 1000

        eval_inputs = tf.random_uniform((batch_size, height, width, 3))
        logits, _ = shufflenet_v2.shufflenet_v2(eval_inputs, num_classes,
                                                is_training=False)
        predictions = tf.argmax(logits, 1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(predictions)
            self.assertEquals(output.shape, (batch_size,))

    # 18th test
    def testTrainEvalWithReuse(self):
        train_batch_size = 5
        eval_batch_size = 2
        height, width = 150, 150
        num_classes = 1000

        train_inputs = tf.random_uniform((train_batch_size, height, width, 3))
        shufflenet_v2.shufflenet_v2(train_inputs, num_classes)
        eval_inputs = tf.random_uniform((eval_batch_size, height, width, 3))
        logits, _ = shufflenet_v2.shufflenet_v2(eval_inputs, num_classes,
                                                reuse=True)
        predictions = tf.argmax(logits, 1)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(predictions)
            self.assertEquals(output.shape, (eval_batch_size,))

    # 19th test
    def testLogitsNotSqueezed(self):
        num_classes = 25
        images = tf.random_uniform([1, 224, 224, 3])
        logits, _ = shufflenet_v2.shufflenet_v2(images,
                                                num_classes=num_classes,
                                                spatial_squeeze=False)

        with self.test_session() as sess:
            tf.global_variables_initializer().run()
            logits_out = sess.run(logits)
            self.assertListEqual(list(logits_out.shape), [1, 1, 1, num_classes])

    # 20th test
    def testBatchNormScopeDoesNotHaveIsTrainingWhenItsSetToNone(self):
        sc = shufflenet_v2.shufflenet_v2_arg_scope(is_training=None)
        self.assertNotIn('is_training', sc[slim.arg_scope_func_key(
            slim.batch_norm)])

    # 21th test
    def testBatchNormScopeDoesHasIsTrainingWhenItsNotNone(self):
        sc = shufflenet_v2.shufflenet_v2_arg_scope(is_training=True)
        self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])
        sc = shufflenet_v2.shufflenet_v2_arg_scope(is_training=False)
        self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])
        sc = shufflenet_v2.shufflenet_v2_arg_scope()
        self.assertIn('is_training', sc[slim.arg_scope_func_key(slim.batch_norm)])

    # 22th test
    def testBasicNetworkModelHasExpectedNumberOfParameters(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))

        endpoints_num_params = {
            # Conv1 & MaxPool
            'Conv2d_0': 720, 'MaxPool2d_0': 720,
            # Stage 1 with 4 units
            'Stage_0/Unit_0': 8374, 'Stage_0/Unit_1': 16146,
            'Stage_0/Unit_2': 23918, 'Stage_0/Unit_3': 31690,
            # Stage 2 with 8 units
            'Stage_1/Unit_0': 75886, 'Stage_1/Unit_1': 104886,
            'Stage_1/Unit_2': 133886, 'Stage_1/Unit_3': 162886,
            'Stage_1/Unit_4': 191886, 'Stage_1/Unit_5': 220886,
            'Stage_1/Unit_6': 249886, 'Stage_1/Unit_7': 278886,
            # Stage 3 with 4 units
            'Stage_2/Unit_0': 448014, 'Stage_2/Unit_1': 559838,
            'Stage_2/Unit_2': 671662, 'Stage_2/Unit_3': 783486,
            # Conv2d_1
            'Conv2d_1': 1261694,
        }
        for scope, end_point in enumerate(endpoints_num_params):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                normalizer_fn=slim.batch_norm):
                shufflenet_v2.shufflenet_v2_base(inputs, final_endpoint=end_point, scope=str(scope))
                total_params, _ = slim.model_analyzer.analyze_vars(
                    slim.get_model_variables(scope=str(scope)))

                self.assertAlmostEqual(endpoints_num_params[end_point], total_params)

    # 23th test
    def test0_5xBasicNetworkModelHasExpectedNumberOfParameters(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))

        endpoints_num_params = {
            # Conv1 & MaxPool
            'Conv2d_0': 720, 'MaxPool2d_0': 720,
            # Stage 1 with 4 units
            'Stage_0/Unit_0': 3240, 'Stage_0/Unit_1': 4824,
            'Stage_0/Unit_2': 6408, 'Stage_0/Unit_3': 7992,
            # Stage 2 with 8 units
            'Stage_1/Unit_0': 16488, 'Stage_1/Unit_1': 21960,
            'Stage_1/Unit_2': 27432, 'Stage_1/Unit_3': 32904,
            'Stage_1/Unit_4': 38376, 'Stage_1/Unit_5': 43848,
            'Stage_1/Unit_6': 49320, 'Stage_1/Unit_7': 54792,
            # Stage 3 with 4 units
            'Stage_2/Unit_0': 85608, 'Stage_2/Unit_1': 105768,
            'Stage_2/Unit_2': 125928, 'Stage_2/Unit_3': 146088,
            # Conv2d_1
            'Conv2d_1': 345768,
        }
        for scope, end_point in enumerate(endpoints_num_params):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                normalizer_fn=slim.batch_norm):
                shufflenet_v2.shufflenet_v2_base(inputs, final_endpoint=end_point,
                                                 depth_multiplier=0.5, scope=str(scope))
                total_params, _ = slim.model_analyzer.analyze_vars(
                    slim.get_model_variables(scope=str(scope)))

                self.assertAlmostEqual(endpoints_num_params[end_point], total_params)

    # 24th test
    def test1_5xBasicNetworkModelHasExpectedNumberOfParameters(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))

        endpoints_num_params = {
            # Conv1 & MaxPool
            'Conv2d_0': 720, 'MaxPool2d_0': 720,
            # Stage 1 with 4 units
            'Stage_0/Unit_0': 14824, 'Stage_0/Unit_1': 31896,
            'Stage_0/Unit_2': 48968, 'Stage_0/Unit_3': 66040,
            # Stage 2 with 8 units
            'Stage_1/Unit_0': 164776, 'Stage_1/Unit_1': 229896,
            'Stage_1/Unit_2': 295016, 'Stage_1/Unit_3': 360136,
            'Stage_1/Unit_4': 425256, 'Stage_1/Unit_5': 490376,
            'Stage_1/Unit_6': 555496, 'Stage_1/Unit_7': 620616,
            # Stage 3 with 4 units
            'Stage_2/Unit_0': 1003944, 'Stage_2/Unit_1': 1258088,
            'Stage_2/Unit_2': 1512232, 'Stage_2/Unit_3': 1766376,
            # Conv2d_1
            'Conv2d_1': 2490344,
        }
        for scope, end_point in enumerate(endpoints_num_params):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                normalizer_fn=slim.batch_norm):
                shufflenet_v2.shufflenet_v2_base(inputs, final_endpoint=end_point,
                                                 depth_multiplier=1.5, scope=str(scope))
                total_params, _ = slim.model_analyzer.analyze_vars(
                    slim.get_model_variables(scope=str(scope)))

                self.assertAlmostEqual(endpoints_num_params[end_point], total_params)

    # 25th test
    def test2_0xBasicNetworkModelHasExpectedNumberOfParameters(self):
        batch_size = 5
        height, width = 224, 224
        inputs = tf.random_uniform((batch_size, height, width, 3))

        endpoints_num_params = {
            # Conv1 & MaxPool
            'Conv2d_0': 720, 'MaxPool2d_0': 720,
            # Stage 1 with 4 units
            'Stage_0/Unit_0': 24310, 'Stage_0/Unit_1': 56274,
            'Stage_0/Unit_2': 88238, 'Stage_0/Unit_3': 120202,
            # Stage 2 with 8 units
            'Stage_1/Unit_0': 306862, 'Stage_1/Unit_1': 430326,
            'Stage_1/Unit_2': 553790, 'Stage_1/Unit_3': 677254,
            'Stage_1/Unit_4': 800718, 'Stage_1/Unit_5': 924182,
            'Stage_1/Unit_6': 1047646, 'Stage_1/Unit_7': 1171110,
            # Stage 3 with 4 units
            'Stage_2/Unit_0': 1901646, 'Stage_2/Unit_1': 2386718,
            'Stage_2/Unit_2': 2871790, 'Stage_2/Unit_3': 3356862,
            # Conv2d_1
            'Conv2d_1': 5361854,
        }
        for scope, end_point in enumerate(endpoints_num_params):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                normalizer_fn=slim.batch_norm):
                shufflenet_v2.shufflenet_v2_base(inputs, final_endpoint=end_point,
                                                 depth_multiplier=2.0, scope=str(scope))
                total_params, _ = slim.model_analyzer.analyze_vars(
                    slim.get_model_variables(scope=str(scope)))

                self.assertAlmostEqual(endpoints_num_params[end_point], total_params)


if __name__ == '__main__':
    tf.test.main()
