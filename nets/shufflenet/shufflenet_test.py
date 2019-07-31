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

from nets.shufflenet import shufflenet


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
            'Shufflenet/conv1',
            'Shufflenet/pool1',

            'Shufflenet/stage1/shufflenet_unit/group_conv1/group_conv1',
            'Shufflenet/stage1/shufflenet_unit/depthwise_conv',
            'Shufflenet/stage1/shufflenet_unit/group_conv2/group_conv2',
            'Shufflenet/stage1/shufflenet_unit',
            'Shufflenet/stage1/shufflenet_unit_1/group_conv1/group0',
            'Shufflenet/stage1/shufflenet_unit_1/group_conv1/group1',
            'Shufflenet/stage1/shufflenet_unit_1/group_conv1/group2',
            'Shufflenet/stage1/shufflenet_unit_1/depthwise_conv',
            'Shufflenet/stage1/shufflenet_unit_1/group_conv2/group0',
            'Shufflenet/stage1/shufflenet_unit_1/group_conv2/group1',
            'Shufflenet/stage1/shufflenet_unit_1/group_conv2/group2',
            'Shufflenet/stage1/shufflenet_unit_1',
            'Shufflenet/stage1/shufflenet_unit_2/group_conv1/group0',
            'Shufflenet/stage1/shufflenet_unit_2/group_conv1/group1',
            'Shufflenet/stage1/shufflenet_unit_2/group_conv1/group2',
            'Shufflenet/stage1/shufflenet_unit_2/depthwise_conv',
            'Shufflenet/stage1/shufflenet_unit_2/group_conv2/group0',
            'Shufflenet/stage1/shufflenet_unit_2/group_conv2/group1',
            'Shufflenet/stage1/shufflenet_unit_2/group_conv2/group2',
            'Shufflenet/stage1/shufflenet_unit_2',
            'Shufflenet/stage1/shufflenet_unit_3/group_conv1/group0',
            'Shufflenet/stage1/shufflenet_unit_3/group_conv1/group1',
            'Shufflenet/stage1/shufflenet_unit_3/group_conv1/group2',
            'Shufflenet/stage1/shufflenet_unit_3/depthwise_conv',
            'Shufflenet/stage1/shufflenet_unit_3/group_conv2/group0',
            'Shufflenet/stage1/shufflenet_unit_3/group_conv2/group1',
            'Shufflenet/stage1/shufflenet_unit_3/group_conv2/group2',
            'Shufflenet/stage1/shufflenet_unit_3',
            'Shufflenet/stage1',
            'Shufflenet/stage2/shufflenet_unit/group_conv1/group0',
            'Shufflenet/stage2/shufflenet_unit/group_conv1/group1',
            'Shufflenet/stage2/shufflenet_unit/group_conv1/group2',
            'Shufflenet/stage2/shufflenet_unit/depthwise_conv',
            'Shufflenet/stage2/shufflenet_unit/group_conv2/group0',
            'Shufflenet/stage2/shufflenet_unit/group_conv2/group1',
            'Shufflenet/stage2/shufflenet_unit/group_conv2/group2',
            'Shufflenet/stage2/shufflenet_unit',
            'Shufflenet/stage2/shufflenet_unit_1/group_conv1/group0',
            'Shufflenet/stage2/shufflenet_unit_1/group_conv1/group1',
            'Shufflenet/stage2/shufflenet_unit_1/group_conv1/group2',
            'Shufflenet/stage2/shufflenet_unit_1/depthwise_conv',
            'Shufflenet/stage2/shufflenet_unit_1/group_conv2/group0',
            'Shufflenet/stage2/shufflenet_unit_1/group_conv2/group1',
            'Shufflenet/stage2/shufflenet_unit_1/group_conv2/group2',
            'Shufflenet/stage2/shufflenet_unit_1',
            'Shufflenet/stage2/shufflenet_unit_2/group_conv1/group0',
            'Shufflenet/stage2/shufflenet_unit_2/group_conv1/group1',
            'Shufflenet/stage2/shufflenet_unit_2/group_conv1/group2',
            'Shufflenet/stage2/shufflenet_unit_2/depthwise_conv',
            'Shufflenet/stage2/shufflenet_unit_2/group_conv2/group0',
            'Shufflenet/stage2/shufflenet_unit_2/group_conv2/group1',
            'Shufflenet/stage2/shufflenet_unit_2/group_conv2/group2',
            'Shufflenet/stage2/shufflenet_unit_2',
            'Shufflenet/stage2/shufflenet_unit_3/group_conv1/group0',
            'Shufflenet/stage2/shufflenet_unit_3/group_conv1/group1',
            'Shufflenet/stage2/shufflenet_unit_3/group_conv1/group2',
            'Shufflenet/stage2/shufflenet_unit_3/depthwise_conv',
            'Shufflenet/stage2/shufflenet_unit_3/group_conv2/group0',
            'Shufflenet/stage2/shufflenet_unit_3/group_conv2/group1',
            'Shufflenet/stage2/shufflenet_unit_3/group_conv2/group2',
            'Shufflenet/stage2/shufflenet_unit_3',
            'Shufflenet/stage2/shufflenet_unit_4/group_conv1/group0',
            'Shufflenet/stage2/shufflenet_unit_4/group_conv1/group1',
            'Shufflenet/stage2/shufflenet_unit_4/group_conv1/group2',
            'Shufflenet/stage2/shufflenet_unit_4/depthwise_conv',
            'Shufflenet/stage2/shufflenet_unit_4/group_conv2/group0',
            'Shufflenet/stage2/shufflenet_unit_4/group_conv2/group1',
            'Shufflenet/stage2/shufflenet_unit_4/group_conv2/group2',
            'Shufflenet/stage2/shufflenet_unit_4',
            'Shufflenet/stage2/shufflenet_unit_5/group_conv1/group0',
            'Shufflenet/stage2/shufflenet_unit_5/group_conv1/group1',
            'Shufflenet/stage2/shufflenet_unit_5/group_conv1/group2',
            'Shufflenet/stage2/shufflenet_unit_5/depthwise_conv',
            'Shufflenet/stage2/shufflenet_unit_5/group_conv2/group0',
            'Shufflenet/stage2/shufflenet_unit_5/group_conv2/group1',
            'Shufflenet/stage2/shufflenet_unit_5/group_conv2/group2',
            'Shufflenet/stage2/shufflenet_unit_5',
            'Shufflenet/stage2/shufflenet_unit_6/group_conv1/group0',
            'Shufflenet/stage2/shufflenet_unit_6/group_conv1/group1',
            'Shufflenet/stage2/shufflenet_unit_6/group_conv1/group2',
            'Shufflenet/stage2/shufflenet_unit_6/depthwise_conv',
            'Shufflenet/stage2/shufflenet_unit_6/group_conv2/group0',
            'Shufflenet/stage2/shufflenet_unit_6/group_conv2/group1',
            'Shufflenet/stage2/shufflenet_unit_6/group_conv2/group2',
            'Shufflenet/stage2/shufflenet_unit_6',
            'Shufflenet/stage2/shufflenet_unit_7/group_conv1/group0',
            'Shufflenet/stage2/shufflenet_unit_7/group_conv1/group1',
            'Shufflenet/stage2/shufflenet_unit_7/group_conv1/group2',
            'Shufflenet/stage2/shufflenet_unit_7/depthwise_conv',
            'Shufflenet/stage2/shufflenet_unit_7/group_conv2/group0',
            'Shufflenet/stage2/shufflenet_unit_7/group_conv2/group1',
            'Shufflenet/stage2/shufflenet_unit_7/group_conv2/group2',
            'Shufflenet/stage2/shufflenet_unit_7',
            'Shufflenet/stage2',
            'Shufflenet/stage3/shufflenet_unit/group_conv1/group0',
            'Shufflenet/stage3/shufflenet_unit/group_conv1/group1',
            'Shufflenet/stage3/shufflenet_unit/group_conv1/group2',
            'Shufflenet/stage3/shufflenet_unit/depthwise_conv',
            'Shufflenet/stage3/shufflenet_unit/group_conv2/group0',
            'Shufflenet/stage3/shufflenet_unit/group_conv2/group1',
            'Shufflenet/stage3/shufflenet_unit/group_conv2/group2',
            'Shufflenet/stage3/shufflenet_unit',
            'Shufflenet/stage3/shufflenet_unit_1/group_conv1/group0',
            'Shufflenet/stage3/shufflenet_unit_1/group_conv1/group1',
            'Shufflenet/stage3/shufflenet_unit_1/group_conv1/group2',
            'Shufflenet/stage3/shufflenet_unit_1/depthwise_conv',
            'Shufflenet/stage3/shufflenet_unit_1/group_conv2/group0',
            'Shufflenet/stage3/shufflenet_unit_1/group_conv2/group1',
            'Shufflenet/stage3/shufflenet_unit_1/group_conv2/group2',
            'Shufflenet/stage3/shufflenet_unit_1',
            'Shufflenet/stage3/shufflenet_unit_2/group_conv1/group0',
            'Shufflenet/stage3/shufflenet_unit_2/group_conv1/group1',
            'Shufflenet/stage3/shufflenet_unit_2/group_conv1/group2',
            'Shufflenet/stage3/shufflenet_unit_2/depthwise_conv',
            'Shufflenet/stage3/shufflenet_unit_2/group_conv2/group0',
            'Shufflenet/stage3/shufflenet_unit_2/group_conv2/group1',
            'Shufflenet/stage3/shufflenet_unit_2/group_conv2/group2',
            'Shufflenet/stage3/shufflenet_unit_2',
            'Shufflenet/stage3/shufflenet_unit_3/group_conv1/group0',
            'Shufflenet/stage3/shufflenet_unit_3/group_conv1/group1',
            'Shufflenet/stage3/shufflenet_unit_3/group_conv1/group2',
            'Shufflenet/stage3/shufflenet_unit_3/depthwise_conv',
            'Shufflenet/stage3/shufflenet_unit_3/group_conv2/group0',
            'Shufflenet/stage3/shufflenet_unit_3/group_conv2/group1',
            'Shufflenet/stage3/shufflenet_unit_3/group_conv2/group2',
            'Shufflenet/stage3/shufflenet_unit_3',
            'Shufflenet/stage3',
        ]
        expected_endpoints = []
        self.maxDiff = None
        self.assertItemsEqual(endpoints.keys(), expected_endpoints)


if __name__ == '__main__':
    tf.test.main()
