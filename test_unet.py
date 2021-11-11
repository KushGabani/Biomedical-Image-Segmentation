from unet import UNet
import tensorflow as tf
import numpy as np
import unittest


class AssertModelArchitecture(unittest.TestCase):
    def test_architecture_256x256(self):
        model = UNet()
        output = model.call(tf.random.normal((1, 256, 256, 1)))
        del model
        self.assertEqual(output.shape, (1, 256, 256, 1))

    def test_architecture_512x512(self):
        model = UNet()
        output = model.call(tf.random.normal((1, 512, 512, 1)))
        del model
        self.assertEqual(output.shape,  (1, 512, 512, 1))

    def test_architecture_1024x1024(self):
        model = UNet()
        output = model.call(tf.random.normal((1, 1024, 1024, 1)))
        del model
        self.assertEqual(output.shape, (1, 1024, 1024, 1))

    def test_architecture_512x512x3(self):
        model = UNet()
        output = model.call(tf.random.normal((1, 512, 512, 3)))
        del model
        self.assertEqual(output.shape, (1, 512, 512, 1))

    def test_architecture_1024x1024x3(self):
        model = UNet()
        output = model.call(tf.random.normal((1, 1024, 1024, 3)))
        del model
        self.assertEqual(output.shape, (1, 1024, 1024, 1))

    def test_output_classes(self):
        model = UNet(n_classes=3)
        output = model.call(tf.random.normal((1, 512, 512, 1)))
        del model
        self.assertEqual(output.shape,  (1, 512, 512, 3))

    def test_trainable_weight_count(self):
        model = UNet()
        model.build(input_shape=(1, 128, 128, 1))
        trainable_weight_count = np.sum([
            tf.keras.backend.count_params(weights) for weights in model.trainable_weights
        ])
        del model
        self.assertEqual(trainable_weight_count, 31024704)

    def test_trainable_weight_count_RGB(self):
        model = UNet()
        model.build(input_shape=(1, 128, 128, 3))
        trainable_weight_count = np.sum([
            tf.keras.backend.count_params(weights) for weights in model.trainable_weights
        ])
        del model
        self.assertEqual(trainable_weight_count, 31025856)
