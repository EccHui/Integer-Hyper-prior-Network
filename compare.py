import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import numpy as np

from integer_conv2d_transpose import IntegerConv2DTranspose

class HyperSynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperSynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
            padding="same_zeros", use_bias=True, kernel_parameterizer=None,
            activation=None),
    ]
    super(HyperSynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class IntegerHyperSynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(IntegerHyperSynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        IntegerConv2DTranspose(self.num_filters, 5, [2,2], network_precision=8, output_precision=8),
        IntegerConv2DTranspose(self.num_filters, 5, [2,2], network_precision=8, output_precision=8),
        IntegerConv2DTranspose(self.num_filters, 3, [1,1], network_precision=8, output_precision=6),
    ]
    super(IntegerHyperSynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor

x = np.random.random_integers(-50,50,[1, 32, 32, 192])
x = tf.round( tf.constant(x, dtype=tf.float32) )

integer_network = IntegerHyperSynthesisTransform(192)
float_network = HyperSynthesisTransform(192)

y = integer_network(x)
z = float_network(x)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  with tf.device('/cpu:0'):
    y_cpu = sess.run(y)
    z_cpu = sess.run(z)
  with tf.device('/gpu:0'):
    y_gpu = sess.run(y)
    z_gpu = sess.run(z)
  print( "Integer Network Determinism:", (y_cpu == y_gpu).all(), ", Error:{}".format(np.mean(y_cpu-y_gpu)) )
  print( "Float Network Determinism:", (z_cpu == z_gpu).all(), ", Error:{}".format(np.mean(z_cpu-z_gpu))  )
