import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import numpy as np

num_filters = 192
epsilon = 2**(-8)

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

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

@tf.custom_gradient
def qrelu(x, upper):
  upper = tf.cast(upper, dtype=tf.float32)
  def grad(dy):
    alpha = -0.674969789311173 # (1/4*Gamma(1/4))**4
    return dy * tf.exp( alpha * (tf.abs(2.0/upper*x - 1))**4 ), tf.zeros_like(upper)
  return tf.round(tf.clip_by_value(x, 0, upper)), grad

@tf.custom_gradient
def kernel_quantizer(kernel, K):
  upper = tf.cast(2**(K-1) - 1, dtype=tf.float32)
  lower = - upper - 1
  k_max = tf.reduce_max(tf.reduce_max(kernel, axis=0, keepdims=True), axis=1, keepdims=True) / upper
  k_min = tf.reduce_min(tf.reduce_min(kernel, axis=0, keepdims=True), axis=1, keepdims=True) / lower
  k_scale = tf.nn.relu(tf.maximum(k_max, k_min) - 1e-20) + 1e-20
  kernel = tf.round(tf.divide(kernel, tf.broadcast_to(k_scale, tf.shape(kernel))))
  def grad(dy):
    return tf.divide(dy, tf.broadcast_to(k_scale, tf.shape(dy))), tf.zeros_like(K)
  return kernel, grad

@tf.custom_gradient
def bias_quantizer(bias, K):
  s = tf.cast( 2**K, dtype=tf.float32)
  bias = tf.round( s * bias )
  def grad(dy):
    return dy * s, tf.zeros_like(K)
  return bias, grad

@tf.custom_gradient
def c_quantizer(c, K):
  c_thd = np.sqrt(1 + epsilon**2)
  c_scale = (tf.nn.relu(c - c_thd) + c_thd)**2 - epsilon**2
  s = tf.cast( 2**K, dtype=tf.float32)
  c = tf.round( s * c_scale )
  def grad(dy):
    idx = tf.cast(tf.logical_or(tf.where(c>=c_thd), tf.where(dy<0)), dtype=tf.float32)
    return dy * idx * s,  tf.zeros_like(K)
  return c, grad

def hyper_synthesis_transform(x):
  K = 8
  for i in range(3):
    with tf.name_scope( "layer_{}".format(i) ):
      x_shape = tf.shape(x)
      if i < 2:
        y_shape = [x_shape[0],2*x_shape[1],2*x_shape[2],num_filters]
        strides = [2, 2]
        kernel_size = 5
      else:
        y_shape = x_shape
        strides = [1, 1]
        kernel_size = 3

      kernel = tf.Variable(tf.random_normal([kernel_size,kernel_size,num_filters,x_shape[-1]], stddev=1), trainable=True)
      bias = tf.Variable(tf.random_normal([1,1,1,num_filters], stddev=1), trainable=True)
      c = tf.Variable(tf.random_normal([1,1,1,num_filters], stddev=1), trainable=True)

      kernel = kernel_quantizer(kernel, K)
      bias = bias_quantizer(bias, K)
      c = c_quantizer(c, K)

      y = tf.nn.conv2d_transpose(x, kernel, y_shape, strides=strides, padding="SAME")
      x = y + tf.broadcast_to(bias, y_shape)
      x = tf.divide(x, tf.broadcast_to(c, y_shape))

      if i < 2:
        x = qrelu(x, 255)
      else:
        x = qrelu(x, SCALES_LEVELS-1)

  return x

x = np.random.random_integers(-50,50,[1, 32, 32, 192])
x = tf.round( tf.constant(x, dtype=tf.float32) )
y = hyper_synthesis_transform(x)

hyper2 = HyperSynthesisTransform(192)
z = hyper2(x)

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