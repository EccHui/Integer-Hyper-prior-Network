import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import numpy as np

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

class IntegerConv2DTranspose(tf.keras.layers.Layer):
  def __init__(self, num_filters, kernel_size, strides, network_precision, 
    output_precision, *args, **kwargs):
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.K = network_precision
    self.L = output_precision
    super(IntegerConv2DTranspose, self).__init__(*args, **kwargs)
  
  def build(self, input_shape):
    self.kernel = self.add_weight(name='kernel', 
      shape=(self.kernel_size, self.kernel_size, self.num_filters, input_shape[-1]), 
      initializer='random_normal', trainable=True)
    self.bias = self.add_weight(name='bias', 
      shape=(1, 1, 1, self.num_filters), initializer='uniform', 
      trainable=True)
    self.c = self.add_weight(name='c', 
      shape=(1, 1, 1, self.num_filters), initializer='uniform', 
      trainable=True)
    self.conv_shape = [input_shape[0], self.strides[0]*input_shape[1], 
      self.strides[1]*input_shape[2], self.num_filters]
    super(IntegerConv2DTranspose, self).build(input_shape)

  def call(self, x):
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
      c_thd = np.sqrt(1 + 2**-16)
      c_scale = (tf.nn.relu(c - c_thd) + c_thd)**2 - 2**-16
      s = tf.cast( 2**K, dtype=tf.float32)
      c = tf.round( s * c_scale )
      def grad(dy):
        idx = tf.cast(tf.logical_or(tf.where(c>=c_thd), tf.where(dy<0)), dtype=tf.float32)
        return dy * idx * s,  tf.zeros_like(K)
      return c, grad

    kernel = kernel_quantizer(self.kernel, self.K)
    bias = bias_quantizer(self.bias, self.K)
    c = c_quantizer(self.c, self.K)

    y = tf.nn.conv2d_transpose(x, kernel, self.conv_shape, strides=self.strides, padding="SAME")
    x = y + tf.broadcast_to(bias, self.conv_shape)
    x = tf.divide(x, tf.broadcast_to(c, self.conv_shape))
    
    x = qrelu(x, 2**self.L-1)
    return x

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
