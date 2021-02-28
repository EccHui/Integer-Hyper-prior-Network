import tensorflow.compat.v1 as tf
import numpy as np

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
      k_max = tf.reduce_max(kernel, axis=(0,1), keepdims=True) / upper
      k_min = tf.reduce_min(kernel, axis=(0,1), keepdims=True) / lower
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
        idx = tf.cast(tf.logical_or(c>=c_thd, dy<0), dtype=tf.float32)
        return dy * idx * s,  tf.zeros_like(K)
      return c, grad

    kernel = kernel_quantizer(self.kernel, self.K)
    bias = bias_quantizer(self.bias, self.K)
    c = c_quantizer(self.c, self.K)

    x = tf.nn.conv2d_transpose(x, kernel, self.conv_shape, strides=self.strides, padding="SAME")
    x = x + tf.broadcast_to(bias, self.conv_shape)
    x = tf.divide(x, tf.broadcast_to(c, self.conv_shape))
    
    x = qrelu(x, 2**self.L-1)
    return x
