# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Nonlinear transform coder with hyperprior for RGB images.

This is the image compression model published in:
J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
"Variational Image Compression with a Scale Hyperprior"
Int. Conf. on Learning Representations (ICLR), 2018
https://arxiv.org/abs/1802.01436

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.

Currently, this script requires tensorflow-compression v1.3.
"""

import argparse
import glob
import sys

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def read_png(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)


class AnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(AnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_0")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_1")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_2")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(AnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class SynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(SynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_2", inverse=True)),
        tfc.SignalConv2D(
            3, (5, 5), name="layer_3", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(SynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class HyperAnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform for the entropy model parameters."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(HyperAnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
            padding="same_zeros", use_bias=True,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=False,
            activation=None),
    ]
    super(HyperAnalysisTransform, self).build(input_shape)

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

    x_shape = tf.shape(x)
    deconv_shape = self.conv_shape = [x_shape[0], self.strides[0]*x_shape[1], 
      self.strides[1]*x_shape[2], self.num_filters]
    
    x = tf.nn.conv2d_transpose(x, kernel, deconv_shape, strides=self.strides, padding="SAME")
    x = x + tf.broadcast_to(bias, self.conv_shape)
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


def train(args):
  """Trains the model."""

  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device("/cpu:0"):
    train_files = glob.glob(args.train_glob)
    if not train_files:
      raise RuntimeError(
          "No training images found with glob '{}'.".format(args.train_glob))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        read_png, num_parallel_calls=args.preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(32)

  num_pixels = args.batchsize * args.patchsize ** 2

  # Get training patch from dataset.
  x = train_dataset.make_one_shot_iterator().get_next()

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  synthesis_transform = SynthesisTransform(args.num_filters)
  hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
  hyper_synthesis_transform = IntegerHyperSynthesisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()

  # Build autoencoder and hyperprior.
  y = analysis_transform(x)
  z = hyper_analysis_transform(abs(y))
  z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
  sigma_ = hyper_synthesis_transform(z_tilde)
  sigma = tf.exp(np.log(SCALES_MIN) + (np.log(SCALES_MAX) - np.log(SCALES_MIN)) / (SCALES_LEVELS-1) * sigma_ )
  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table)
  y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)
  x_tilde = synthesis_transform(y_tilde)

  # Total number of bits divided by number of pixels.
  train_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
               tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2

  # The rate-distortion cost.
  train_loss = args.lmbda * train_mse + train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", train_bpp)
  tf.summary.scalar("mse", train_mse)

  tf.summary.image("original", quantize_image(x))
  tf.summary.image("reconstruction", quantize_image(x_tilde))

  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]
  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=args.checkpoint_dir,
      save_checkpoint_secs=300, save_summaries_secs=60) as sess:
    while not sess.should_stop():
      sess.run(train_op)


def compress(args):
  """Compresses an image."""

  # Load input image and add batch dimension.
  x = read_png(args.input_file)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])
  x_shape = tf.shape(x)

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  synthesis_transform = SynthesisTransform(args.num_filters)
  hyper_analysis_transform = HyperAnalysisTransform(args.num_filters)
  hyper_synthesis_transform = IntegerHyperSynthesisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()

  # Transform and compress the image.
  y = analysis_transform(x)
  y_shape = tf.shape(y)
  z = hyper_analysis_transform(abs(y))
  z_hat, z_likelihoods = entropy_bottleneck(z, training=False)
  sigma_idx = hyper_synthesis_transform(z_hat)
  sigma_idx = sigma_idx[:, :y_shape[1], :y_shape[2], :]
  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  # `scale` would be overrided when indexes are given
  conditional_bottleneck = tfc.GaussianConditional(sigma_idx, scale_table, 
      indexes=tf.cast(sigma_idx, dtype=tf.int32))
  side_string = entropy_bottleneck.compress(z)
  string = conditional_bottleneck.compress(y)

  # Transform the quantized image back (if requested).
  y_hat, y_likelihoods = conditional_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat)
  x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]

  num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

  # Total number of bits divided by number of pixels.
  eval_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
              tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

  # Bring both images back to 0..255 range.
  x *= 255
  x_hat = tf.clip_by_value(x_hat, 0, 1)
  x_hat = tf.round(x_hat * 255)

  mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
  psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
  msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    tensors = [string, side_string,
               tf.shape(x)[1:-1], tf.shape(y)[1:-1], tf.shape(z)[1:-1]]
    arrays = sess.run(tensors)

    # Write a binary file with the shape information and the compressed string.
    packed = tfc.PackedTensors()
    packed.pack(tensors, arrays)
    with open(args.output_file, "wb") as f:
      f.write(packed.string)

    # If requested, transform the quantized image back and measure performance.
    if args.verbose:
      eval_bpp, mse, psnr, msssim, num_pixels = sess.run(
          [eval_bpp, mse, psnr, msssim, num_pixels])

      # The actual bits per pixel including overhead.
      bpp = len(packed.string) * 8 / num_pixels

      print("Mean squared error: {:0.4f}".format(mse))
      print("PSNR (dB): {:0.2f}".format(psnr))
      print("Multiscale SSIM: {:0.4f}".format(msssim))
      print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim)))
      print("Information content in bpp: {:0.4f}".format(eval_bpp))
      print("Actual bits per pixel: {:0.4f}".format(bpp))

      with open("results.txt", "a+") as f:
        f.write("Input file: \t{}\n".format(args.input_file))
        f.write("Mean squared error: \t{:0.4f}\n".format(mse))
        f.write("PSNR (dB): \t{:0.2f}\n".format(psnr))
        f.write("Multiscale SSIM: \t{:0.4f}\n".format(msssim))
        f.write("Multiscale SSIM (dB): \t{:0.2f}\n".format(-10 * np.log10(1 - msssim)))
        f.write("Information content in bpp: \t{:0.4f}\n".format(eval_bpp))
        f.write("Actual bits per pixel: \t{:0.4f}\n".format(bpp))
        f.write("\n")


def decompress(args):
  """Decompresses an image."""

  # Read the shape information and compressed string from the binary file.
  string = tf.placeholder(tf.string, [1])
  side_string = tf.placeholder(tf.string, [1])
  x_shape = tf.placeholder(tf.int32, [2])
  y_shape = tf.placeholder(tf.int32, [2])
  z_shape = tf.placeholder(tf.int32, [2])
  with open(args.input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  tensors = [string, side_string, x_shape, y_shape, z_shape]
  arrays = packed.unpack(tensors)

  # Instantiate model.
  synthesis_transform = SynthesisTransform(args.num_filters)
  hyper_synthesis_transform = IntegerHyperSynthesisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)

  # Decompress and transform the image back.
  z_shape = tf.concat([z_shape, [args.num_filters]], axis=0)
  z_hat = entropy_bottleneck.decompress(
      side_string, z_shape, channels=args.num_filters)
  sigma_idx = hyper_synthesis_transform(z_hat)
  sigma_idx = sigma_idx[:, :y_shape[0], :y_shape[1], :]
  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(sigma_idx, scale_table, 
      indexes=tf.cast(sigma_idx, dtype=tf.int32), dtype=tf.float32)
  y_hat = conditional_bottleneck.decompress(string)
  x_hat = synthesis_transform(y_hat)

  # Remove batch dimension, and crop away any extraneous padding on the bottom
  # or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = write_png(args.output_file, x_hat)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op, feed_dict=dict(zip(tensors, arrays)))


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=192,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="models",
      help="Directory where to save/load model checkpoints.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model.")
  train_cmd.add_argument(
      "--train_glob", default="/home/lsh/Data/PNG_DATA/train/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.02, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help="Output filename (optional). If not provided, appends '{}' to "
             "the input filename.".format(ext))

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".tfci"
    compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
