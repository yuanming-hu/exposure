import tensorflow as tf
import tf_slim as ly
from util import lrelu


def cnn(net, is_train, cfg):
  net = net - 0.5
  channels = cfg.base_channels
  size = int(net.get_shape()[2])
  print('Critic CNN:')
  print('    ', str(net.get_shape()))
  size /= 2
  net = ly.conv2d(
      net,
      num_outputs=channels,
      kernel_size=4,
      stride=2,
      activation_fn=lrelu,
      normalizer_fn=None)
  print('    ', str(net.get_shape()))
  while size > 4:
    channels *= 2
    size /= 2
    net = ly.conv2d(
        net,
        num_outputs=channels,
        kernel_size=4,
        stride=2,
        activation_fn=lrelu,
        normalizer_fn=None,
        normalizer_params={
            'is_training': is_train,
            'decay': 0.9,
            'updates_collections': None
        })
    print('    ', str(net.get_shape()))
  net = tf.reshape(net, [-1, 4 * 4 * channels])
  return net


# Input: float \in [0, 1]
def critic(images, cfg, states=None, is_train=None, reuse=False):
  with tf.compat.v1.variable_scope('critic') as scope:
    if reuse:
      scope.reuse_variables()

    if True:
      lum = (images[:, :, :, 0] * 0.27 + images[:, :, :, 1] * 0.67 +
             images[:, :, :, 2] * 0.06 + 1e-5)[:, :, :]
      # luminance and contrast
      luminance, contrast = tf.nn.moments(x=lum, axes=[1, 2])

      # saturation
      i_max = tf.reduce_max(
          input_tensor=tf.clip_by_value(images, clip_value_min=0.0, clip_value_max=1.0),
          axis=[3])
      i_min = tf.reduce_min(
          input_tensor=tf.clip_by_value(images, clip_value_min=0.0, clip_value_max=1.0),
          axis=[3])
      sat = (i_max - i_min) / (
          tf.minimum(x=i_max + i_min, y=2.0 - i_max - i_min) + 1e-2)
      saturation, _ = tf.nn.moments(x=sat, axes=[1, 2])

      repeatition = 1

      stat_feature = tf.concat(
          [
              tf.tile(luminance[:, None], [1, repeatition]),
              tf.tile(contrast[:, None], [1, repeatition]),
              tf.tile(saturation[:, None], [1, repeatition])
          ],
          axis=1)

      print('stats ', stat_feature.shape)

      if states is None:
        states = stat_feature
      else:
        assert len(states.shape) == len(stat_feature.shape)
        states = tf.concat([states, stat_feature], axis=1)

    if True:
      if states is not None:
        print('States:', states.shape)
        states = states[:, None, None, :] + (images[:, :, :, 0:1] * 0)
        print('     States:', states.shape)
        images = tf.concat([images, states], axis=3)

      cnn_feature = cnn(images, cfg=cfg, is_train=is_train)
      print('     CNN shape: ', cnn_feature.shape)
      net = cnn_feature

    print('Before final FCs', net.shape)
    net = ly.fully_connected(net, cfg.fc1_size, activation_fn=lrelu)
    print('     ', net.shape)

    outputs = ly.fully_connected(net, 1, activation_fn=None)
  return outputs, None, None
