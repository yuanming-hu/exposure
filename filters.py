import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from util import lrelu, rgb2lum, tanh_range, lerp
import cv2
import math


class Filter:

  def __init__(self, net, cfg):
    self.cfg = cfg
    self.height, self.width, self.channels = list(map(int, net.get_shape()[1:]))

    # Specified in child classes
    self.num_filter_parameters = None
    self.short_name = None
    self.filter_parameters = None

  def get_short_name(self):
    assert self.short_name
    return self.short_name

  def get_num_filter_parameters(self):
    assert self.num_filter_parameters
    return self.num_filter_parameters

  def extract_parameters(self, features):
    output_dim = self.get_num_filter_parameters(
    ) + self.get_num_mask_parameters()
    features = ly.fully_connected(
        features,
        self.cfg.fc1_size,
        scope='fc1',
        activation_fn=lrelu,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    features = ly.fully_connected(
        features,
        output_dim,
        scope='fc2',
        activation_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    return features[:, :self.get_num_filter_parameters()], \
           features[:, self.get_num_filter_parameters():]

  # Should be implemented in child classes
  def filter_param_regressor(self, features):
    assert False

  # Process the whole image, without masking
  # Should be implemented in child classes
  def process(self, img, param):
    assert False

  def debug_info_batched(self):
    return False

  def no_high_res(self):
    return False

  # Apply the whole filter with masking
  def apply(self,
            img,
            img_features=None,
            specified_parameter=None,
            high_res=None):
    assert (img_features is None) ^ (specified_parameter is None)
    if img_features is not None:
      filter_features, mask_parameters = self.extract_parameters(img_features)
      filter_parameters = self.filter_param_regressor(filter_features)
    else:
      assert not self.use_masking()
      filter_parameters = specified_parameter
      mask_parameters = tf.zeros(
          shape=(1, self.get_num_mask_parameters()), dtype=np.float32)
    if high_res is not None:
      # working on high res...
      pass
    debug_info = {}
    # We only debug the first image of this batch
    if self.debug_info_batched():
      debug_info['filter_parameters'] = filter_parameters
    else:
      debug_info['filter_parameters'] = filter_parameters[0]
    self.mask_parameters = mask_parameters
    self.mask = self.get_mask(img, mask_parameters)
    debug_info['mask'] = self.mask[0]
    low_res_output = lerp(img, self.process(img, filter_parameters), self.mask)
    if high_res is not None:
      if self.no_high_res():
        high_res_output = high_res
      else:
        self.high_res_mask = self.get_mask(high_res, mask_parameters)
        high_res_output = lerp(high_res,
                               self.process(high_res, filter_parameters),
                               self.high_res_mask)
    else:
      high_res_output = None
    return low_res_output, high_res_output, debug_info

  def use_masking(self):
    return self.cfg.masking

  def get_num_mask_parameters(self):
    return 6

  # Input: no need for tanh or sigmoid
  # Closer to 1 values are applied by filter more strongly
  # no additional TF variables inside
  def get_mask(self, img, mask_parameters):
    if not self.use_masking():
      print('* Masking Disabled')
      return tf.ones(shape=(1, 1, 1, 1), dtype=tf.float32)
    else:
      print('* Masking Enabled')
    with tf.name_scope(name='mask'):
      # Six parameters for one filter
      filter_input_range = 5
      assert mask_parameters.shape[1] == self.get_num_mask_parameters()
      mask_parameters = tanh_range(
          l=-filter_input_range, r=filter_input_range,
          initial=0)(mask_parameters)
      size = list(map(int, img.shape[1:3]))
      grid = np.zeros(shape=[1] + size + [2], dtype=np.float32)

      shorter_edge = min(size[0], size[1])
      for i in range(size[0]):
        for j in range(size[1]):
          grid[0, i, j,
               0] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
          grid[0, i, j,
               1] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
      grid = tf.constant(grid)
      # Ax + By + C * L + D
      inp = grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None] + \
            grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None] + \
            mask_parameters[:, None, None, 2, None] * (rgb2lum(img) - 0.5) + \
            mask_parameters[:, None, None, 3, None] * 2
      # Sharpness and inversion
      inp *= self.cfg.maximum_sharpness * mask_parameters[:, None, None, 4,
                                                          None] / filter_input_range
      mask = tf.sigmoid(inp)
      # Strength
      mask = mask * (
          mask_parameters[:, None, None, 5, None] / filter_input_range * 0.5 +
          0.5) * (1 - self.cfg.minimum_strength) + self.cfg.minimum_strength
      print('mask', mask.shape)
    return mask

  def visualize_filter(self, debug_info, canvas):
    # Visualize only the filter information
    assert False

  def visualize_mask(self, debug_info, res):
    return cv2.resize(
        debug_info['mask'] * np.ones((1, 1, 3), dtype=np.float32),
        dsize=res,
        interpolation=cv2.cv2.INTER_NEAREST)

  def draw_high_res_text(self, text, canvas):
    cv2.putText(
        canvas,
        text, (30, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 0),
        thickness=5)
    return canvas


class ExposureFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'E'
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return tanh_range(
        -self.cfg.exposure_range, self.cfg.exposure_range, initial=0)(features)

  def process(self, img, param):
    return img * tf.exp(param[:, None, None, :] * np.log(2))

  def visualize_filter(self, debug_info, canvas):
    exposure = debug_info['filter_parameters'][0]
    if canvas.shape[0] == 64:
      cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
      cv2.putText(canvas, 'EV %+.2f' % exposure, (8, 48),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
    else:
      self.draw_high_res_text('Exposure %+.2f' % exposure, canvas)


class GammaFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'G'
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    log_gamma_range = np.log(self.cfg.gamma_range)
    return tf.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

  def process(self, img, param):
    return tf.pow(tf.maximum(img, 0.001), param[:, None, None, :])

  def visualize_filter(self, debug_info, canvas):
    gamma = debug_info['filter_parameters']
    cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
    cv2.putText(canvas, 'G 1/%.2f' % (1.0 / gamma), (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))


class ImprovedWhiteBalanceFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'W'
    self.channels = 3
    self.num_filter_parameters = self.channels

  def filter_param_regressor(self, features):
    log_wb_range = 0.5
    mask = np.array(((0, 1, 1)), dtype=np.float32).reshape(1, 3)
    print(mask.shape)
    assert mask.shape == (1, 3)
    features = features * mask
    color_scaling = tf.exp(tanh_range(-log_wb_range, log_wb_range)(features))
    # There will be no division by zero here unless the WB range lower bound is 0
    # normalize by luminance
    color_scaling *= 1.0 / (
        1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
        0.06 * color_scaling[:, 2])[:, None]
    return color_scaling

  def process(self, img, param):
    return img * param[:, None, None, :]

  def visualize_filter(self, debug_info, canvas):
    scaling = debug_info['filter_parameters']
    s = canvas.shape[0]
    cv2.rectangle(canvas, (int(s * 0.2), int(s * 0.4)), (int(s * 0.8), int(
        s * 0.6)), list(map(float, scaling)), cv2.FILLED)


class ColorFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.curve_steps = cfg.curve_steps
    self.channels = int(net.shape[3])
    self.short_name = 'C'
    self.num_filter_parameters = self.channels * cfg.curve_steps

  def filter_param_regressor(self, features):
    color_curve = tf.reshape(
        features, shape=(-1, self.channels,
                         self.cfg.curve_steps))[:, None, None, :]
    color_curve = tanh_range(
        *self.cfg.color_curve_range, initial=1)(color_curve)
    return color_curve

  def process(self, img, param):
    color_curve = param
    # There will be no division by zero here unless the color filter range lower bound is 0
    color_curve_sum = tf.reduce_sum(param, axis=4) + 1e-30
    total_image = img * 0
    for i in range(self.cfg.curve_steps):
      total_image += tf.clip_by_value(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) * \
                     color_curve[:, :, :, :, i]
    total_image *= self.cfg.curve_steps / color_curve_sum
    return total_image

  def visualize_filter(self, debug_info, canvas):
    curve = debug_info['filter_parameters']
    height, width = canvas.shape[:2]
    for i in range(self.channels):
      values = np.array([0] + list(curve[0][0][i]))
      values /= sum(values) + 1e-30
      scale = 1
      values *= scale
      for j in range(0, self.cfg.curve_steps):
        values[j + 1] += values[j]
      for j in range(self.cfg.curve_steps):
        p1 = tuple(
            map(int, (width / self.cfg.curve_steps * j, height - 1 -
                      values[j] * height)))
        p2 = tuple(
            map(int, (width / self.cfg.curve_steps * (j + 1), height - 1 -
                      values[j + 1] * height)))
        color = []
        for t in range(self.channels):
          color.append(1 if t == i else 0)
        cv2.line(canvas, p1, p2, tuple(color), thickness=1)


class ToneFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.curve_steps = cfg.curve_steps
    self.short_name = 'T'
    self.num_filter_parameters = cfg.curve_steps

  def filter_param_regressor(self, features):
    tone_curve = tf.reshape(
        features, shape=(-1, 1, self.cfg.curve_steps))[:, None, None, :]
    tone_curve = tanh_range(*self.cfg.tone_curve_range)(tone_curve)
    return tone_curve

  def process(self, img, param):
    # img = tf.minimum(img, 1.0)
    tone_curve = param
    tone_curve_sum = tf.reduce_sum(tone_curve, axis=4) + 1e-30
    total_image = img * 0
    for i in range(self.cfg.curve_steps):
      total_image += tf.clip_by_value(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) \
                     * param[:, :, :, :, i]
    total_image *= self.cfg.curve_steps / tone_curve_sum
    img = total_image
    return img

  def visualize_filter(self, debug_info, canvas):
    curve = debug_info['filter_parameters']
    height, width = canvas.shape[:2]
    values = np.array([0] + list(curve[0][0][0]))
    values /= sum(values) + 1e-30
    for j in range(0, self.curve_steps):
      values[j + 1] += values[j]
    for j in range(self.curve_steps):
      p1 = tuple(
          map(int, (width / self.curve_steps * j, height - 1 -
                    values[j] * height)))
      p2 = tuple(
          map(int, (width / self.curve_steps * (j + 1), height - 1 -
                    values[j + 1] * height)))
      cv2.line(canvas, p1, p2, (0, 0, 0), thickness=1)


class VignetFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'V'
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return tf.sigmoid(features)

  def process(self, img, param):
    return img * 0  # + param[:, None, None, :]

  def get_num_mask_parameters(self):
    return 5

  # Input: no need for tanh or sigmoid
  # Closer to 1 values are applied by filter more strongly
  # no additional TF variables inside
  def get_mask(self, img, mask_parameters):
    with tf.name_scope(name='mask'):
      # Five parameters for one filter
      filter_input_range = 5
      assert mask_parameters.shape[1] == self.get_num_mask_parameters()
      mask_parameters = tanh_range(
          l=-filter_input_range, r=filter_input_range,
          initial=0)(mask_parameters)
      size = list(map(int, img.shape[1:3]))
      grid = np.zeros(shape=[1] + size + [2], dtype=np.float32)

      shorter_edge = min(size[0], size[1])
      for i in range(size[0]):
        for j in range(size[1]):
          grid[0, i, j,
               0] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
          grid[0, i, j,
               1] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
      grid = tf.constant(grid)
      # (Ax)^2 + (By)^2 + C
      inp = (grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None]) ** 2 + \
            (grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None]) ** 2 + \
            mask_parameters[:, None, None, 2, None] - filter_input_range
      # Sharpness and inversion
      inp *= self.cfg.maximum_sharpness * mask_parameters[:, None, None, 3,
                                                          None] / filter_input_range
      mask = tf.sigmoid(inp)
      # Strength
      mask *= mask_parameters[:, None, None, 4,
                              None] / filter_input_range * 0.5 + 0.5
      if not self.use_masking():
        print('* Masking Disabled')
        mask = mask * 0 + 1
      else:
        print('* Masking Enabled')
      print('mask', mask.shape)
    return mask

  def visualize_filter(self, debug_info, canvas):
    brightness = float(debug_info['filter_parameters'][0])
    cv2.rectangle(canvas, (8, 40), (56, 52), (brightness, brightness,
                                              brightness), cv2.FILLED)


class ContrastFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'Ct'
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    # return tf.sigmoid(features)
    return tf.tanh(features)

  def process(self, img, param):
    luminance = tf.minimum(tf.maximum(rgb2lum(img), 0.0), 1.0)
    contrast_lum = -tf.cos(math.pi * luminance) * 0.5 + 0.5
    contrast_image = img / (luminance + 1e-6) * contrast_lum
    return lerp(img, contrast_image, param[:, :, None, None])

  def visualize_filter(self, debug_info, canvas):
    exposure = debug_info['filter_parameters'][0]
    cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
    cv2.putText(canvas, 'Ct %+.2f' % exposure, (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))


class WNBFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'BW'
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return tf.sigmoid(features)

  def process(self, img, param):
    luminance = rgb2lum(img)
    return lerp(img, luminance, param[:, :, None, None])

  def visualize_filter(self, debug_info, canvas):
    exposure = debug_info['filter_parameters'][0]
    cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
    cv2.putText(canvas, 'B&W%+.2f' % exposure, (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))


class LevelFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'Le'
    self.num_filter_parameters = 2

  def filter_param_regressor(self, features):
    return tf.sigmoid(features)

  def process(self, img, param):
    lower = param[:, 0]
    upper = param[:, 1] + 1
    lower = lower[:, None, None, None]
    upper = upper[:, None, None, None]
    return tf.clip_by_value((img - lower) / (upper - lower + 1e-6), 0.0, 1.0)

  def visualize_filter(self, debug_info, canvas):
    level = list(map(float, debug_info['filter_parameters']))
    level[1] += 1
    cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
    cv2.putText(canvas, '%.2f %.2f' % tuple(level), (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0))


class SaturationPlusFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'S+'
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return tf.sigmoid(features)

  def process(self, img, param):
    img = tf.minimum(img, 1.0)
    hsv = tf.image.rgb_to_hsv(img)
    s = hsv[:, :, :, 1:2]
    v = hsv[:, :, :, 2:3]
    # enhanced_s = s + (1 - s) * 0.7 * (0.5 - tf.abs(0.5 - v)) ** 2
    enhanced_s = s + (1 - s) * (0.5 - tf.abs(0.5 - v)) * 0.8
    hsv1 = tf.concat([hsv[:, :, :, 0:1], enhanced_s, hsv[:, :, :, 2:]], axis=3)
    full_color = tf.image.hsv_to_rgb(hsv1)

    param = param[:, :, None, None]
    color_param = param
    img_param = 1.0 - param

    return img * img_param + full_color * color_param

  def visualize_filter(self, debug_info, canvas):
    exposure = debug_info['filter_parameters'][0]
    if canvas.shape[0] == 64:
      cv2.rectangle(canvas, (8, 40), (56, 52), (1, 1, 1), cv2.FILLED)
      cv2.putText(canvas, 'S %+.2f' % exposure, (8, 48),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
    else:
      self.draw_high_res_text('Saturation %+.2f' % exposure, canvas)
