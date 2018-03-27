import tensorflow as tf
import tensorflow.contrib.layers as ly
from util import lrelu
import cv2
import math
from pdf_sample_layer import pdf_sample
from util import enrich_image_input
from util import STATE_DROPOUT_BEGIN, STATE_REWARD_DIM, STATE_STEP_DIM, STATE_STOPPED_DIM


def feature_extractor(net, output_dim, cfg):
  net = net - 0.5
  min_feature_map_size = 4
  assert output_dim % (
      min_feature_map_size**2) == 0, 'output dim=%d' % output_dim
  size = int(net.get_shape()[2])
  print('Agent CNN:')
  channels = cfg.base_channels
  print('    ', str(net.get_shape()))
  size /= 2
  net = ly.conv2d(
      net, num_outputs=channels, kernel_size=4, stride=2, activation_fn=lrelu)
  print('    ', str(net.get_shape()))
  while size > min_feature_map_size:
    if size == min_feature_map_size * 2:
      channels = output_dim / (min_feature_map_size**2)
    else:
      channels *= 2
    assert size % 2 == 0
    size /= 2
    net = ly.conv2d(
        net, num_outputs=channels, kernel_size=4, stride=2, activation_fn=lrelu)
    print('    ', str(net.get_shape()))
  print('before fc: ', net.get_shape()[1])
  net = tf.reshape(net, [-1, output_dim])
  net = tf.nn.dropout(net, cfg.dropout_keep_prob)
  return net


# Output: float \in [0, 1]
def agent_generator(inp, is_train, progress, cfg, high_res=None, alex_in=None):
  net, z, states = inp
  filters = cfg.filters

  filters = [x(net, cfg) for x in filters]

  selection_noise = z[:, 0:1]
  filtered_images = []
  filter_debug_info = []
  high_res_outputs = []

  if cfg.shared_feature_extractor:
    filter_features = feature_extractor(
        net=enrich_image_input(cfg, net, states),
        output_dim=cfg.feature_extractor_dims,
        cfg=cfg)
    # filter_features = ly.dropout(filter_features)
  for j, filter in enumerate(filters):
    with tf.variable_scope('filter_%d' % j):
      print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.',
            filter.get_short_name())
      if not cfg.shared_feature_extractor:
        filter_features = \
            feature_extractor(net=enrich_image_input(cfg, net),
                              output_dim=cfg.feature_extractor_dims, cfg=cfg)
      print('      filter_features:', filter_features.shape)
      filtered_image_batch, high_res_output, per_filter_debug_info = filter.apply(
          net, filter_features, high_res=high_res)
      high_res_outputs.append(high_res_output)
      filtered_images.append(filtered_image_batch)
      filter_debug_info.append(per_filter_debug_info)
      print('      output:', filtered_image_batch.shape)

  # [batch_size, #filters, H, W, C]
  for img in filtered_images:
    print('img', img.shape)
  filtered_images = tf.stack(values=filtered_images, axis=1)
  print('    filtered_images:', filtered_images.shape)

  with tf.variable_scope('action_selection'):
    selector_features = feature_extractor(
        net=enrich_image_input(cfg, net, states),
        output_dim=cfg.feature_extractor_dims,
        cfg=cfg)
    print('    selector features:', selector_features.shape)

    selector_features = ly.fully_connected(
        selector_features,
        num_outputs=cfg.fc1_size,
        scope='selector_fc1',
        activation_fn=lrelu)

    # selector_features = ly.dropout(selector_features)

    pdf = ly.fully_connected(
        selector_features,
        num_outputs=len(filters),
        activation_fn=None,
        scope='selector_fc2')
    pdf = tf.nn.softmax(pdf) + 1e-37
    print('    pdf_filter', pdf[:, 1:].shape)
    # print('    pdf_mask', states[:, STATE_DROPOUT_BEGIN:].shape)

    pdf = pdf * (1 - cfg.exploration) + cfg.exploration * 1.0 / len(filters)
    # pdf = tf.to_float(is_train) * tf.concat([pdf[:, :1], pdf[:, 1:] * states[:, STATE_DROPOUT_BEGIN:]], axis=1) \
    # + (1.0 - tf.to_float(is_train)) * pdf
    pdf = pdf / (tf.reduce_sum(pdf, axis=1, keep_dims=True) + 1e-30)
    entropy = -pdf * tf.log(pdf)
    entropy = tf.reduce_sum(entropy, axis=1)[:, None]
    print('    pdf:', pdf.shape)
    print('    entropy:', entropy.shape)
    print('    selection_noise:', selection_noise.shape)
    random_filter_id = pdf_sample(pdf, selection_noise)
    max_filter_id = tf.cast(tf.argmax(pdf, axis=1), tf.int32)
    selected_filter_id = is_train * random_filter_id + (
        1 - is_train) * max_filter_id
    print('    selected_filter_id:', selected_filter_id.shape)
    filter_one_hot = tf.one_hot(
        selected_filter_id, depth=len(filters), dtype=tf.float32)
    print('    filter one_hot', filter_one_hot.shape)
    surrogate = tf.reduce_sum(
        filter_one_hot * tf.log(pdf + 1e-10), axis=1, keep_dims=True)

  net = tf.reduce_sum(
      filtered_images * filter_one_hot[:, :, None, None, None], axis=1)
  if high_res is not None:
    high_res_outputs = tf.stack(values=high_res_outputs, axis=1)
    high_res_output = tf.reduce_sum(
        high_res_outputs * filter_one_hot[:, :, None, None, None], axis=1)

  # only the first image will get debug_info
  debug_info = {
      'state': states,
      'selected_filter_id': selected_filter_id[0],
      'filter_debug_info': filter_debug_info,
      'pdf': pdf[0]
  }

  # Combined: Three in one 64x64 ?
  #           otherwise returns pdf, detail, mask
  def debugger(debug_info, combined=True):
    size = 8
    img = None
    images = [None for i in range(3)]
    for i, filter in enumerate(filters):
      selected = i == debug_info['selected_filter_id']
      if selected:
        img = filter.visualize_mask(debug_info['filter_debug_info'][i],
                                    (64, 64)) * 0.8
    assert img is not None
    if not combined:
      # Mask
      images[2] = img.copy()
      # reset img
      img = img * 0 + 0.5

    c = 0
    for i, filter in enumerate(filters):
      pdf = debug_info['pdf'][i]
      if pdf < 1e-10:
        continue
      else:
        c += 1
      selected = i == debug_info['selected_filter_id']
      if selected:
        filter.visualize_filter(debug_info['filter_debug_info'][i], img)
    if not combined:
      # detail
      images[1] = img.copy()
      # reset img
      img = img * 0 + 0.5
    c = 0
    for i, filter in enumerate(filters):
      per_col = 4
      x = c // per_col * 30
      y = size * (c % per_col + 1)
      pdf = debug_info['pdf'][i]
      if pdf < 1e-10:
        continue
      else:
        c += 1
      cv2.putText(img,
                  filter.get_short_name(), (x + 6, y + 4),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.233, (255, 255, 255))
      selected = i == debug_info['selected_filter_id']
      color = 1.0 if selected else 0.3
      width = int(pdf * 20)
      height = 0.35
      corners = [(x + 16, int(y + (1 - height) * size // 2)),
                 (x + 16 + width, int(y + (1 + height) * size // 2))]
      cv2.rectangle(img, (corners[0][0] - 1, corners[0][1] - 1),
                    (corners[1][0] + 1, corners[1][1] + 1), (1, 1,
                                                             1), cv2.FILLED)
      cv2.rectangle(img, corners[0], corners[1], (color, 0.3, 0.3), cv2.FILLED)
    if not combined:
      # pdf
      images[0] = img.copy()

    if combined:
      return img
    else:
      return images

  debugger.width = int(net.shape[1])
  print('    surrogate: ', surrogate.shape)

  # Calculate new states
  new_states = [None for _ in range(STATE_DROPOUT_BEGIN + 1)]
  is_last_step = tf.cast(
      tf.abs(states[:, STATE_STEP_DIM:STATE_STEP_DIM + 1] + 1 - cfg.test_steps)
      < 1e-4,
      dtype=tf.float32)
  submitted = is_last_step

  new_states[STATE_REWARD_DIM] = submitted
  new_states[STATE_STOPPED_DIM] = submitted
  # Increment the step
  new_states[STATE_STEP_DIM] = (states[:, STATE_STEP_DIM] + 1)[:, None]

  # Update filter usage
  filter_usage = states[:, STATE_STEP_DIM + 1:]
  print('usage v.s. onehot', filter_usage.shape, filter_one_hot.shape)
  assert len(filter_usage.shape) == len(filter_one_hot.shape)

  regular_filter_start = 0

  # Penalize submission action that is not the final action.
  early_stop_penalty = (1 - is_last_step) * submitted * cfg.early_stop_penalty

  usage_penalty = tf.reduce_sum(
      filter_usage * filter_one_hot[:, regular_filter_start:],
      axis=1,
      keep_dims=True)
  new_filter_usage = tf.maximum(filter_usage,
                                filter_one_hot[:, regular_filter_start:])
  new_states[STATE_STEP_DIM + 1] = new_filter_usage

  print(submitted.shape, new_states[STATE_STEP_DIM].shape)
  new_states = tf.concat(new_states, axis=1)
  print('new_states:', new_states.shape)

  if cfg.clamp:
    net = tf.clip_by_value(net, 0.0, 5.0)

  entropy_penalty = (1.0 - progress) * cfg.exploration_penalty * (
      -entropy + math.log(len(filters)))

  # Will be substracted from award
  penalty = tf.reduce_mean(
      tf.maximum(net - 1, 0)**2, axis=(1, 2, 3)
  )[:,
    None] + entropy_penalty + usage_penalty * cfg.filter_usage_penalty + early_stop_penalty

  print('states, new_states:', states.shape, new_states.shape)
  print('penalty:', penalty.shape)

  if high_res is None:
    return (net, new_states, surrogate, penalty), debug_info, debugger
  else:
    return (net, new_states, high_res_output), debug_info, debugger
