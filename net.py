from util import STATE_DROPOUT_BEGIN, STATE_REWARD_DIM, STATE_STEP_DIM, STATE_STOPPED_DIM
import pickle as pickle
import os
import shutil
import time
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly
from replay_memory import ReplayMemory
from util import make_image_grid, Tee, merge_dict, Dict

device = '/gpu:0'

# A small part of this script is based on https://github.com/Zardinality/WGAN-tensorflow


class GAN:

  def __init__(self, cfg, restore=False):
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    self.sess = tf.Session(config=sess_config)
    self.cfg = cfg
    assert cfg.gan == 'ls' or cfg.gan == 'w'
    self.dir = os.path.join('models', cfg.name)
    self.image_dir = os.path.join(self.dir,
                                  'images-' + cfg.name.replace('/', '-'))
    self.dump_dir = os.path.join(self.dir, 'dump-' + cfg.name.replace('/', '-'))
    if not os.path.exists(self.dir):
      os.makedirs(self.dir)
    if not os.path.exists(self.dump_dir):
      os.makedirs(self.dump_dir)
    if not os.path.exists(self.image_dir):
      os.makedirs(self.image_dir)

    if not restore:
      self.backup_scripts()
      self.tee = Tee(os.path.join(self.dir, 'log.txt'))

    self.is_train = tf.placeholder(tf.int32, shape=[], name='is_train')
    self.is_training = tf.equal(self.is_train, 1)
    self.memory = ReplayMemory(cfg, load=not restore)

    self.z = self.memory.z
    self.real_data = self.memory.real_data
    self.real_data_feature = self.memory.real_data_feature
    self.fake_input = self.memory.fake_input
    self.fake_input_feature = self.memory.fake_input_feature
    self.states = self.memory.states
    self.ground_truth = self.memory.ground_truth
    self.progress = self.memory.progress

    self.surrogate_loss_addition = 0
    with tf.variable_scope('generator'):
      fake_output, self.generator_debug_output, self.generator_debugger = cfg.generator(
          [self.fake_input, self.z, self.states],
          is_train=self.is_train,
          progress=self.progress,
          cfg=cfg)
      self.fake_output, self.new_states, self.surrogate_loss_addition, self.penalty = fake_output
      self.fake_output_feature = self.fake_input_feature
      self.memory.fake_output_feature = self.fake_output_feature
      self.memory.fake_output = self.fake_output

    print(cfg.critic)
    self.real_logit, self.real_embeddings, self.test_real_gradients = cfg.critic(
        images=self.real_data, cfg=cfg, is_train=self.is_training)
    self.fake_logit, self.fake_embeddings, self.test_fake_gradients = cfg.critic(
        images=self.fake_output, cfg=cfg, reuse=True, is_train=self.is_training)
    self.fake_input_logit, self.fake_input_embeddings, _ = cfg.critic(
        images=self.fake_input, cfg=cfg, reuse=True, is_train=self.is_training)
    print('real_logit', self.real_logit.shape)

    with tf.variable_scope('rl_value'):
      print('self.states', self.states.shape)
      print('self.new_states', self.new_states.shape)
      self.old_value, _, _ = cfg.value(
          images=self.fake_input,
          states=self.states,
          cfg=cfg,
          reuse=False,
          is_train=self.is_training)
      self.new_value, _, _ = cfg.value(
          images=self.fake_output,
          states=self.new_states,
          cfg=cfg,
          reuse=True,
          is_train=self.is_training)

    stopped = self.new_states[:, STATE_STOPPED_DIM:STATE_STOPPED_DIM + 1]
    clear_final = tf.cast(self.new_states[:, STATE_STEP_DIM:STATE_STEP_DIM + 1]
                          > self.cfg.maximum_trajectory_length, tf.float32)
    print('clear final', clear_final.shape)
    print('new_value', self.new_value.shape)
    self.new_value = self.new_value * (1.0 - clear_final)
    # Reward: the bigger, the better

    if cfg.supervised:
      self.raw_reward = (cfg.all_reward +
                         (1 - cfg.all_reward) * stopped) * (-self.fake_logit)
    else:
      if cfg.gan == 'ls':
        self.raw_reward = (cfg.all_reward + (1 - cfg.all_reward) * stopped) * (
            1 - (self.fake_logit - 1)**2)
      else:
        self.raw_reward = (cfg.all_reward + (1 - cfg.all_reward) * stopped) * (
            self.fake_logit - tf.stop_gradient(self.fake_input_logit)
        ) * cfg.critic_logit_multiplier
    self.reward = self.raw_reward
    if cfg.use_penalty:
      self.reward -= self.penalty
    print('new_states_slice', self.new_states)
    print('new_states_slice',
          self.new_states[:, STATE_REWARD_DIM:STATE_REWARD_DIM + 1])
    print('fake_logit', self.fake_logit.shape)

    self.exp_moving_average = tf.train.ExponentialMovingAverage(
        decay=0.99, zero_debias=True)

    # TD learning
    print('reward', self.reward.shape)
    # If it stops, future return should be zero
    self.q_value = self.reward + (
        1.0 - stopped) * cfg.discount_factor * self.new_value
    print('q', self.q_value.shape)
    self.advantage = tf.stop_gradient(self.q_value) - self.old_value
    self.v_loss = tf.reduce_mean(self.advantage**2, axis=(0, 1))

    if cfg.gan == 'ls':
      print('** LSGAN')
      self.c_loss = tf.reduce_mean(self.fake_logit**2) + tf.reduce_mean(
          (self.real_logit - 1)**2)
      if cfg.use_TD:
        routine_loss = -self.q_value * self.cfg.parameter_lr_mul
        advantage = -self.advantage
      else:
        routine_loss = -self.reward
        advantage = -self.reward
      print('routine_loss', routine_loss.shape)
      print('pg_loss', self.surrogate_loss_addition.shape)
      assert len(routine_loss.shape) == len(self.surrogate_loss_addition.shape)

      self.g_loss = tf.reduce_mean(routine_loss + self.surrogate_loss_addition *
                                   tf.stop_gradient(advantage))
      self.emd = self.c_loss
      self.c_average = tf.constant(0, dtype=tf.float32)
    else:
      print('** WGAN')
      self.c_loss = tf.reduce_mean(self.fake_logit - self.real_logit)
      if cfg.use_TD:
        routine_loss = -self.q_value * self.cfg.parameter_lr_mul
        advantage = -self.advantage
      else:
        routine_loss = -self.reward
        advantage = -self.reward
      print('routine_loss', routine_loss.shape)
      print('pg_loss', self.surrogate_loss_addition.shape)
      assert len(routine_loss.shape) == len(self.surrogate_loss_addition.shape)

      self.g_loss = tf.reduce_mean(routine_loss + self.surrogate_loss_addition *
                                   tf.stop_gradient(advantage))
      self.emd = -self.c_loss
      self.c_average = tf.reduce_mean(self.fake_logit + self.real_logit) * 0.5
    update_average = self.exp_moving_average.apply([self.c_average])
    self.c_average_smoothed = self.exp_moving_average.average(self.c_average)
    self.centered_fake_logit = self.fake_logit - self.c_average_smoothed
    self.fake_gradients = tf.gradients(self.fake_logit, [
        self.fake_output,
    ])[0]

    # Critic gradient norm and penalty
    alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
    alpha = alpha_dist.sample((cfg.batch_size, 1, 1, 1))
    interpolated = self.real_data + alpha * (self.fake_output - self.real_data)

    inte_logit, inte_embeddings, _ = cfg.critic(
        images=interpolated, cfg=cfg, reuse=True, is_train=self.is_training)

    gradients = tf.gradients(inte_logit, [
        interpolated,
    ])[0]

    gradient_norm = tf.sqrt(1e-6 + tf.reduce_sum(gradients**2, axis=[1, 2, 3]))
    gradient_penalty = cfg.gradient_penalty_lambda * tf.reduce_mean(
        tf.maximum(gradient_norm - 1.0, 0.0)**2)
    _ = tf.summary.scalar("grad_penalty_loss", gradient_penalty)
    self.critic_gradient_norm = tf.reduce_mean(gradient_norm)
    _ = tf.summary.scalar("grad_norm", self.critic_gradient_norm)
    if cfg.gan == 'w':
      if cfg.gradient_penalty_lambda > 0:
        print('** Using gradient penalty')
        self.c_loss += gradient_penalty
    else:
      gradient_norm = tf.sqrt(
          tf.reduce_sum(self.fake_gradients**2, axis=[1, 2, 3]))
      self.critic_gradient_norm = tf.reduce_mean(gradient_norm)
      print('** NOT using gradient penalty')

    _ = tf.summary.scalar("g_loss", self.g_loss)
    _ = tf.summary.scalar("neg_c_loss", -self.c_loss)
    _ = tf.summary.scalar("EMD", self.emd)

    self.theta_g = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    self.theta_c = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
    self.theta_v = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='rl_value')
    print('# variables')
    print('    generator:', len(self.theta_g))
    print('    value:', len(self.theta_v))
    print('    critic:', len(self.theta_c))

    self.lr_g = tf.placeholder(dtype=tf.float32, shape=[], name='lr_g')
    self.lr_c = tf.placeholder(dtype=tf.float32, shape=[], name='lr_c')

    # Optimizer for Value estimator, use the same lr as g
    self.counter_v = tf.Variable(
        trainable=False, initial_value=0, dtype=tf.int32)
    self.opt_v = ly.optimize_loss(
        loss=self.v_loss,
        learning_rate=self.cfg.value_lr_mul * self.lr_g,
        optimizer=cfg.generator_optimizer,
        variables=self.theta_v,
        global_step=self.counter_v,
        summaries=['gradient_norm'])

    # Optimize for Generator (Actor)
    self.counter_g = tf.Variable(
        trainable=False, initial_value=0, dtype=tf.int32)
    self.opt_g = ly.optimize_loss(
        loss=self.g_loss,
        learning_rate=self.lr_g,
        optimizer=cfg.generator_optimizer,
        variables=self.theta_g,
        global_step=self.counter_g,
        summaries=['gradient_norm'])

    # Optimize for Discriminator (critic in WGAN or discriminator in LSGAN)
    self.counter_c = tf.Variable(
        trainable=False, initial_value=0, dtype=tf.int32)
    if not self.cfg.supervised:
      self.opt_c = ly.optimize_loss(
          loss=self.c_loss,
          learning_rate=self.lr_c,
          optimizer=cfg.critic_optimizer,
          variables=self.theta_c,
          global_step=self.counter_c,
          summaries=['gradient_norm'])

      if cfg.gan == 'w' and cfg.gradient_penalty_lambda <= 0:
        print(
            '** make sure your NN input has mean 0, as biases will also be clamped.'
        )
        # Merge the clip operations on critic variables
        # For WGAN
        clipped_var_c = [
            tf.assign(var,
                      tf.clip_by_value(var, -self.cfg.clamp_critic,
                                       self.cfg.clamp_critic))
            for var in self.theta_c
        ]
        with tf.control_dependencies([self.opt_c]):
          self.opt_c = tf.tuple(clipped_var_c)

      with tf.control_dependencies([self.opt_c]):
        self.opt_c = tf.group(update_average)

    self.saver = tf.train.Saver(
        max_to_keep=1)  # save all checkpoints  max_to_keep=None

    self.sess.run(tf.global_variables_initializer())

    self.merged_all = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(self.dir, self.sess.graph)

    if not restore:
      self.fixed_feed_dict_random = self.memory.get_feed_dict(
          self.cfg.num_samples)
    self.high_res_nets = {}

  def get_training_feed_dict_and_states(self, iter):
    feed_dict, features = self.memory.get_feed_dict_and_states(
        self.cfg.batch_size)
    feed_dict[self.lr_g] = self.cfg.lr_g(iter)
    feed_dict[self.lr_c] = self.cfg.lr_c(iter)
    feed_dict[self.is_train] = 1
    return feed_dict, features

  def get_replay_feed_dict(self, iter):
    feed_dict = self.memory.get_replay_feed_dict(self.cfg.batch_size)
    feed_dict[self.lr_c] = self.cfg.lr_c(iter)
    feed_dict[self.is_train] = 1
    return feed_dict

  def train(self):
    start_t = time.time()

    g_loss_pool = []
    v_loss_pool = []
    emd_pool = []
    # critic gradient (critic logit w.r.t. critic input image) norm
    cgn = 0

    for iter in range(self.cfg.max_iter_step + 1):
      progress = float(iter) / self.cfg.max_iter_step
      iter_start_time = time.time()
      run_options = tf.RunOptions()
      run_metadata = tf.RunMetadata()
      if self.cfg.gan == 'w' and (iter < self.cfg.critic_initialization or
                                  iter % 500 == 0):
        citers = 100
      else:
        citers = self.cfg.citers

      if iter == 0:
        # Make sure there are terminating states
        giters = 100
      else:
        giters = self.cfg.giters

      # Update generator actor/critic
      for j in range(giters):
        feed_dict, features = self.get_training_feed_dict_and_states(iter)
        if iter == 0:
          feed_dict[self.lr_g] = 0
        feed_dict[self.progress] = progress
        _, g_loss, v_loss, fake_output, new_states = self.sess.run(
            [(self.opt_g, self.opt_v), self.g_loss, self.v_loss,
             self.fake_output, self.new_states],
            feed_dict=feed_dict,
            options=run_options,
            run_metadata=run_metadata)
        if self.cfg.supervised:
          ground_truth = feed_dict[self.ground_truth]
        else:
          ground_truth = None
        self.memory.replace_memory(
            self.memory.images_and_states_to_records(
                fake_output, new_states, features, ground_truth=ground_truth))
        v_loss_pool.append(v_loss)
        g_loss_pool.append(g_loss)

        if iter % self.cfg.summary_freq == 0 and j == 0:
          merged = self.sess.run(
              self.merged_all,
              feed_dict=feed_dict,
              options=run_options,
              run_metadata=run_metadata)
          self.summary_writer.add_summary(merged, iter)
          self.summary_writer.add_run_metadata(
              run_metadata, 'critic_metadata {}'.format(iter), iter)

      merged = []
      # Update GAN discriminator ('critic' for WGAN)
      for j in range(citers):
        feed_dict = self.get_replay_feed_dict(iter)
        if not self.cfg.supervised:
          # update discriminator only if it is unsupervised
          _, emd, cgn = self.sess.run(
              [self.opt_c, self.emd, self.critic_gradient_norm],
              feed_dict=feed_dict)
          emd_pool.append(emd)

      if merged:
        self.summary_writer.add_summary(merged, iter)
        self.summary_writer.add_run_metadata(
            run_metadata, 'generator_metadata {}'.format(iter), iter)

      # Visualizations
      if self.cfg.realtime_vis or iter % self.cfg.write_image_interval == 0:
        self.visualize(iter)

      v_loss_pool = v_loss_pool[-self.cfg.median_filter_size:]
      g_loss_pool = g_loss_pool[-self.cfg.median_filter_size:]
      emd_pool = emd_pool[-self.cfg.median_filter_size:]

      if (iter + 1) % 500 == 0:
        self.saver.save(
            self.sess,
            os.path.join(self.dir, "model.ckpt"),
            global_step=(iter + 1))

      if iter % 100 == 0:
        eta = (time.time() - start_t) / (iter + 1) / 3600 * (
            self.cfg.max_iter_step - iter)
        tot_time = (time.time() - start_t) / (iter + 1) / 3600 * (
            self.cfg.max_iter_step)
        if iter < 500:
          eta = tot_time = 0
        print('#--------------------------------------------')
        print('# Task: %s  ela. %.2f min  ETA: %.1f/%.1f h' %
              (self.cfg.name, (time.time() - start_t) / 60.0, eta, tot_time))
        self.memory.debug()

      if iter % 10 == 0:
        print(
            'it%6d,%5.0f ms/it, g_loss=%.2f, v_loss=%.2f, EMD=%.3f, cgn=%.2f' %
            (iter, 1000 * (time.time() - iter_start_time),
             np.median(g_loss_pool), np.median(v_loss_pool),
             np.median(emd_pool), cgn))

  def restore(self, ckpt):
    self.saver.restore(self.sess, os.path.join(self.dir,
                                               "model.ckpt-%s" % ckpt))

  def gradient_processor(self, grads):
    if self.cfg.gan == 'ls':
      # We show negative grad. (since we are mininizing)
      real_grads = []
      for g in grads:
        if (abs(np.mean(g) - 1)) > 0.001:
          real_grads.append(g)
      return -grads / np.std(real_grads) * 0.2 + 0.5
    else:
      return 10 * grads + 0.5

  def visualize(self, iter):
    progress = float(iter) / self.cfg.max_iter_step
    lower_regions = []
    pool_images, pool_states, pool_features = self.memory.records_to_images_states_features(
        self.memory.image_pool[:self.cfg.num_samples])

    if self.cfg.supervised:
      gt0 = [x[1] for x in pool_images]
      pool_images = [x[0] for x in pool_images]
    else:
      gt0 = None
    lower_regions.append(pool_images)

    # Generated data
    feed_dict = merge_dict(self.fixed_feed_dict_random, {
        self.is_train: self.cfg.test_random_walk,
        self.progress: progress
    })
    eval_images = []
    eval_states = []
    gt1 = self.fixed_feed_dict_random[self.ground_truth]
    for i in range(self.cfg.test_steps):
      output_images, output_states = self.sess.run(
          [self.fake_output, self.new_states], feed_dict=feed_dict)
      feed_dict[self.fake_input] = output_images
      feed_dict[self.states] = output_states

      eval_images.append(output_images)
      eval_states.append(output_states)

    best_outputs = []
    best_indices = []
    for i in range(self.cfg.num_samples):
      best_index = self.cfg.test_steps - 1
      for j in range(self.cfg.test_steps):
        if eval_states[j][i][STATE_REWARD_DIM] > 0:
          best_index = j
          break
      best_image = eval_images[best_index][i]
      best_indices.append(best_index + 1)
      best_outputs.append(best_image)

    lower_regions.append(best_outputs)
    # Real data
    lower_regions.append(self.fixed_feed_dict_random[self.real_data])

    if self.cfg.vis_draw_critic_scores:
      lower_regions[0] = self.draw_critic_scores(
          lower_regions[0], ground_truth=gt0)
      lower_regions[1] = self.draw_critic_scores(
          lower_regions[1], ground_truth=gt1)
      if not self.cfg.supervised:
        lower_regions[2] = self.draw_critic_scores(lower_regions[2])

    for img, state in zip(lower_regions[0], pool_states):
      cv2.putText(img,
                  str(state), (4, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                  (1.0, 0.0, 0.0))

    for img, ind in zip(lower_regions[1], best_indices):
      cv2.putText(img,
                  str(ind), (23, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0, 0.0,
                                                                      0.0))

    lower_regions = list(map(make_image_grid, lower_regions))
    seperator = np.ones(
        (lower_regions[0].shape[0], 16, lower_regions[0].shape[2]),
        dtype=np.float32)
    lower_region = np.hstack([
        lower_regions[0], seperator, lower_regions[1], seperator,
        lower_regions[2]
    ])

    upper_region = np.ones_like(lower_region)

    per_row = lower_region.shape[1] // (self.generator_debugger.width + 4)

    # The upper part
    h, w = self.cfg.source_img_size, self.cfg.source_img_size
    images = []
    debug_plots = []
    gradients = []
    rows = lower_region.shape[0] // (h + 2) // 3
    groups_per_row = per_row // (self.cfg.test_steps + 1)
    per_row = (self.cfg.test_steps + 1) * groups_per_row

    gts = []
    for j in range(min(self.cfg.num_samples, rows * groups_per_row)):
      if self.cfg.supervised:
        img_gt = self.memory.get_next_RAW(1, test=self.cfg.vis_step_test)[0][0]
        img, gt = img_gt[0], img_gt[1]
      else:
        img = self.memory.get_next_RAW(1)[0][0]
        gt = None
      # z is useless at test time...
      images_, debug_plots_, gradients_ = self.draw_steps(
          img,
          ground_truth=gt,
          is_train=self.cfg.test_random_walk,
          progress=progress)
      images += images_
      if self.cfg.supervised:
        gts += [gt] * len(images_)
        gradients_ = [gt] * len(images_)
      debug_plots += debug_plots_
      gradients += gradients_

    if not self.cfg.supervised:
      gradients = self.gradient_processor(np.stack(gradients, axis=0))

    pad = 0
    for i in range(rows):
      for j in range(per_row):
        start_x, start_y = pad + 3 * i * (h + 2), pad + j * (w + 4)
        index = i * per_row + j
        if index < len(images):
          upper_region[start_x:start_x + h, start_y:start_y + w] = images[index]
          upper_region[start_x + h + 1:start_x + h * 2 + 1, start_y:
                       start_y + w] = gradients[index]
          upper_region[start_x + 2 * (h + 1):start_x + h * 3 + 2, start_y:
                       start_y + w] = debug_plots[index]

    seperator = np.ones(
        (16, upper_region.shape[1], upper_region.shape[2]), dtype=np.float32)
    upper_region = np.vstack([seperator, upper_region, seperator])

    img = np.vstack([upper_region, lower_region])
    if self.cfg.realtime_vis:
      cv2.imshow('vis', img[:, :, ::-1])
      cv2.waitKey(20)
    if iter % self.cfg.write_image_interval == 0:
      fn = os.path.join(self.image_dir, '%06d.png' % iter)
      cv2.imwrite(fn, img[:, :, ::-1] * 255.0)

  def draw_value_reward_score(self, img, value, reward, score):
    img = img.copy()
    # Average with 0.5 for semi-transparent background
    img[:14] = img[:14] * 0.5 + 0.25
    img[50:] = img[50:] * 0.5 + 0.25
    if self.cfg.gan == 'ls':
      red = -np.tanh(float(score) / 1) * 0.5 + 0.5
    else:
      red = -np.tanh(float(score) / 10.0) * 0.5 + 0.5
    top = '%+.2f %+.2f' % (value, reward)
    cv2.putText(img, top, (3, 7), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                (1.0, 1.0 - red, 1.0 - red))
    score = '%+.3f' % score
    cv2.putText(img, score, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (1.0, 1.0 - red, 1.0 - red))
    return img

  def draw_steps(self, img, progress, ground_truth=None, is_train=0):
    images = []
    debug_plots = []
    gradients = []
    states = self.memory.get_initial_states(self.cfg.batch_size)

    tmp_fake_output = [img] * self.cfg.batch_size
    tmp_fake_output = np.stack(tmp_fake_output, axis=0)
    initial_value, initial_score = self.sess.run(
        [self.new_value[0], self.centered_fake_logit[0]],
        feed_dict={
            self.fake_output: tmp_fake_output,
            self.new_states: states,
            self.progress: progress
        })

    images.append(
        self.draw_value_reward_score(img, initial_value, 0, initial_score))
    debug_plots.append(img * 0 + 1)
    # z is useless at test time...
    gradients.append(img * 0 + 1)
    for k in range(self.cfg.test_steps):
      feed_dict = {
          self.fake_input: [img] * self.cfg.batch_size,
          self.real_data: [img] * self.cfg.batch_size,
          self.z: self.memory.get_noise(self.cfg.batch_size),
          self.is_train: is_train,
          self.states: states,
          self.progress: progress
      }
      if self.cfg.supervised:
        feed_dict[self.ground_truth] = [ground_truth]
        feed_dict[self.progress] = progress
      debug_info, img, grad, new_state, new_value, score, reward = self.sess.run(
          [
              self.generator_debug_output, self.fake_output[0],
              self.fake_gradients[0], self.new_states, self.new_value[0],
              self.centered_fake_logit[0], self.reward[0]
          ],
          feed_dict=feed_dict)
      debug_plot = self.generator_debugger(debug_info)
      images.append(self.draw_value_reward_score(img, new_value, reward, score))
      gradients.append(grad)
      debug_plots.append(debug_plot)
      states = new_state

      if states[0, STATE_STOPPED_DIM] > 0:
        break

    for k in range(len(images), self.cfg.test_steps + 1):
      images.append(img * 0 + 1)
      gradients.append(img * 0 + 1)
      debug_plots.append(img * 0 + 1)
    return images, debug_plots, gradients

  def draw_critic_scores(self, images, ground_truth=None):
    # We do not care about states here, so that value drawn may not make sense.
    images = list(images)
    original_len = len(images)
    if len(images) < self.cfg.batch_size:
      images += [images[0]] * (self.cfg.batch_size - len(images))
    states = self.memory.get_initial_states(self.cfg.batch_size)
    # indexs = self.memory.get_random_indexs(self.cfg,batch_size)
    images = np.stack(images, axis=0)
    if self.cfg.supervised:
      # TODO
      feed_dict = {
          self.real_data: images,
          self.fake_input: images,
          self.ground_truth: ground_truth,
          self.new_states: states,
          self.states: states,
          self.is_train: 0
      }
    else:
      feed_dict = {
          self.fake_output: images,
          self.real_data: images,
      }
    if self.cfg.gan == 'ls':
      logit = self.fake_logit
    else:
      logit = self.centered_fake_logit
    scores = self.sess.run(logit, feed_dict=feed_dict)
    if self.cfg.supervised:
      scores = np.sqrt(scores) * 100.0
    ret = []
    for i in range(len(images)):
      img, score = images[i].copy(), scores[i]
      # Average with 0.5 for semi-transparent background
      img[50:] = img[50:] * 0.5 + 0.25
      if self.cfg.gan == 'ls':
        red = -np.tanh(float(score) / 1) * 0.5 + 0.5
      else:
        red = -np.tanh(float(score) / 10.0) * 0.5 + 0.5
      score = '%+.3f' % score
      cv2.putText(img, score, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                  (1.0, 1.0 - red, 1.0 - red))
      ret.append(img)
    return ret[:original_len]

  def backup_scripts(self):
    script_dir = os.path.join(self.dir, 'scripts')
    try:
      os.mkdir(script_dir)
    except Exception as e:
      pass
    for fn in os.listdir('.'):
      if fn.endswith('.py'):
        shutil.copy(fn, script_dir)
    print('Scripts are backed up. Initializing network...')

  def get_high_resolution_net(self, res):
    if res not in self.high_res_nets:
      print('Creating high_res_network for ', res)
      net = Dict()
      net.high_res_input = tf.placeholder(
          tf.float32,
          shape=(None, res[0], res[1], self.cfg.real_img_channels),
          name='highres_in')
      net.fake_input = self.fake_input
      net.fake_input_feature = self.fake_input_feature
      net.real_data = self.real_data
      net.z = self.z
      net.is_train = self.is_train
      net.states = self.states
      with tf.variable_scope('generator', reuse=True):
        fake_output, net.generator_debug_output, net.generator_debugger = self.cfg.generator(
            [net.fake_input, net.z, net.states],
            is_train=net.is_train,
            cfg=self.cfg,
            high_res=net.high_res_input,
            progress=0)
        net.fake_output, net.new_states, net.high_res_output = fake_output

      net.fake_logit, net.fake_embeddings, _ = self.cfg.critic(
          images=net.fake_output, cfg=self.cfg, reuse=True, is_train=False)
      self.high_res_nets[res] = net
    return self.high_res_nets[res]

  def eval(self,
           spec_files=None,
           output_dir='./outputs',
           step_by_step=False,
           show_linear=True,
           show_input=True):
    from util import get_image_center
    if output_dir is not None:
      try:
        os.mkdir(output_dir)
      except:
        pass
    print(spec_files)

    # Use a fixed noise
    batch_size = 1
    for fn in spec_files:
      print('Processing input {}'.format(fn))

      from util import read_tiff16, linearize_ProPhotoRGB
      if fn.endswith('.tif') or fn.endswith('.tiff'):
        image = read_tiff16(fn)
        high_res_image = linearize_ProPhotoRGB(image)
      else:
        # TODO: deal with png and jpeg files better - they are probably not RAW.
        print(
            'Warning: sRGB color space jpg and png images may not work perfectly. See README for details. (image {})'.
            format(fn))
        image = cv2.imread(fn)[:, :, ::-1]
        if image.dtype == np.uint8:
          image = image / 255.0
        if image.dtype == np.uint16:
          image = image / 65535.0
        else:
          print('image data type {} is not supported. Please email Yuanming Hu.'.format(image.dtype))
        high_res_image = np.power(image, 2.2)  # Linearize sRGB
        high_res_image /= 2 * high_res_image.max() # Mimic RAW exposure

      noises = [
          self.memory.get_noise(batch_size) for _ in range(self.cfg.test_steps)
      ]
      fn = fn.split('/')[-1]

      def get_dir():
        if output_dir is not None:
          d = output_dir
        else:
          d = self.dump_dir
        return d

      try:
        os.mkdir(get_dir())
      except:
        pass

      def show_and_save(x, img):
        img = img[:, :, ::-1]
        #cv2.imshow(x, img)
        cv2.imwrite(os.path.join(get_dir(), fn + '.' + x + '.png'), img * 255.0)

      #if os.path.exists(os.path.join(get_dir(), fn + '.retouched.png')):
      #    print('Skipping', fn)
      #    continue

      high_res_input = high_res_image
      low_res_img = cv2.resize(get_image_center(high_res_image), dsize=(64, 64))
      res = high_res_input.shape[:2]
      net = self.get_high_resolution_net(res)

      low_res_img_trajs = [low_res_img]
      low_res_images = [low_res_img]
      states = self.memory.get_initial_states(batch_size)
      high_res_output = high_res_input
      masks = []
      decisions = []
      operations = []
      debug_info_list = []

      tmp_fake_input = low_res_images * batch_size
      tmp_fake_input = np.array(tmp_fake_input)
      print(tmp_fake_input.shape)

      for i in range(self.cfg.test_steps):
        feed_dict = {
            net.fake_input: low_res_images * batch_size,
            net.z: noises[i],
            net.is_train: 0,
            net.states: states,
            net.high_res_input: [high_res_output] * batch_size
        }
        new_low_res_images, new_scores, new_states, new_high_res_output, debug_info = self.sess.run(
            [
                net.fake_output[0], net.fake_logit[0], net.new_states[0],
                net.high_res_output[0], net.generator_debug_output
            ],
            feed_dict=feed_dict)
        low_res_img_trajs.append(new_low_res_images)
        low_res_images = [new_low_res_images]
        # print('new_states', new_states.shape)
        states = [new_states] * batch_size
        debug_info_list.append(debug_info)
        debug_plots = self.generator_debugger(debug_info, combined=False)
        decisions.append(debug_plots[0])
        operations.append(debug_plots[1])
        masks.append(debug_plots[2])
        high_res_output = new_high_res_output
        if states[0][STATE_STOPPED_DIM] > 0:
          break
        if step_by_step:
          show_and_save('intermediate%02d' % i, high_res_output)

      linear_high_res = high_res_input

      # Max to white, and then gamma correction
      high_res_input = (high_res_input / high_res_input.max())**(1 / 2.4)

      # Save linear
      if show_linear:
        show_and_save('linear', linear_high_res)

      # Save corrected
      if show_input:
        show_and_save('input_tone_mapped', high_res_input)

      # Save retouched
      show_and_save('retouched', high_res_output)

      # Steps & debugging information
      with open(os.path.join(get_dir(), fn + '_debug.pkl'), 'wb') as f:
        pickle.dump(debug_info_list, f)

      padding = 4
      patch = 64
      grid = patch + padding
      steps = len(low_res_img_trajs)

      fused = np.ones(shape=(grid * 4, grid * steps, 3), dtype=np.float32)

      for i in range(len(low_res_img_trajs)):
        sx = grid * i
        sy = 0
        fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
            low_res_img_trajs[i],
            dsize=(patch, patch),
            interpolation=cv2.cv2.INTER_NEAREST)

      for i in range(len(low_res_img_trajs) - 1):
        sx = grid * i + grid // 2
        sy = grid
        fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
            decisions[i],
            dsize=(patch, patch),
            interpolation=cv2.cv2.INTER_NEAREST)
        sy = grid * 2 - padding // 2
        fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
            operations[i],
            dsize=(patch, patch),
            interpolation=cv2.cv2.INTER_NEAREST)
        sy = grid * 3 - padding
        fused[sy:sy + patch, sx:sx + patch] = cv2.resize(
            masks[i], dsize=(patch, patch), interpolation=cv2.cv2.INTER_NEAREST)

      # Save steps
      show_and_save('steps', fused)
