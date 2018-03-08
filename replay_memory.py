import random
import numpy as np
import tensorflow as tf
from util import Dict
from util import STATE_DROPOUT_BEGIN, STATE_REWARD_DIM, STATE_STEP_DIM, STATE_STOPPED_DIM



class ReplayMemory:
    def __init__(self, cfg, load):
        self.cfg = cfg
        self.real_dataset = cfg.real_data_provider()
        if load:
            self.fake_dataset = cfg.fake_data_provider()
            self.fake_dataset_test = cfg.fake_data_provider_test()
        self.fake_input = tf.placeholder(
            tf.float32,
            shape=(None, cfg.source_img_size, cfg.source_img_size,
                   cfg.real_img_channels),
            name='fake_input')
        self.fake_input_feature = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='fake_input_feature')
        self.ground_truth = tf.placeholder(
            tf.float32,
            shape=(None, cfg.source_img_size, cfg.source_img_size,
                   cfg.real_img_channels),
            name='ground_truth')
        self.states = tf.placeholder(
            tf.float32, shape=(None, self.cfg.num_state_dim), name='states')
        self.progress = tf.placeholder(tf.float32, shape=(), name='progress')
        self.real_data = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self.cfg.real_img_size, self.cfg.real_img_size,
                   cfg.real_img_channels),
            name='real_data')
        self.real_data_feature = tf.placeholder(
            dtype=tf.float32,
            shape=(None, ),  # self.cfg.feature_size),
            name='real_data_feature')
        self.z = tf.placeholder(tf.float32, shape=(None, cfg.z_dim), name='z')

        # The images with labels of #operations applied
        self.image_pool = []
        self.target_pool_size = cfg.replay_memory_size
        self.fake_output = None
        self.fake_output_feature = None
        if load:
            self.load()

    def load(self):
        self.fill_pool()

    def get_initial_states(self, batch_size):
        states = np.zeros(
            shape=(batch_size, self.cfg.num_state_dim), dtype=np.float32)
        for k in range(batch_size):
            for i in range(len(self.cfg.filters)):
                # states[k, -(i + 1)] = 1 if random.random() < self.cfg.filter_dropout_keep_prob else 0
                # Used or not?
                # Initially nothing has been used
                states[k, -(i + 1)] = 0
        return states

    def fill_pool(self):
        while len(self.image_pool) < self.target_pool_size:
            batch, features = self.fake_dataset.get_next_batch(
                self.cfg.batch_size)
            for i in range(len(batch)):
                self.image_pool.append(
                    Dict(
                        image=batch[i],
                        state=self.get_initial_states(1)[0],
                        feature=features[i]))
        self.image_pool = self.image_pool[:self.target_pool_size]
        assert len(self.image_pool) == self.target_pool_size, '%d, %d' % (
            len(self.image_pool), self.target_pool_size)

    def get_next_RAW(self, batch_size, test=False):
        if test:
            batch = self.fake_dataset_test.get_next_batch(batch_size)[0]
        else:
            batch = self.fake_dataset.get_next_batch(batch_size)[0]
        pool = []
        for img in batch:
            pool.append(Dict(image=img, state=self.get_initial_states(1)[0]))
        return self.records_to_images_and_states(pool)

    def get_next_RAW_test(self, batch_size):
        batch = self.fake_dataset_test.get_next_batch(batch_size)[0]
        pool = []
        for img in batch:
            pool.append(Dict(image=img, state=self.get_initial_states(1)[0]))
        return self.records_to_images_and_states(pool)

    def get_next_RAW_train_all(self):
        batch = self.fake_dataset_train.get_all()[0]
        pool = []
        for img in batch:
            pool.append(Dict(image=img, state=self.get_initial_states(1)[0]))
        return self.records_to_images_and_states(pool)

    def get_dummy_ground_truth(self, batch_size):
        return np.zeros(
            shape=[
                batch_size,
            ] + list(map(int, self.ground_truth.shape[1:])),
            dtype=np.float32)

    def get_next_RAW_test_all(self):
        batch = self.fake_dataset_test.get_all()[0]
        pool = []
        for img in batch:
            pool.append(Dict(image=img, state=self.get_initial_states(1)[0]))
        return self.records_to_images_and_states(pool)

    def get_dummy_ground_truth(self, batch_size):
        return np.zeros(
            shape=[
                batch_size,
            ] + list(map(int, self.ground_truth.shape[1:])),
            dtype=np.float32)

    def get_feed_dict(self, batch_size):
        images, states, features = self.get_next_fake_batch(batch_size)
        if self.cfg.supervised:
            images, ground_truth = images[:, 0], images[:, 1]
        else:
            ground_truth = self.get_dummy_ground_truth(batch_size)
        tmp_real_data, tmp_real_features = self.real_dataset.get_next_batch(
            batch_size)
        return {
            self.states: states,
            self.fake_input: images,
            self.fake_input_feature: features,
            self.ground_truth: ground_truth,
            self.real_data: tmp_real_data,
            self.real_data_feature: tmp_real_features,
            self.z: self.get_noise(batch_size)
        }

    def get_feed_dict_and_states(self, batch_size):
        images, states, features = self.get_next_fake_batch(batch_size)
        if self.cfg.supervised:
            images, ground_truth = images[:, 0], images[:, 1]
        else:
            ground_truth = self.get_dummy_ground_truth(batch_size)
        tmp_real_data, tmp_real_featuers = self.real_dataset.get_next_batch(
            batch_size)
        return {
            self.fake_input: images,
            self.fake_input_feature: features,
            self.ground_truth: ground_truth,
            self.states: states,
            self.real_data: tmp_real_data,
            self.real_data_feature: tmp_real_featuers,
            self.z: self.get_noise(batch_size)
        }, features

    # For training critic: only terminated states should be used.
    def get_replay_feed_dict(self, batch_size):
        images, states, features = self.replay_fake_batch(batch_size)
        if self.cfg.supervised:
            images, ground_truth = images[:, 0], images[:, 1]
        else:
            ground_truth = self.get_dummy_ground_truth(batch_size)
        tmp_real_data, tmp_real_features = self.real_dataset.get_next_batch(
            batch_size)
        return {
            self.fake_output: images,
            self.fake_output_feature: features,
            self.ground_truth: ground_truth,
            self.real_data: tmp_real_data,
            self.real_data_feature: tmp_real_features
        }

    # Not actually used.
    def get_noise(self, batch_size):
        if self.cfg.z_type == 'normal':
            return np.random.normal(0, 1, [batch_size,
                                           self.cfg.z_dim]).astype(np.float32)
        elif self.cfg.z_type == 'uniform':
            return np.random.uniform(
                0, 1, [batch_size, self.cfg.z_dim]).astype(np.float32)
        else:
            assert False, 'Unknown noise type: %s' % self.cfg.z_type

    # Note, we add finished images since the discriminator needs them for training.
    def replace_memory(self, new_images):
        random.shuffle(self.image_pool)
        # Insert only PART of new images
        for r in new_images:
            if r.state[STATE_STEP_DIM] < self.cfg.maximum_trajectory_length or random.random(
            ) < self.cfg.over_length_keep_prob:
                self.image_pool.append(r)
        # ... and add some brand new RAW images
        self.fill_pool()
        random.shuffle(self.image_pool)

    # For supervised learning case, images should be [batch size, 2, size, size, channels]
    @staticmethod
    def records_to_images_and_states(batch):
        images = [x.image for x in batch]
        states = [x.state for x in batch]
        return np.stack(images, axis=0), np.stack(states, axis=0)

    @staticmethod
    def records_to_images_states_features(batch):
        images = [x.image for x in batch]
        states = [x.state for x in batch]
        features = [x.feature for x in batch]
        return np.stack(
            images, axis=0), np.stack(
                states, axis=0), np.stack(
                    features, axis=0)

    @staticmethod
    def images_and_states_to_records(images,
                                     states,
                                     features,
                                     ground_truth=None):
        assert len(images) == len(states)
        assert len(images) == len(features)
        records = []
        if ground_truth is None:
            for img, state, feature in zip(images, states, features):
                records.append(Dict(image=img, state=state, feature=feature))
        else:
            for img, gt, state, feature in zip(images, ground_truth, states,
                                               features):
                img = np.stack([img, gt])
                records.append(Dict(image=img, state=state, feature=feature))
        return records

    def get_next_fake_batch(self, batch_size):
        # print('get_next')
        random.shuffle(self.image_pool)
        assert batch_size <= len(self.image_pool)
        batch = []
        while len(batch) < batch_size:
            if len(self.image_pool) == 0:
                self.fill_pool()
            record = self.image_pool[0]
            self.image_pool = self.image_pool[1:]
            if record.state[STATE_STOPPED_DIM] != 1:
                # We avoid adding any finished images here.
                batch.append(record)
        images, states = self.records_to_images_and_states(batch)
        features = [x.feature for x in batch]
        features = np.stack(features, axis=0)
        return images, states, features

    # We choose terminated states only
    def replay_fake_batch(self, batch_size):
        # print('replay next')
        self.fill_pool()
        random.shuffle(self.image_pool)
        assert batch_size <= len(self.image_pool)
        # batch = self.image_pool[:batch_size]
        batch = []
        counter = 0
        while len(batch) < batch_size:
            counter += 1
            if counter > batch_size * 10:
                assert False, 'No terminated states discovered'
            for i in range(len(self.image_pool)):
                record = self.image_pool[i]
                if record.state[STATE_STOPPED_DIM] > 0:
                    # terminated
                    batch.append(record)
                    if len(batch) >= batch_size:
                        break
        assert len(batch) == batch_size
        # add by cx
        images, states = self.records_to_images_and_states(batch)
        features = [x.feature for x in batch]
        features = np.stack(features, axis=0)
        return images, states, features

    def debug(self):
        tot_trajectory = 0
        for r in self.image_pool:
            tot_trajectory += r.state[STATE_STEP_DIM]
        average_trajectory = 1.0 * tot_trajectory / len(self.image_pool)
        print('# Replay memory: size %d, avg. traj. %.2f' %
              (len(self.image_pool), average_trajectory))
        print('#--------------------------------------------')
