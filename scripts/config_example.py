from artist import ArtistDataProvider
from fivek import FiveKDataProvider
from critics import critic
from agent import agent_generator
from util import Dict
from filters import *

cfg = Dict()

###########################################################################
###########################################################################
# Here is a list of parameters. Instead of hard coding them in the script, I summarize them here.
# You do not need to modify most of them except the dataset part (see bottom), unless for good reasons.
###########################################################################
###########################################################################

#--------------------------------------------------------------------------

###########################################################################
# Filter Parameters
###########################################################################
cfg.filters = [
    ExposureFilter, GammaFilter, ImprovedWhiteBalanceFilter,
    SaturationPlusFilter, ToneFilter, ContrastFilter, WNBFilter, ColorFilter
]
# Gamma = 1/x ~ x
cfg.curve_steps = 8
cfg.gamma_range = 3
cfg.exposure_range = 3.5
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)

# Masking is DISABLED
cfg.masking = False
cfg.minimum_strength = 0.3
cfg.maximum_sharpness = 1
cfg.clamp = False


###########################################################################
# RL Parameters
###########################################################################
cfg.critic_logit_multiplier = 0.05
cfg.discount_factor = 1.0
# Each time the agent reuse a filter, a penalty is subtracted from the reward. Set to 0 to disable.
cfg.filter_usage_penalty = 1.0
# Use temporal difference error (thereby the value network is used) or directly a single step award (greedy)?
cfg.use_TD = True
# During test, do we use random walk or pick the action with maximized prob.?
cfg.test_random_walk = False
# Replay memory
cfg.replay_memory_size = 128
# Note, a trajectory will be killed either after achieving this value (by chance) or submission
# Thus exploration will leads to kills as well.
cfg.maximum_trajectory_length = 7
cfg.over_length_keep_prob = 0.5
cfg.all_reward = 1.0
# Append input image with states?
cfg.img_include_states = True
# with prob. cfg.exploration, we randomly pick one action during training
cfg.exploration = 0.05
# Action entropy penalization
cfg.exploration_penalty = 0.05
cfg.early_stop_penalty = 1.0


###########################################################################
# CNN Parameters
###########################################################################
cfg.source_img_size = 64
cfg.base_channels = 32
cfg.dropout_keep_prob = 0.5
# G and C use the same feed dict?
cfg.share_feed_dict = True
cfg.shared_feature_extractor = True
cfg.fc1_size = 128
cfg.bnw = False
# number of filters for the first convolutional layers for all networks
#                      (stochastic/deterministic policy, critic, value)
cfg.feature_extractor_dims = 4096


###########################################################################
# GAN Parameters
###########################################################################
# For WGAN only
cfg.use_penalty = True
# LSGAN or WGAN? (LSGAN is not supported now, so please do not change this)
cfg.gan = 'w'


##################################
# Generator
##################################
cfg.generator = agent_generator
cfg.giters = 1

##################################
# Critic & Value Networks
##################################
cfg.critic = critic
cfg.value = critic
cfg.gradient_penalty_lambda = 10
# max iter step, note the one step indicates that a Citers updates of critic and one update of generator
cfg.citers = 5
cfg.critic_initialization = 10
# the upper bound and lower bound of parameters in critic
# when using gradient penalty, clamping is disabled
cfg.clamp_critic = 0.01

# EMD output filter size
cfg.median_filter_size = 101

# Noise defined here is not actually used
cfg.z_type = 'uniform'
cfg.z_dim_per_filter = 16

cfg.num_state_dim = 3 + len(cfg.filters)
cfg.z_dim = 3 + len(cfg.filters) * cfg.z_dim_per_filter
cfg.test_steps = 5

cfg.real_img_size = 64
cfg.real_img_channels = 1 if cfg.bnw else 3


###########################################################################
# Training
###########################################################################
cfg.supervised = False
cfg.batch_size = 64
multiplier = 2
cfg.max_iter_step = int(10000 * multiplier)

##################################
# Learning Rates
##################################
lr_decay = 0.1
base_lr = 5e-5

segments = 3

generator_lr_mul = 0.3
cfg.parameter_lr_mul = 1
cfg.value_lr_mul = 10
critic_lr_mul = 1


def g_lr_callback(t):
    return generator_lr_mul * base_lr * lr_decay**(
        1.0 * t * segments / cfg.max_iter_step)

def c_lr_callback(t):
    return critic_lr_mul * base_lr * lr_decay**(
        1.0 * t * segments / cfg.max_iter_step)

cfg.lr_g = g_lr_callback
cfg.lr_c = c_lr_callback

optimizer = lambda lr: tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9)

cfg.generator_optimizer = optimizer
cfg.critic_optimizer = optimizer

cfg.num_samples = 64
cfg.img_channels = 1 if cfg.bnw else 3
cfg.summary_freq = 100

##################################
# Debugging Outputs
##################################
cfg.vis_draw_critic_scores = True
cfg.vis_step_test = False
cfg.realtime_vis = False
cfg.write_image_interval = int(200 * multiplier)


###########################################################################
# Dataset Parameters
###########################################################################

# Input dataset (train)
cfg.fake_data_provider = lambda: FiveKDataProvider(raw=True, bnw=cfg.bnw,
                                                   output_size=64,
                                                   default_batch_size=cfg.batch_size,
                                                   augmentation=0.3,
                                                   set_name='2k_train',
                                                   )

# Input dataset (test)
cfg.fake_data_provider_test = lambda: FiveKDataProvider(set_name='u_test', raw=True, bnw=cfg.bnw,
                                                        output_size=64,
                                                        default_batch_size=cfg.batch_size,
                                                        augmentation=0.0)

# Target dataset
cfg.real_data_provider = lambda: ArtistDataProvider(augmentation=1.0, name='fk_C',
                                                    output_size=64, bnw=cfg.bnw,
                                                    default_batch_size=cfg.batch_size, target=None,
                                                    set_name='2k_target')
