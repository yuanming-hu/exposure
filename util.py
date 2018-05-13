import math
import cv2
import tensorflow as tf
import os
import sys
'''
output states:
    0: has rewards?
    1: stopped?
    2: num steps
    3:
'''
STATE_REWARD_DIM = 0
STATE_STOPPED_DIM = 1
STATE_STEP_DIM = 2
STATE_DROPOUT_BEGIN = 3


def get_expert_file_path(expert):
  expert_path = 'data/artists/fk_%s/' % expert
  return expert_path


# From github.com/OlavHN/fast-neural-style
def instance_norm(x):
  epsilon = 1e-9
  mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
  return (x - mean) / tf.sqrt(var + epsilon)


def enrich_image_input(cfg, net, states):
  if cfg.img_include_states:
    print(("states for enriching", states.shape))
    states = states[:, None, None, :] + (net[:, :, :, 0:1] * 0)
    net = tf.concat([net, states], axis=3)
  return net


# based on https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
class Dict(dict):
  """
    Example:
    m = Dict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

  def __init__(self, *args, **kwargs):
    super(Dict, self).__init__(*args, **kwargs)
    for arg in args:
      if isinstance(arg, dict):
        for k, v in arg.items():
          self[k] = v

    if kwargs:
      for k, v in kwargs.items():
        self[k] = v

  def __getattr__(self, attr):
    return self[attr]

  def __setattr__(self, key, value):
    self.__setitem__(key, value)

  def __setitem__(self, key, value):
    super(Dict, self).__setitem__(key, value)
    self.__dict__.update({key: value})

  def __delattr__(self, item):
    self.__delitem__(item)

  def __delitem__(self, key):
    super(Dict, self).__delitem__(key)
    del self.__dict__[key]


def make_image_grid(images, per_row=8, padding=2):
  npad = ((0, 0), (padding, padding), (padding, padding), (0, 0))
  images = np.pad(images, pad_width=npad, mode='constant', constant_values=1.0)
  assert images.shape[0] % per_row == 0
  num_rows = images.shape[0] // per_row
  image_rows = []
  for i in range(num_rows):
    image_rows.append(np.hstack(images[i * per_row:(i + 1) * per_row]))
  return np.vstack(image_rows)


def get_image_center(image):
  if image.shape[0] > image.shape[1]:
    start = (image.shape[0] - image.shape[1]) // 2
    image = image[start:start + image.shape[1], :]

  if image.shape[1] > image.shape[0]:
    start = (image.shape[1] - image.shape[0]) // 2
    image = image[:, start:start + image.shape[0]]
  return image


def rotate_image(image, angle):
  """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

  # Get the image size
  # No that's not an error - NumPy stores image matricies backwards
  image_size = (image.shape[1], image.shape[0])
  image_center = tuple(np.array(image_size) // 2)

  # Convert the OpenCV 3x2 rotation matrix to 3x3
  rot_mat = np.vstack(
      [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

  rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

  # Shorthand for below calcs
  image_w2 = image_size[0] * 0.5
  image_h2 = image_size[1] * 0.5

  # Obtain the rotated coordinates of the image corners
  rotated_coords = [
      (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
      (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
      (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
  ]

  # Find the size of the new image
  x_coords = [pt[0] for pt in rotated_coords]
  x_pos = [x for x in x_coords if x > 0]
  x_neg = [x for x in x_coords if x < 0]

  y_coords = [pt[1] for pt in rotated_coords]
  y_pos = [y for y in y_coords if y > 0]
  y_neg = [y for y in y_coords if y < 0]

  right_bound = max(x_pos)
  left_bound = min(x_neg)
  top_bound = max(y_pos)
  bot_bound = min(y_neg)

  new_w = int(abs(right_bound - left_bound))
  new_h = int(abs(top_bound - bot_bound))

  # We require a translation matrix to keep the image centred
  trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)],
                         [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

  # Compute the tranform for the combined rotation and translation
  affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

  # Apply the transform
  result = cv2.warpAffine(
      image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

  return result


def largest_rotated_rect(w, h, angle):
  """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

  quadrant = int(math.floor(angle / (math.pi / 2))) & 3
  sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
  alpha = (sign_alpha % math.pi + math.pi) % math.pi

  bb_w = w * math.cos(alpha) + h * math.sin(alpha)
  bb_h = w * math.sin(alpha) + h * math.cos(alpha)

  gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

  delta = math.pi - alpha - gamma

  length = h if (w < h) else w

  d = length * math.cos(alpha)
  a = d * math.sin(alpha) / math.sin(delta)

  y = a * math.cos(gamma)
  x = y * math.tan(gamma)

  return (bb_w - 2 * x, bb_h - 2 * y)


def crop_around_center(image, width, height):
  """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

  image_size = (image.shape[1], image.shape[0])
  image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

  if (width > image_size[0]):
    width = image_size[0]

  if (height > image_size[1]):
    height = image_size[1]

  x1 = int(image_center[0] - width * 0.5)
  x2 = int(image_center[0] + width * 0.5)
  y1 = int(image_center[1] - height * 0.5)
  y2 = int(image_center[1] + height * 0.5)

  return image[y1:y2, x1:x2]


# angle: degrees
def rotate_and_crop(image, angle):
  image_width, image_height = image.shape[:2]
  image_rotated = rotate_image(image, angle)
  image_rotated_cropped = crop_around_center(image_rotated,
                                             *largest_rotated_rect(
                                                 image_width, image_height,
                                                 math.radians(angle)))
  return image_rotated_cropped


def lrelu(x, leak=0.2, name="lrelu"):
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


# clamps to 0, 1 with leak
def double_lrelu(x, leak=0.1, name="double_lrelu"):
  with tf.variable_scope(name):
    return tf.minimum(tf.maximum(leak * x, x), leak * x - (leak - 1))


# clamp to lower, upper; leak is RELATIVE
def leaky_clamp(x, lower, upper, leak=0.1, name="leaky_clamp"):
  with tf.variable_scope(name):
    x = (x - lower) / (upper - lower)
    return tf.minimum(tf.maximum(leak * x, x), leak * x -
                      (leak - 1)) * (upper - lower) + lower


class Tee(object):

  def __init__(self, name):
    self.file = open(name, 'w')
    self.stdout = sys.stdout
    self.stderr = sys.stderr
    sys.stdout = self
    sys.stderr = self

  def __del__(self):
    self.file.close()

  def write(self, data):
    self.file.write(data)
    self.stdout.write(data)
    self.file.flush()
    self.stdout.flush()

  def write_to_file(self, data):
    self.file.write(data)

  def flush(self):
    self.file.flush()


def rgb2lum(image):
  image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :,
                                                  1] + 0.06 * image[:, :, :, 2]
  return image[:, :, :, None]


def tanh01(x):
  return tf.tanh(x) * 0.5 + 0.5


def tanh_range(l, r, initial=None):

  def get_activation(left, right, initial):

    def activation(x):
      if initial is not None:
        bias = math.atanh(2 * (initial - left) / (right - left) - 1)
      else:
        bias = 0
      return tanh01(x + bias) * (right - left) + left

    return activation

  return get_activation(l, r, initial)


def merge_dict(a, b):
  ret = a.copy()
  for key, val in list(b.items()):
    if key in ret:
      assert False, 'Item ' + key + 'already exists'
    else:
      ret[key] = val
  return ret


def lerp(a, b, l):
  return (1 - l) * a + l * b


def read_tiff16(fn):
  import tifffile
  import numpy as np
  img = tifffile.imread(fn)
  if img.dtype == np.uint8:
    depth = 8
  elif img.dtype == np.uint16:
    depth = 16
  else:
    print("Warning: unsupported data type {}. Assuming 16-bit.", img.dtype)
    depth = 16

  return (img * (1.0 / (2**depth - 1))).astype(np.float32)


def load_config(config_name):
  scope = {}
  exec ('from config_%s import cfg' % config_name, scope)
  return scope['cfg']


# ======================================================================================================================
# added by Hao He
# ======================================================================================================================
def get_artist_batch(folder, size=128, num=64):
  import os
  js = os.listdir(folder)
  np.random.shuffle(js)
  imgs = np.zeros((num, size, size, 3))
  for i, jpg in enumerate(js[:num]):
    img = cv2.imread(folder + '/' + jpg)
    img = get_image_center(img) / 255.
    imgs[i] = cv2.resize(img, dsize=(size, size))
  return imgs


def show_artist_subnails(folder, size=128, num_row=8, num_column=8):
  imgs = get_artist_batch(folder, size, num_row * num_column)
  return make_image_grid(imgs, per_row=num_row)


def np_tanh_range(l, r):

  def get_activation(left, right):

    def activation(x):
      return np.tanh(x) * (right - left) + left

    return activation

  return get_activation(l, r)


class WB2:

  def filter_param_regressor(self, features):
    log_wb_range = np.log(5)
    color_scaling = np.exp(
        np_tanh_range(-log_wb_range, log_wb_range)(features[:, :3]))
    # There will be no division by zero here unless the WB range lower bound is 0
    return color_scaling

  def process(self, img, param):
    lum = (img[:, :, :, 0] * 0.27 + img[:, :, :, 1] * 0.67 +
           img[:, :, :, 2] * 0.06 + 1e-5)[:, :, :, None]
    tmp = img * param[:, None, None, :]
    tmp = tmp / (tmp[:, :, :, 0] * 0.27 + tmp[:, :, :, 1] * 0.67 +
                 tmp[:, :, :, 2] * 0.06 + 1e-5)[:, :, :, None] * lum
    return tmp


def degrade_images_in_folder(
    folder,
    dst_folder_suffix,
    LIGHTDOWN=True,
    UNBALANCECOLOR=True,):
  import os
  js = os.listdir(folder)
  dst_folder = folder + '-' + dst_folder_suffix
  try:
    os.mkdir(dst_folder)
  except:
    print('dir exist!')
  print('in ' + dst_folder)
  num = 3
  for j in js:
    img = cv2.imread(folder + '/' + j) / 255.
    if LIGHTDOWN:
      for _ in range(num - 1):
        out = pow(img, np.random.uniform(0.4, 0.6)) * np.random.uniform(
            0.25, 0.5)
        cv2.imwrite(dst_folder + '/' + ('L%d-' % _) + j, out * 255.)
      out = img * img
      out = out * (1.0 / out.max())
      cv2.imwrite(dst_folder + '/' + ('L%d-' % num) + j, out * 255.)
    if UNBALANCECOLOR:
      filter = WB2()
      outs = np.array([img] * num)
      features = np.abs(np.random.rand(num, 3))
      for _, out in enumerate(
          filter.process(outs, filter.filter_param_regressor(features))):
        # print out.max()
        out /= out.max()
        out *= np.random.uniform(0.7, 1)
        cv2.imwrite(dst_folder + '/' + ('C%d-' % _) + j, out * 255.)


def vis_images_and_indexs(images, features, dir, name):
  # indexs = np.reshape(indexs, (len(indexs),))
  # print('visualizing images and indexs: ', images.shape, indexs.shape)
  id_imgs = []
  for feature in features:
    img = np.ones((64, 64, 3))
    cv2.putText(img,
                str(feature), (4, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.25,
                (1.0, 0.0, 0.0))
    id_imgs.append(img)
  id_imgs = np.stack(id_imgs, axis=0)
  # print('id imgs: ', id_imgs.shape)

  vis_imgs = np.vstack([images, id_imgs])
  image = make_image_grid(vis_imgs, per_row=images.shape[0])
  vis_dir = dir
  try:
    os.mkdir(vis_dir)
  except:
    pass
  cv2.imwrite(os.path.join(vis_dir, name + '.png'), image[:, :, ::-1] * 255.0)


def read_set(name):
  if name == 'u_test':
    fn = 'data/folds/FiveK_test.txt'
    need_reverse = False
  elif name == 'u_amt':
    fn = 'data/folds/FiveK_test_AMT.txt'
    need_reverse = False
  elif name == '5k':  # add by hao
    return list(range(1, 5001))
  elif name == '2k_train':
    fn = 'data/folds/FiveK_train_first2k.txt'
    need_reverse = False
  elif name == '2k_target':
    fn = 'data/folds/FiveK_train_second2k.txt'
    need_reverse = False
  else:
    assert False, name + ' not found'

  l = []
  ln = 0
  with open(fn, 'r') as f:
    for i in f:
      if i[0] != '#':
        try:
          i = int(i)
          ln += 1
          l.append(i)
        except Exception as e:
          print(e)
          pass
  if need_reverse:
    l = list(set(range(1, 5001)) - set(l))
  return l


'''
    util_image.py
    Copyright (c) 2014     Zhicheng Yan (zhicheng.yan@live.com)
        modified 2017  by Yuanming Hu  (yuanmhu@gmail.com)
        note that some of the color space conversions are NOT exact, like gamma 1.8 or 2.2
'''

import numpy as np
from skimage import color
import tifffile as tiff


class UtilImageError(Exception):
  pass


''' undo gamma correction '''


def linearize_ProPhotoRGB(pp_rgb, reverse=False):
  if not reverse:
    gamma = 1.8
  else:
    gamma = 1.0 / 1.8
  pp_rgb = np.power(pp_rgb, gamma)
  return pp_rgb


def XYZ_chromatic_adapt(xyz, src_white='D65', dest_white='D50'):
  if src_white == 'D65' and dest_white == 'D50':
    M = [[1.0478112, 0.0228866, -0.0501270], \
         [0.0295424, 0.9904844, -0.0170491], \
         [-0.0092345, 0.0150436, 0.7521316]]
  elif src_white == 'D50' and dest_white == 'D65':
    M = [[0.9555766, -0.0230393, 0.0631636], \
         [-0.0282895, 1.0099416, 0.0210077], \
         [0.0122982, -0.0204830, 1.3299098]]
  else:
    raise UtilCnnImageEnhanceError('invalid pair of source and destination white reference %s,%s') \
          % (src_white, dest_white)
  M = np.array(M)
  sp = xyz.shape
  assert sp[2] == 3
  xyz = np.transpose(np.dot(M, np.transpose(xyz.reshape((sp[0] * sp[1], 3)))))
  return xyz.reshape((sp[0], sp[1], 3))


# pp_rgb float in range [0,1], linear ProPhotoRGB
# refernce white is D50
def ProPhotoRGB2XYZ(pp_rgb, reverse=False):
  if not reverse:
    M = [[0.7976749, 0.1351917, 0.0313534], \
         [0.2880402, 0.7118741, 0.0000857], \
         [0.0000000, 0.0000000, 0.8252100]]
  else:
    M = [[1.34594337, -0.25560752, -0.05111183], \
         [-0.54459882, 1.5081673, 0.02053511], \
         [0, 0, 1.21181275]]
  M = np.array(M)
  sp = pp_rgb.shape
  xyz = np.transpose(
      np.dot(M, np.transpose(pp_rgb.reshape((sp[0] * sp[1], sp[2])))))
  return xyz.reshape((sp[0], sp[1], 3))


''' normalize L channel so that minimum of L is 0 and maximum of L is 100 '''


def normalize_Lab_image(lab_image):
  h, w, ch = lab_image.shape[0], lab_image.shape[1], lab_image.shape[2]
  assert ch == 3
  lab_image = lab_image.reshape((h * w, ch))
  L_ch = lab_image[:, 0]
  L_min, L_max = np.min(L_ch), np.max(L_ch)
  #     print 'before normalization L min %f,Lmax %f' % (L_min,L_max)
  scale = 100.0 / (L_max - L_min)
  lab_image[:, 0] = (lab_image[:, 0] - L_min) * scale
  #     print 'after normalization L min %f,Lmax %f' %\
  (np.min(lab_image[:, 0]), np.max(lab_image[:, 0]))
  return lab_image.reshape((h, w, ch))


''' white reference 'D65' '''


def read_tiff_16bit_img_into_XYZ(tiff_fn, exposure=0):
  pp_rgb = tiff.imread(tiff_fn)
  pp_rgb = np.float64(pp_rgb) / (2**16 - 1.0)
  if not pp_rgb.shape[2] == 3:
    print('pp_rgb shape', pp_rgb.shape)
    raise UtilImageError('image channel number is not 3')
  pp_rgb = linearize_ProPhotoRGB(pp_rgb)
  pp_rgb *= np.power(2, exposure)
  xyz = ProPhotoRGB2XYZ(pp_rgb)
  xyz = XYZ_chromatic_adapt(xyz, src_white='D50', dest_white='D65')
  return xyz


def ProPhotoRGB2Lab(img):
  if not img.shape[2] == 3:
    print('pp_rgb shape', img.shape)
    raise UtilImageError('image channel number is not 3')
  img = linearize_ProPhotoRGB(img)
  xyz = ProPhotoRGB2XYZ(img)
  lab = color.xyz2lab(xyz)
  return lab


def linearProPhotoRGB2Lab(img):
  if not img.shape[2] == 3:
    print('pp_rgb shape', img.shape)
    raise UtilImageError('image channel number is not 3')
  xyz = ProPhotoRGB2XYZ(img)
  lab = color.xyz2lab(xyz)
  return lab

import threading
import time


class AsyncTaskManager:

  def __init__(self, target, args=(), kwargs={}):
    self.target = target
    self.args = args
    self.kwargs = kwargs
    self.condition = threading.Condition()
    self.result = None
    self.thread = threading.Thread(target=self.worker)
    self.stopped = False
    self.thread.daemon = True
    self.thread.start()

  def worker(self):
    while True:
      self.condition.acquire()
      while self.result is not None:
        if self.stopped:
          self.condition.release()
          return
        self.condition.notify()
        self.condition.wait()
      self.condition.notify()
      self.condition.release()

      result = (self.target(*self.args, **self.kwargs),)

      self.condition.acquire()
      self.result = result
      self.condition.notify()
      self.condition.release()

  def get_next(self):
    self.condition.acquire()
    while self.result is None:
      self.condition.notify()
      self.condition.wait()
    result = self.result[0]
    self.result = None
    self.condition.notify()
    self.condition.release()
    return result

  def stop(self):
    while self.thread.is_alive():
      self.condition.acquire()
      self.stopped = True
      self.condition.notify()
      self.condition.release()


def test_async_task_manager():
  def task():
    print('begin sleeping...')
    time.sleep(1)
    print('end sleeping.')
    task.i += 1
    print('returns', task.i)
    return task.i

  task.i = 0

  async = AsyncTaskManager(task)
  t = time.time()
  for i in range(5):
    ret = async.get_next()
    # ret = task()
    print('got', ret)
    time.sleep(1)
  async.stop()
  print(time.time() - t)
