import tensorflow as tf
import numpy as np


def pdf_sample(pdf, uniform_noise):
  pdf = pdf / (tf.reduce_sum(input_tensor=pdf, axis=1, keepdims=True) + 1e-36)
  cdf = tf.cumsum(pdf, axis=1, exclusive=True)
  indices = tf.reduce_sum(
      input_tensor=tf.cast(tf.less(cdf, uniform_noise), tf.int32), axis=1) - 1
  return indices


def pdf_sample_2d(pdf, uniform_noise):
  height, width = pdf.get_shape()[1], pdf.get_shape()[2]
  pdf = tf.reshape(pdf, (int(pdf.get_shape()[0]), -1))
  indices_1d = pdf_sample(pdf, uniform_noise)
  indices = tf.stack(
      [tf.clip_by_value(indices_1d / width, 0, height - 1), indices_1d % width],
      axis=1)
  return indices


def test1():
  import cv2
  batch_size = 1024
  img = cv2.imread('data/doggy.jpg').mean(axis=2)

  pdf_batch = np.empty(
      shape=(batch_size, img.shape[0], img.shape[1]), dtype=np.float32)

  for i in range(batch_size):
    pdf_batch[i] = img

  pdf = tf.compat.v1.placeholder(tf.float32, (batch_size, img.shape[0], img.shape[1]))
  noise = tf.compat.v1.placeholder(tf.float32, (batch_size, 1))

  with tf.compat.v1.Session() as sess:
    indices = pdf_sample_2d(pdf, noise)
    image_buffer = np.zeros(
        shape=(img.shape[0], img.shape[1]), dtype=np.float32)

    while True:
      indices_out = sess.run(
          indices,
          feed_dict={pdf: pdf_batch,
                     noise: np.random.rand(batch_size, 1)})

      for ind in indices_out:
        image_buffer[ind[0]][ind[1]] += 1

      cv2.imshow('img', image_buffer / np.max(image_buffer))
      cv2.waitKey(30)


def test2():
  batch_size = 1024
  n = 3

  pdf_batch = [[2.0**i for i in range(1, n + 1)] for _ in range(batch_size)]

  pdf = tf.compat.v1.placeholder(tf.float32, (batch_size, n))
  noise = tf.compat.v1.placeholder(tf.float32, (batch_size, 1))

  counter = [0 for _ in range(n)]

  with tf.compat.v1.Session() as sess:
    indices = pdf_sample(pdf_batch, noise)

    for i in range(1000):
      indices_out = sess.run(
          indices,
          feed_dict={pdf: pdf_batch,
                     noise: np.random.rand(batch_size, 1)})
      for i in indices_out:
        counter[indices_out[i]] += 1

    for i in range(n):
      print(counter[i] * 1.0 / 100 / batch_size)


if __name__ == '__main__':
  test2()
  # test()
