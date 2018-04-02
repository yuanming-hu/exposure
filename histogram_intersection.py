import numpy as np
import cv2
import sys
import os
import random
from util import read_set

HIST_BINS = 32


def hist_intersection(a, b):
    return np.minimum(a, b).sum()


def get_statistics(img):
    img = np.clip(img, a_min=0.0, a_max=1.0)
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lum = img[:, :, 0] * 0.27 + img[:, :, 1] * 0.67 + img[:, :, 2] * 0.06
    sat = HLS[:, :, 2].mean()
    return [lum.mean(), lum.std() * 2, sat]


def calc_hist(arr, nbins, xrange):
    h, _ = np.histogram(a=arr, bins=nbins, range=xrange, density=False)
    return h / float(len(arr))


def get_histograms(images):
    statistics = np.array(zip(*map(get_statistics, images)))
    hists = map(lambda x: calc_hist(x, HIST_BINS, (0.0, 1.0)), statistics)
    return hists, statistics


def read_images(src, tag=None, set=None):
    files = os.listdir(src)
    images = []
    if set is not None:
        set = read_set(set)
    for f in files:
        if tag and f.find(tag) == -1:
            continue
        if set is not None:
            if int(f.split('.')[0]) not in set:
                continue
        image = (cv2.imread(os.path.join(src, f))[:, :, ::-1] / 255.0).astype(np.float32)
        longer_edge = min(image.shape[0], image.shape[1])
        for i in range(4):
            sx = random.randrange(0, image.shape[0] - longer_edge + 1)
            sy = random.randrange(0, image.shape[1] - longer_edge + 1)
            new_image = image[sx:sx + longer_edge, sy:sy + longer_edge]
            patch = cv2.resize(new_image, dsize=(80, 80), interpolation=cv2.cv.CV_INTER_AREA)
            for j in range(4):
                target_size = 64
                ssx = random.randrange(0, patch.shape[0] - target_size)
                ssy = random.randrange(0, patch.shape[1] - target_size)
                images.append(patch[ssx:ssx + target_size, ssy:ssy + target_size])
    return images


if __name__ == '__main__':
    output_src = sys.argv[1]
    target_src = sys.argv[2]

    output_imgs = read_images(output_src)
    target_imgs = read_images(target_src)

    output_hists, fake_stats = get_histograms(output_imgs)
    target_hists, real_stats = get_histograms(target_imgs)
    output_hists, real_hists = np.array(output_hists), np.array(target_hists)
    hist_ints = map(hist_intersection, output_hists, real_hists)
    print('Hist. Inter.: %.2f%% %.2f%% %.2f%%' % (hist_ints[0] * 100, hist_ints[1] * 100, hist_ints[2] * 100))
    print('         Avg: %.2f%%' % (sum(hist_ints) / len(hist_ints) * 100))
