import numpy as np
import os
import cv2
import random
from util import get_image_center
from data_provider import DataProvider

LIMIT = 10000
SOURCE_DIR = 'data/artists/'

# The data provider for loading a set of images from data/``name''


class ArtistDataProvider(DataProvider):
    def __init__(self,
                 read_limit=-1,
                 name='C',
                 main_size=80,
                 crop_size=64,
                 augmentation_factor=4,
                 *args,
                 **kwargs):
        folder = os.path.join(SOURCE_DIR, name)
        files = os.listdir(folder)
        if read_limit != -1:
            files = files[:read_limit]
        data = []
        files.sort()
        for f in files:
            image = (cv2.imread(os.path.join(folder, f))[:, :, ::-1] /
                     255.0).astype(np.float32)
            image = get_image_center(image)
            # image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
            # data.append(image)
            image = cv2.resize(
                image, (main_size, main_size), interpolation=cv2.INTER_AREA)
            for i in range(augmentation_factor):
                new_image = image
                if random.random() < 0.5:
                    new_image = new_image[:, ::-1, :]
                sx, sy = random.randrange(main_size - crop_size +
                                          1), random.randrange(
                                              main_size - crop_size + 1)
                data.append(new_image[sx:sx + crop_size, sy:sy + crop_size])
        data = np.stack(data, axis=0)
        print("# image after augmentation =", len(data))
        super(ArtistDataProvider, self).__init__(data, *args, **kwargs)


def test():
    dp = ArtistDataProvider('C')
    while True:
        d = dp.get_next_batch(64)
        cv2.imshow('img', d[0][0, :, :, ::-1])
        cv2.waitKey(0)


if __name__ == '__main__':
    test()
    # preprocess()
