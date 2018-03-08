import numpy as np
import os
import pickle as pickle
import cv2
import random
from data_provider import DataProvider
import multiprocessing.dummy
from util import read_tiff16, read_set

LIMIT = 5000000
image_size = 80
AUGMENTATION_ANGLE = 0
AUGMENTATION_FACTOR = 4
SOURCE_DIR = '/data/yuanming/fivek_dataset/FiveK_Lightroom_Export_InputDayLight/'
BATCHED_DIR = '/data/yuanming/fivek_dataset/sup_batched%daug_daylight' % image_size

try:
    os.mkdir(BATCHED_DIR)
except:
    pass

image_pack_path = os.path.join(BATCHED_DIR, 'image.npy')

def preprocess_RAW_aug():
    image_pack_path = os.path.join(BATCHED_DIR, 'image_raw.npy')
    files = sorted(os.listdir(SOURCE_DIR + '/'))[:LIMIT]
    data = {}
    data['filenames'] = [None for _ in range(len(files))]
    augmentation_factor = AUGMENTATION_FACTOR
    images = np.empty(
        (augmentation_factor * len(files), image_size, image_size, 3),
        dtype=np.float32)

    p = multiprocessing.dummy.Pool(16)

    from util import rotate_and_crop, linearize_ProPhotoRGB

    def load(i):
        fn = files[i]
        data['filenames'][i] = fn
        print('%d / %d' % (i, len(files)))
        image = read_tiff16(os.path.join(SOURCE_DIR + '/', fn))
        image = linearize_ProPhotoRGB(image)

        print(image.dtype)
        print(image.max())
        print(image.mean())
        longer_edge = min(image.shape[0], image.shape[1])

        # Crop some patches so that non-square images are better covered
        for j in range(augmentation_factor):
            sx = random.randrange(0, image.shape[0] - longer_edge + 1)
            sy = random.randrange(0, image.shape[1] - longer_edge + 1)
            new_image = image[sx:sx + longer_edge, sy:sy + longer_edge]
            if AUGMENTATION_ANGLE > 0:
                angle = random.uniform(-1, 1) * AUGMENTATION_ANGLE
                new_image = rotate_and_crop(new_image, angle)
            images[i * augmentation_factor + j] = cv2.resize(
                new_image,
                dsize=(image_size, image_size),
                interpolation=cv2.cv2.INTER_AREA)

    p.map(load, list(range(len(files))))
    print('Writing....')
    pickle.dump(
        data,
        open(os.path.join(BATCHED_DIR, 'meta_raw.pkl'), 'wb'),
        protocol=-1)
    np.save(open(image_pack_path, 'wb'), images)
    print()


class FiveKDataProvider(DataProvider):
    raw_image_pack = None

    @staticmethod
    def get_raw_image_pack():
        if FiveKDataProvider.raw_image_pack is None:
            image_pack_path = os.path.join(BATCHED_DIR, 'image_raw.npy')
            raw_data = np.load(image_pack_path)
            # for i in range(len(raw_data)):
            #    raw_data[i] = (raw_data[i] - raw_data[i].min()) / (raw_data[i].max() - raw_data[i].min())
            FiveKDataProvider.raw_image_pack = raw_data

        return FiveKDataProvider.raw_image_pack

    def __init__(self, set_name, raw=True, *args, **kwargs):
        fn_list = read_set(set_name)
        print(('len', len(fn_list)))
        print(('len set', len(set(fn_list))))
        if raw:
            data = self.get_raw_image_pack()
            print(("#image pack", len(data)))
        else:
            image_pack_path = os.path.join(BATCHED_DIR, 'image_retouched.npy')
            data = np.load(image_pack_path)

        new_data = []
        for i in range(len(data)):
            if (i // AUGMENTATION_FACTOR + 1) in fn_list:
                new_data.append(data[i])

        data = np.stack(new_data)
        print(('final #data', len(data)))
        super(FiveKDataProvider, self).__init__(data, *args, **kwargs)


def test():
    dp = FiveKDataProvider('u_train')
    while True:
        d = dp.get_next_batch(64)
        cv2.imshow('img', d[0][0, :, :, ::-1])
        cv2.waitKey(0)


if __name__ == '__main__':
    # preprocess_RAW_aug()
    test()
