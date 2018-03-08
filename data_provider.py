import numpy as np
import random
import cv2
from async_task_manager import AsyncTaskManager
from util import rotate_and_crop


class DataProvider(object):
    def __init__(self,
                 data,
                 output_size=-1,
                 limit=-1,
                 synchronous=False,
                 augmentation=0,
                 bnw=False,
                 blur=False,
                 default_batch_size=64,
                 train=True,
                 seperation=None,
                 image_scaling=1.0,
                 *args,
                 **kwargs):
        print((data.shape))
        self.blur = blur
        if limit == -1:
            limit = data.shape[0]
        elif isinstance(limit, float):
            limit = int(data.shape[0] * limit)
        else:
            limit = limit
        self.image_scaling = image_scaling
        self.data = data[:limit]
        if seperation is not None:
            seperator = int(round(len(self.data) * seperation))
            if train:
                self.data = self.data[:seperator]
            else:
                self.data = self.data[seperator:]
        self.bnw = bnw
        if self.bnw:
            self.data = 0.27 * self.data[:, :, :,
                                         0] + 0.67 * self.data[:, :, :,
                                                               1] + 0.06 * self.data[:, :, :,
                                                                                     2]
            self.data = self.data[:, :, :, None]
        self.num_images = len(self.data)
        self.default_batch_size = default_batch_size
        self.image_size = data.shape[1:3]
        self.augmentation = augmentation
        self.indices = list(range(self.num_images))
        random.shuffle(self.indices)
        self.synchronous = synchronous
        self.async_task = None
        if output_size == -1:
            self.output_size = data.shape[1:3]
        else:
            self.output_size = (output_size, output_size)


    def augment(self, img, strength):
        s = self.output_size[0]
        start_x = random.randrange(0, img.shape[0] - s + 1)
        start_y = random.randrange(0, img.shape[1] - s + 1)
        img = img[start_x:start_x + s, start_y:start_y + s]
        ### No resizing and rotating....
        # img = rotate_and_crop(img, (random.random() - 0.5) * strength * 300)
        # img = cv2.resize(img, self.output_size)
        if random.random() < 0.5:
            # left-right flip
            img = img[:, ::-1]
        if len(img.shape) < 3:
            img = img[:, :, None]
        if self.blur:
            angle = random.uniform(-1, 1) * 10
            # img = cv2.GaussianBlur(img, (3, 3), 0)
            img = rotate_and_crop(img, angle)
            img = rotate_and_crop(img, -angle)
            img = cv2.resize(img, dsize=self.output_size)
        return img

    def get_next_batch_(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            s = min(len(self.indices), batch_size - len(batch))
            batch += self.indices[:s]
            self.indices = self.indices[s:]
            if len(self.indices) == 0:
                self.indices = list(range(self.num_images))
                random.shuffle(self.indices)
        batch_images = np.empty(
            (batch_size, ) + self.output_size + self.data.shape[3:],
            dtype=self.data.dtype)
        if self.augmentation > 0:
            for i in range(len(batch)):
                batch_images[i] = self.augment(self.data[batch[i]],
                                               self.augmentation)
        else:
            for i in range(len(batch)):
                batch_images[i] = cv2.resize(self.data[batch[i]],
                                             self.output_size)
        batch = np.array(batch)

        ## Hao
        return batch_images * self.image_scaling, np.zeros((batch_size, ))
        # print(batch.shape)

        # return batch_images * self.image_scaling, batch # np.zeros((batch_size,))

    def get_next_batch(self, batch_size):
        if self.synchronous or (self.async_task
                                and batch_size != self.default_batch_size):
            return self.get_next_batch_(batch_size)
        else:
            if self.async_task is None:
                self.async_task = AsyncTaskManager(
                    target=self.get_next_batch_,
                    args=(self.default_batch_size, ))
            if batch_size != self.default_batch_size:
                ret = self.get_next_batch_(batch_size)
            else:
                ret = self.async_task.get_next()
            return ret

    def get_random_batch(self, batch_size):
        indices = list(range(self.num_images))
        random.shuffle(indices)
        indices = indices[:batch_size]
        return self.data[indices], np.zeros((self.num_images, ))

    # Returns a list of image batches
    # the last one may not be a full batch
    def get_test_batches(self, batch_size):
        batches = []
        for i in range((len(self.data) + batch_size - 1) // batch_size):
            batch = []
            for img in self.data[i * batch_size:(i + 1) * batch_size]:
                img *= self.image_scaling
                if self.augmentation > 0:
                    batch.append(self.augment(img, self.augmentation))
                else:
                    batch.append(cv2.resize(img, self.output_size))
            batch = np.stack(batch, axis=0)
            batches.append(batch)
        return batches, None
