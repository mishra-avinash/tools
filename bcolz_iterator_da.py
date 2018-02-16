from __future__ import division
import numpy as np
import bcolz
import threading
import cv2
import pdb
import matplotlib.pyplot as plt
from keras.preprocessing.image import flip_axis, random_channel_shift, apply_transform, transform_matrix_offset_center


class BcolzArrayIterator(object):
    """
    Returns an iterator object into Bcolz carray files
    Original version by Thiago Ramon Goncalves Montoya
    Docs (and discovery) by MPJansen
    Refactoring, performance improvements, fixes by Jeremy Howard jfast.ai
        :Example:
        X = bcolz.open('file_path/feature_file.bc', mode='r')
        y = bcolz.open('file_path/label_file.bc', mode='r')
        trn_batches = BcolzArrayIterator(X, y, batch_size=64, shuffle=True)
        model.fit_generator(generator=trn_batches, samples_per_epoch=trn_batches.N, nb_epoch=1)
        :param X: Input features
        :param y: (optional) Input labels
        :param w: (optional) Input feature weights
        :param batch_size: (optional) Batch size, defaults to 32
        :param shuffle: (optional) Shuffle batches, defaults to false
        :param seed: (optional) Provide a seed to shuffle, defaults to a random seed
        :rtype: BcolzArrayIterator
        >>> A = np.random.random((32*10 + 17, 10, 10))
        >>> c = bcolz.carray(A, rootdir='test.bc', mode='w', expectedlen=A.shape[0], chunklen=16)
        >>> c.flush()
        >>> Bc = bcolz.open('test.bc')
        >>> bc_it = BcolzArrayIterator(Bc, shuffle=True)
        >>> C_list = [next(bc_it) for i in range(11)]
        >>> C = np.concatenate(C_list)
        >>> np.allclose(sorted(A.flatten()), sorted(C.flatten()))
        True
    """

    def __init__(self, X, y=None, num_classes=None, batch_size=32, shuffle=False, seed=None, rescale=None, debug=False, segmentation=False,
                 da=False, rotation_range=0., width_shift_range=0., height_shift_range=0., shear_range=0., zoom_range=[1,1],
                 channel_shift_range=0., hue_range=0., saturation_power_range=[1,1], value_power_range=[1,1], fill_mode='nearest', cval=0., horizontal_flip=False, vertical_flip=False):
        if y is not None and len(X) != len(y):
            raise ValueError('X (features) and y (labels) should have the same length'
                             'Found: X.shape = %s, y.shape = %s' % (X.shape, y.shape))
        if batch_size % X.chunklen != 0:
            raise ValueError('batch_size needs to be a multiple of X.chunklen')

        self.chunks_per_batch = batch_size // X.chunklen
        self.X = X
        self.y = y if y is not None else None
        self.num_classes = num_classes
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.shuffle = shuffle
        self.seed = seed
        self.rescale = rescale
        self.debug = debug
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.hue_range = hue_range
        self.saturation_power_range = saturation_power_range
        self.value_power_range = value_power_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.row_axis = 1
        self.col_axis = 2
        self.channel_axis = 3
        self.segmentation = segmentation
        self.da = da

    def reset(self):
        self.batch_index = 0

    def random_transform(self, x, y=None, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)
            if y is not None:
                y = apply_transform(y, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if (self.hue_range != 0) | (self.saturation_power_range[0] != 1) | (self.value_power_range[0] != 1):
            x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(x)
            #pdb.set_trace()
            if self.hue_range != 0:
                hue_shift = np.random.uniform(-self.hue_range, self.hue_range)
                h = cv2.add(h, hue_shift)
            if self.saturation_power_range[0] != 1:
                if np.random.random() < 0.5:
                    sat_shift = np.random.uniform(self.saturation_power_range[0], 1)
                else:
                    sat_shift = np.random.uniform(1, self.saturation_power_range[1])
                #print("Saturation Power: {}".format(sat_shift))
                s = cv2.pow(s, sat_shift)
            if self.value_power_range[0] != 1:
                if np.random.random() < 0.5:
                    val_shift = np.random.uniform(self.value_power_range[0], 1)
                else:
                    val_shift = np.random.uniform(1, self.value_power_range[1])
                #print("Value Power: {}".format(val_shift))
                v = cv2.pow(v/255., val_shift)*255.
            x = cv2.merge((h, s, v))
            x = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)
                if y is not None:
                    y = flip_axis(y, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
                if y is not None:
                    y = flip_axis(y, img_row_axis)
        if y is not None:
            return x, y
        return x

    def next(self):
        #pdb.set_trace()
        with self.lock:
            if self.batch_index == 0:
                if self.seed is not None:
                    np.random.seed(self.seed + self.total_batches_seen)
                if self.N%self.batch_size==0:
                    self.index_array = (np.random.permutation(self.X.nchunks) if self.shuffle 
                                        else np.arange(self.X.nchunks))
                else:
                    self.index_array = (np.random.permutation(self.X.nchunks + 1) if self.shuffle 
                                        else np.arange(self.X.nchunks + 1))

            # batches_x = np.zeros((self.batch_size,)+self.X.shape[1:])
            batches_x, batches_y, batches_w = [], [], []
            for i in range(self.chunks_per_batch):
                current_index = self.index_array[self.batch_index]
                if current_index == self.X.nchunks:
                    batches_x.append(self.X.leftover_array[:self.X.leftover_elements])
                    current_batch_size = self.X.leftover_elements
                else:
                    batches_x.append(self.X.chunks[current_index][:])
                    current_batch_size = self.X.chunklen
                self.batch_index += 1
                #print(self.batch_index)
                self.total_batches_seen += 1

                idx = current_index * self.X.chunklen
                if not self.y is None: batches_y.append(self.y[idx: idx + current_batch_size])
                if self.batch_index >= len(self.index_array):
                    #pdb.set_trace()
                    self.batch_index = 0
                    break

            batch_x = np.concatenate(batches_x).astype(np.float32)

            if self.y is None: return batch_x

            if self.segmentation:
                batch_y = np.expand_dims(np.concatenate(batches_y), -1).astype(np.float32)
            else:
	       # adding below line
		batch_y = np.concatenate(batches_y).astype(np.float32)
               # batch_y = np.zeros((len(batch_x), self.num_classes), dtype=np.float32)
                #for idx, j in enumerate(np.concatenate(batches_y)):
                 #   batch_y[idx, j] = 1
                    
            if self.da==False:
                if self.rescale:
                    batch_x *= self.rescale
                return batch_x, batch_y
            else:
                x_batch, y_batch = [], []
                for idx in range(len(batch_y)):
                    if self.segmentation==True:
                        x, y = self.random_transform(batch_x[idx], batch_y[idx])
                        if self.debug:
                            #print(y.shape)
                            f, axarr = plt.subplots(2, 2, figsize=(15, 10))
                            axarr[0, 0].imshow(batch_x[idx].astype(np.uint8))
                            axarr[0, 1].imshow(x.astype(np.uint8))
                            axarr[1, 0].imshow(batch_y[idx][:, :, 0], cmap="gray")
                            axarr[1, 1].imshow(y[:,:,0], cmap="gray")
                            plt.show()
                        y_batch.append(y)
                    else:
                        if self.debug:
                            f, axarr = plt.subplots(1, 2, figsize=(15, 10))
                            axarr[0].imshow(batch_x[idx].astype(np.uint8))
                        x = self.random_transform(batch_x[idx])
                        if self.debug:
                            axarr[1].imshow(x.astype(np.uint8))
                            plt.show()
                    x_batch.append(x)
                x_batch = np.array(x_batch, np.float32)
                if x_batch.shape[0] == 0:
                    pdb.set_trace()
                if self.rescale:
                    x_batch *= self.rescale
                if self.segmentation == True:
                    y_batch = np.array(y_batch, np.float32)
                    return x_batch, y_batch
                else:
                    return x_batch, batch_y
            

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

