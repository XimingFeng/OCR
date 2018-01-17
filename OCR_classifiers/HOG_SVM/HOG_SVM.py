import numpy as np
import cv2

class classifier_HOG_SVM(object):

    def __init__(self, imgs, meta, num_train, num_val):
        """
        put images and meta data into training, validation and testing sets
        :param imgs: height * width * num_train, stack of images
        :param meta: meta data that contains meta info for all the images
        :param num_train: number of images for training
        :param num_val: number of images for validation
        """
        self.img = imgs
        self.meta = meta
        num_total = len(self.meta)
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_total - num_train
        selected_train_index = np.random.choice(range(num_total), self.num_train, replace=False)
        self.X_train = self.img[:, :, selected_train_index]
        self.y_train = self.meta[selected_train_index]
        selected_val_index = np.random.choice(range(self.num_train), self.num_val, replace=False)
        self.X_val = self.X_train[:, :, selected_val_index]
        self.y_val = self.y_train[selected_val_index]
        selected_test_index = np.delete(np.arange(num_total), selected_train_index)
        self.X_test = self.img[:, :, selected_test_index]
        self.y_test = self.meta[selected_test_index]

    def data_preprocessing(self, img):
        """
        convert raw images into HOG vectors
        :param img: 200 * 200 * num_train binarized image
        :return: HOG vectors for images
        """
        window_size = (200, 200)
        block_size = (20, 20)
        block_stride = (2, 2)
        cell_size = (10, 10)
        n_bins = 9
        derive_aperture = 1
        wind_sigma = -1
        histogram_norm_type = 0
        l2_hys_threshold = 0.2
        gamma_correction = 1
        n_levels = 64
        signed_gradients = True
        hog = cv2.HOGDescriptor(
            window_size,
            block_size,
            block_stride,
            cell_size,
            n_bins,
            derive_aperture,
            wind_sigma,
            histogram_norm_type,
            l2_hys_threshold,
            gamma_correction,
            n_levels,
            signed_gradients
        )
        img_hog = []
        for img_idx in range(self.num_train):
            img = self.X_train[:, :, img_idx]
            descriptor = hog.compute(img)
            img_hog.append(descriptor)
        img_hog = np.dstack(img_hog)
        return img_hog



