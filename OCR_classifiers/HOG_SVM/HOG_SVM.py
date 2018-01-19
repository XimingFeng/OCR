import numpy as np
import cv2
from sklearn import svm

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
        self.num_total = len(self.meta)
        self.imgs_HOG, self.y, self.label_table = self.data_preprocessing(imgs, meta)
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = self.num_total - num_train
        selected_train_index = np.random.choice(range(self.num_total), self.num_train, replace=False)
        self.X_train = self.imgs_HOG[:, selected_train_index]
        self.y_train = self.y[selected_train_index]
        selected_val_index = np.random.choice(range(self.num_train), self.num_val, replace=False)
        self.X_val = self.X_train[:, selected_val_index]
        self.y_val = self.y_train[selected_val_index]
        selected_test_index = np.delete(np.arange(self.num_total), selected_train_index)
        self.X_test = self.imgs_HOG[:, selected_test_index]
        self.y_test = self.y[selected_test_index]
        print("The shape of the training, validation and tesing data is ",
        self.X_train.shape, self.X_val.shape, self.X_test.shape)

    def data_preprocessing(self, imgs, meta_list):
        """
        convert string labels of images to int labels
        convert raw images into HOG vectors
        :param img: 200 * 200 * num_train binarized image
        :return: HOG vectors for images, integer label of images and {label_str: label_int} hash table
        """
        print("Meta data is ", meta_list)
        y = np.array([])
        label_table = {}
        for i in range(self.num_total):
            word = meta_list[i][8]
            if word not in label_table:
                label_int = i
                label_table[word] = i
            else:
                label_int = label_table[word]
            y = np.append(y, label_int)
        print("Labels are ", y)
        num_imgs = imgs.shape[2]
        window_size = (256, 256)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
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
        for img_idx in range(num_imgs):
            img = imgs[:, :, img_idx]
            descriptor = hog.compute(img)
            descriptor = descriptor.reshape((descriptor.shape[0], ))
            print("The descriptor is ", descriptor.shape)
            img_hog.append(descriptor)
            # img_hog = np.insert(img_hog, descriptor, axis=0)
        # img_hog = np.dstack(img_hog)
        # img_hog = np.reshape(img_hog, (img_hog.shape[1], img_hog[2]))
        img_hog = np.array(img_hog)
        print("The shape of processed input data is ", img_hog.shape)
        return img_hog, y, label_table


    def train(self):
        model = svm.SVC()
        model.fit(self.X_train, self.y_train)

