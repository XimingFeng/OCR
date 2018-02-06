import numpy as np
import cv2
from sklearn import svm
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import time

class Classifier_HOG_SVM(object):

    def __init__(self, imgs, meta, num_train, num_val, orientation=9, pixel_per_cell=64):
        """
        put images and meta data into training, validation and testing sets
        :param imgs: height * width * num_train, stack of images
        :param meta: meta data that contains meta info for all the images
        :param num_train: number of images for training
        :param num_val: number of images for validation
        :param pixel_per_cell: number of pixel per cell. It is used when generating histogram
        """
        start_time = time.time()
        self.img = imgs
        self.meta = meta
        self.num_total = len(self.meta)
        self.imgs_HOG, self.y, self.label_table, self.grad_images = \
            self.data_preprocessing(imgs, meta, orientation=orientation, pixel_per_cell=pixel_per_cell)
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = self.num_total - num_train
        selected_train_index = np.random.choice(range(self.num_total), self.num_train, replace=False)
        self.X_train = self.imgs_HOG[selected_train_index, :]
        self.y_train = self.y[selected_train_index]
        selected_val_index = np.random.choice(range(self.num_train), self.num_val, replace=False)
        self.X_val = self.X_train[selected_val_index, :]
        self.y_val = self.y_train[selected_val_index]
        selected_test_index = np.delete(np.arange(self.num_total), selected_train_index)
        self.X_test = self.imgs_HOG[selected_test_index, :]
        self.y_test = self.y[selected_test_index]
        print("The shape of the training, validation and testing data is ",
        self.X_train.shape, self.X_val.shape, self.X_test.shape)
        print("It takes ", time.time() - start_time, " seconds to initiate this model")

    def data_preprocessing(self, imgs, meta_list, orientation, pixel_per_cell):
        """
        convert string labels of images to int labels
        convert raw images into HOG vectors
        :param img: 200 * 200 * num_train binarized image
        :return: HOG vectors for images, integer label of images and {label_str: label_int} hash table
        """
        # print("Meta data is ", meta_list)
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
        # print("Labels are ", y)
        num_imgs = imgs.shape[2]
        img_hog = []
        grad_images = []
        for img_idx in range(num_imgs):
            img = imgs[:, :, img_idx]
            (descriptor, grad_image) = \
                hog(img,
                    orientations=orientation,
                    pixels_per_cell=(pixel_per_cell, pixel_per_cell),
                    cells_per_block=(2, 2),
                    visualise=True,
                    transform_sqrt=True)
            descriptor = descriptor.reshape((descriptor.shape[0], ))
            img_hog.append(descriptor)
            grad_images.append(grad_image)
            if img_idx % 100 == 0:
                print(img_idx, " images has been preprocessed")
        img_hog = np.array(img_hog)
        grad_images = np.array(grad_images)
        print("Shape of gradient images is ", grad_images.shape)
        return img_hog, y, label_table, grad_images

    def train(self, kernel="linear", C=1, gamma="auto"):
        """
        Training data
        :return:
        """
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma)
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        print("It takes ", time.time() - start_time, " seconds to train the model")

    def predict(self, x):
        """
        The input data
        :param x:
        :return:
        """
        return self.model.predict(x)

    def show_random_histogram(self):
        """
        plot the original images and gradient images in one row
        :return:
        """
        selected_img = np.random.choice(range(self.num_total), 3, replace=False)
        print("We selected the following images", selected_img)
        plt.figure(1)
        axs = 321
        for img_index in selected_img:
            plt.subplot(axs)
            plt.title(self.meta[img_index][8])
            plt.imshow(self.img[:, :, img_index], cmap='Greys')
            plt.subplot(axs + 1)
            plt.title(self.meta[img_index][8] + " gradient image")
            plt.imshow(self.grad_images[img_index, :, :], cmap=plt.cm.gray)
            axs += 2
        plt.show()