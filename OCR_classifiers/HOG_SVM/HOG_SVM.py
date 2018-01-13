import numpy as np

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
        print("Total number of images is ", num_total)
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
        self.y_test = self.y_train[selected_test_index]



