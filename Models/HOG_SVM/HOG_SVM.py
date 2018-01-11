import numpy as np

class HOG_SVM(object):

    def __init__(self, imgs, meta, num_train, num_val):

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
        self.y_test = self.y_train[selected_test_index]



