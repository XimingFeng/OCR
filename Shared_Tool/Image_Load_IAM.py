import cv2
import numpy as np
from matplotlib import pyplot as plt


class IAM_Img_Handler(object):

    def __init__(self, content_type="words"):
        """
        load the meta data from /Data/IAM/ascii/***
        define the data type of each column
        abandon invalid data (data that has 'err') at second column
        :param content_type: could be [forms, lines, sentences, words]
        """
        self.content_type = content_type
        if content_type == "words":
            self.meta_data = np.genfromtxt("Data/IAM/ascii/words.txt",
                                             dtype= [
                                                 ("img_index", "U14"),
                                                 ("img_seg_result", "U3"),
                                                 ("img_bin_grey", "i8"),
                                                 ("img_x", "i8"),
                                                 ("img_y", "i8"),
                                                 ("img_w", "i8"),
                                                 ("img_h", "i8"),
                                                 ("img_gram_tag", "U5"),
                                                 ("img_transcript", "U100")
                                             ],
                                             invalid_raise=False)
            delete_list = []
            for i in range(self.meta_data.shape[0]):
                if self.meta_data[i][1] == "err":
                    delete_list.append(i)
            self.meta_data = np.delete(self.meta_data, delete_list, 0)
            # self.meta_data = np.asarray(self.meta_data)
            self.data_size = self.meta_data.shape[0]
            print("Data loaded and deleted data with 'err' flag, the size of valid data is ", self.data_size)

    def show_8_pic(self):
        """
        Show 10 random pictures in one figure (3 Row, 4 Columns)
        :return:
        """
        selected_meta_index = np.random.choice(range(self.data_size), 8, replace=False)
        pics_meta = self.meta_data[selected_meta_index]
        plt.figure(1)
        axes = 341
        if self.content_type == "words":
            for pic_meta in pics_meta:
                path = "Data/IAM/words/"
                location = pic_meta[0].split("-")
                word = pic_meta[8]
                loc1 = location[0]
                loc2 = loc1 + "-" + location[1]
                path += loc1 + "/" + loc2 + "/" + "-".join(location) + ".png"
                img = cv2.imread(path)
                plt.subplot(axes)
                plt.title(word)
                plt.imshow(img)
                axes += 1
            plt.show()

    def get_image_by_index(self, img_index):
        """
        Get image(2-D numpy array cause it is grayed when reading) from according to the index
        :param img_index: the first column of the ascii file, which indicate the location of the image
        :return:
        """
        img = None
        if self.content_type == "words":
            path = "Data/IAM/words/"
            location = img_index.split("-")
            loc1 = location[0]
            loc2 = loc1 + "-" + location[1]
            path += loc1 + "/" + loc2 + "/" + img_index + ".png"
            img = cv2.imread(path, 0)
        return img

    def load_imgs(self, num_img):
        """
        Load random images into 200 * 200 * num_img numpy array
        :param num_img: the number of random images we want for training, validation and testing
        :return img_list: it is 200 * 200 * num_img which are resized and binarized(thresholded) images
        :return pics_meta: the meta data which describe each images
        """
        selected_meta_index = np.random.choice(range(self.data_size), num_img, replace=False)
        pics_meta = self.meta_data[selected_meta_index]
        img_list = []
        for meta in pics_meta:
            img_index = meta[0]
            threshold_index = meta[2]
            img = self.get_image_by_index(img_index)
            resized_img = cv2.resize(img, (200, 200))
            ret_val, thresholded_img = cv2.threshold(resized_img, threshold_index, 255, cv2.THRESH_BINARY)
            img_list.append(thresholded_img)
        img_list = np.dstack(img_list)
        return img_list, pics_meta
