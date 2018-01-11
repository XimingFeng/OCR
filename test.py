from Shared_Tool.Image_Load_IAM import IAM_Img_Handler
import cv2
import numpy as np

obj = IAM_Img_Handler()
obj.load_imgs(3)
# obj.show_8_pic()
# ('g01-022-06-04', 'ok', 152, 1705, 2003,  77,  40, 'IN', 'IN')
# ('f04-035-04-01', 'ok', 185,  622, 1627, 128,  48, 'HVD', 'had')
# img = obj.get_image_by_index("g01-022-06-04")
# resized_img = cv2.resize(img, (100, 100))
# ret_img = obj.img_threshold(resized_img, 152)
#
# print(ret_img, ret_img)
# cv2.imshow("In original ", img)
# cv2.imshow("In thresholded", ret_img)
# cv2.imshow("Had resized", resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img1 = obj.get_image_by_index("f04-035-04-01")
# ret_img1 = obj.img_threshold(img1, 185)
# resized_img1 = cv2.resize(ret_img1, (100, 100))

# print("The img numpy array for a01-000u-00-06  is ", img.shape
# print("The shresholded img numpy array for a01-000u-00-06  is", img.shape)
# data_batch = np.dstack((resized_img, resized_img1))
# print("The batch of resized images is ", data_batch.shape, data_batch)
# cv2.imshow("In original ", img)
# cv2.imshow("In thresholded", ret_img)
# cv2.imshow("Had resized", ret_img1)

