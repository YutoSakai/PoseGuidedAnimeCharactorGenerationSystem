import cv2
import numpy as np

@staticmethod
def draw_keypoints(img, keypoints):
    image_h, image_w = img.shape[:2]
    return_images = []
    for i, keypoint in enumerate(keypoints):
        # draw point
        black_img = np.zeros((image_h, image_w), np.uint8)  # 黒い画像を作る(1次元)
        center = (int(keypoint[0]), int(keypoint[1]))
        return_images.append(cv2.circle(black_img, center, 3, 255, thickness=3, lineType=8, shift=0))
        cv2.imwrite("keypoint" + str(i) + ".png",cv2.circle(black_img, center, 3, 255, thickness=3, lineType=8, shift=0))

    return return_images

# image_h, image_w = 288, 288
# keypoint_x, keypoint_y = 0.5, 0.5
# center = (int(keypoint_x * image_w + 0.5), int(keypoint_y * image_h + 0.5))
# black_img = np.zeros((image_h, image_w, 3), np.uint8)  #黒い画像を作る
# black_img1 = np.zeros((image_h, image_w), np.uint8)  #黒い画像を作る
# cv2.imwrite("black.png", black_img)
# cv2.imwrite("black1.png", black_img1)
# cv2.imwrite("black_circle.png", cv2.circle(black_img1, center, 3, 255, thickness=3, lineType=8, shift=0))
# print("--------------")
# print(black_img)
# print("--------------")
# print(black_img1)