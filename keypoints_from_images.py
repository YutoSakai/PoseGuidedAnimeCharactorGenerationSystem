# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import argparse
import time
import numpy as np
import glob
sys.path.append('/openpose/build/python')
from openpose import pyopenpose as op


class Keypoints_from_images:
    def __init__(self):
        self.params = dict()
        self.params["model_folder"] = "/openpose/models/"
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(self.params)
        self.opWrapper.start()

    def draw_keypoints(self, img, humans):
        image_h, image_w = img.shape[:2]
        return_images = []
        all_keypoints_img = np.zeros((image_h, image_w), np.uint8)
        for i, human in enumerate(humans):
            for j, keypoint in enumerate(human):
                black_img = np.zeros((image_h, image_w), np.uint8)
                if (keypoint[0] == 0) and (keypoint[1] == 0):
                    return_images.append(black_img)
                    continue
                # draw point
                center = (int(keypoint[0]), int(keypoint[1]))
                return_images.append(cv2.circle(black_img, center, 3, 255, thickness=3, lineType=8, shift=0))
                all_keypoints_img = cv2.circle(all_keypoints_img, center, 3, 255, thickness=3, lineType=8, shift=0)
        # cv2.imwrite("keypoint" + str(num) + ".png", all_keypoints_img)

        return return_images

    def return_Ib_Pb(self, imagePath):    # [Pb, Ib]をリターン　ポーズが取れなかった場合はNoneをリターン
        # print(imagePath)
        datum = op.Datum()
        image = cv2.imread(imagePath)
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop([datum])
        try:
            int(datum.poseKeypoints)
            return None, None
        except:
            keypoints_images = self.draw_keypoints(image, datum.poseKeypoints)
            # return image, keypoints_images
            return_image = cv2.resize(image, (int(image.shape[0]/8), int(image.shape[1]/8)))
            return_keypoints_images = []
            for keypoints_image in keypoints_images:
                resize_keypoints_image = cv2.resize(keypoints_image, (int(keypoints_image.shape[0] / 8), int(keypoints_image.shape[1] / 8)))
                height, width, channels = resize_keypoints_image.shape[:3]
                    # 取得結果（幅，高さ，チャンネル数，depth）を表示
                print("width: " + str(width))
                print("height: " + str(height))
                print("channels: " + str(channels))
                return_keypoints_images.append(resize_keypoints_image)
            return return_image, return_keypoints_images

if __name__ == '__main__':

    keypoints_estimate = Keypoints_from_images()
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="/img_highres", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/openpose/models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Read frames on directory
        # imagePaths = op.get_images_on_directory(args[0].image_dir);
        imagePaths = glob.glob("/img_highres/**/*.jpg", recursive=True)
        start = time.time()

        # Process and display images
        for num, imagePath in enumerate(imagePaths):
            print(imagePath)
            datum = op.Datum()
            imageToProcess = cv2.imread(imagePath)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])
            # print(datum.poseKeypoints)
            try:
                int(datum.poseKeypoints)
                continue
            except:
                return_img = keypoints_estimate.draw_keypoints(imageToProcess, datum.poseKeypoints)
                np.save(imagePath[:-4] + "_keypoints", return_img)
                # print("Body keypoints: \n" + str(datum.poseKeypoints))

                if not args[0].no_display:
                    cv2.imwrite("test04.png", datum.cvOutputData)
                    key = cv2.waitKey(15)
                    if key == 27: break

        end = time.time()
        print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
    except Exception as e:
        # print(e)
        sys.exit(-1)
