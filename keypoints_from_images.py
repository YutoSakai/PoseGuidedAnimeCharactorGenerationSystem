# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import glob


def draw_keypoints(img, humans, num):
    image_h, image_w = img.shape[:2]
    return_images = []
    all_keypoints_img = np.zeros((image_h, image_w), np.uint8)
    # print(num)
    # print(humans)
    for i, human in enumerate(humans):
        # print("human"+str(i))
        for j, keypoint in enumerate(human):
            black_img = np.zeros((image_h, image_w), np.uint8)  
            if (keypoint[0] == 0) and (keypoint[1] == 0):
                return_images.append(black_img)
                continue
            # draw point
            center = (int(keypoint[0]), int(keypoint[1]))
            return_images.append(cv2.circle(black_img, center, 3, 255, thickness=3, lineType=8, shift=0))
            # cv2.imwrite("keypoint" + str(num) + "-" + str(i) + "-" + str(j) + ".png", cv2.circle(black_img, center, 3, 255, thickness=3, lineType=8, shift=0))
            all_keypoints_img = cv2.circle(all_keypoints_img, center, 3, 255, thickness=3, lineType=8, shift=0)
            # print("all_keypoints_img updated.")
    # cv2.imwrite("keypoint" + str(num) + ".png", all_keypoints_img)
    # print(len(return_images))

    return return_images


# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('/openpose/build/python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

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
            return_img = draw_keypoints(imageToProcess, datum.poseKeypoints, num)
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
