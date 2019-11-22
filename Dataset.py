import torch
import torch.utils.data
import numpy as np
import os
from PIL import Image
import glob
import keypoints_from_images
import cv2

img_folder_path = "/img_highres"


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.img_paths = []
        for img_path in glob.glob("/img_highres/**/*.jpg", recursive=True):
            self.img_paths.append(img_path)

    def return_data(self, idx):
        # バッチを読み込むごとに画像データを読み込んでくる
        pair = []
        for img_path in self.img_paths:
            pbib_data = keypoints_from_images.return_keypoints(img_path)  # [Pb,Ib]　あとはIaを前につなげたい
            data_dir = os.path.dirname(img_path)
            for pair_data_path in glob.glob(data_dir + "/*.jpg", recursive=True):
                if pair_data_path == img_path:
                    continue
                pair_data = cv2.imread(pair_data_path)
                pair.append(np.concatenate(pair_data, pbib_data))   # [Ia,Pb,Ib]でコンキャットできたばず　あとは例外処理


        if self.transform:
            out_data = self.transform(pbib_data)

        return pbib_data


myDataset = MyDataset()
