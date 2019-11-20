import torch
import torch.utils.data
import numpy as np
import os
from PIL import Image
import glob
import keypoints_from_images

img_folder_path = "/img_highres"


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.img_paths = []
        for img_path in glob.glob("/img_highres/**/*.jpg", recursive=True):
            self.img_paths.append(img_path)

    def return_data(self, idx):
        # バッチを読み込むごとに画像データを読み込んでくる
        out_data = keypoints_from_images.return_keypoints(self.img_paths[idx])  # [Pb,Ib]　あとはIaを前につなげたい

        if self.transform:
            out_data = self.transform(out_data)

        return out_data


myDataset = MyDataset()
