import torch
import torch.utils.data
import numpy as np
import os
from PIL import Image
import glob

img_folder_path = "../deepfashion/img_highres"

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.data = []
        self.img_paths = []
        self.keypoints = []
        self.label = []
        for img_path in glob.glob("/img_highres/**/*.jpg", recursive=True):
            try:
                keypoint = np.load(file=img_path[:-4]+".npy")
                self.img_paths.append(img_path)
                self.keypoints.append(keypoint)
                self.data.append(np.concatenate((img_path, keypoint), axis=1))
                print(data)
                exit(1)
            except:
                continue
        print(len(self.path))
        exit(1)
        for root, dirs, files in os.walk("../deepfashion/img_highres"):
            print('---------')
            print("root:" + root)
            print(dirs)
            print(files)
            pair = []
            if files!=[]:
                for i, file in enumerate(files):
                    print(file[1])
            # for dir in dirs:
            #
            #     for file in files:
            #         file = os.path.join(root, file)
            #         print(file)
            #         self.path.append(file)  # 0 から (data_num-1) までのリスト　いつかはここを画像にせな
            #         self.label.append(np.float32(1))  # 偶数ならTrue 奇数ならFalse

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        # バッチを読み込むごとに画像データを読み込んでくる
        out_data = Image.open(self.path[idx])
        # リサイズ
        out_data = Image.resize(out_data)
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

myDataset = MyDataset()
