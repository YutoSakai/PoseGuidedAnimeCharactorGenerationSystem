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
        self.img_paths = [img_path for img_path in glob.glob("/img_highres/**/*.jpg", recursive=True)]
        self.pair = []
        for img_path in self.img_paths:
            # pbib_data = keypoints_from_images.return_keypoints(img_path)  # [Pb,Ib]　あとはIaを前につなげたい
            data_dir = os.path.dirname(img_path)
            basename = os.path.basename(img_path)
            id_str = basename.split('_')[0]
            same_id_paths = [i for i in glob.glob(str(data_dir) + "/" + str(id_str) + "*.jpg", recursive=True) if i != img_path]
            for same_id_path in same_id_paths:
                self.pair.append((img_path, same_id_path))
            # for pair_data_path in os.path.basename(data_dir):
            #     if pair_data_path == img_path:
            #         continue
            #     pair_data = cv2.imread(pair_data_path)
            #     pair.append(np.concatenate(pair_data, pbib_data))   # [Ia,Pb,Ib]でコンキャットできたばず　あとは例外処理
            #     print("pair appended")

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        print("into getitem")
        print(idx)
        Ia_path = self.pair[idx][0]
        Ib_path = self.pair[idx][1]
        Ia_data = cv2.imread(Ia_path)
        Ib_data, Pb_data = keypoints_from_images.return_Pb_Ib(Ib_path)
        Ia_data = cv2.resize(Ia_data, Ib_data.shape[:2])

        if self.transform:
            Ia_data = self.transform(Ia_data)
            Ib_data = self.transform(Ib_data)
            Pb_data = self.transform(Pb_data)

        return Ia_data, Pb_data, Ib_data


if __name__ == '__main__':
    mydataset = MyDataset()
    print(len(mydataset))
    print(mydataset.__getitem__(0))
