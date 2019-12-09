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
            keypoints_estimate = keypoints_from_images.Keypoints_from_images()  #Keypoints_from_imagesクラスをメソッド化
            keypoints = keypoints_estimate.return_Pb_Ib(img_path)
            if keypoints[0] is None:
                print("None")
                continue
            if len(keypoints[1]) != 25:
                print("25以外")
                continue
            data_dir = os.path.dirname(img_path)
            basename = os.path.basename(img_path)
            id_str = basename.split('_')[0]
            same_id_paths = [i for i in glob.glob(str(data_dir) + "/" + str(id_str) + "*.jpg", recursive=True) if i != img_path]
            for same_id_path in same_id_paths:
                self.pair.append((same_id_path, img_path))
                if len(self.pair) > 20:
                    return
            # for pair_data_path in os.path.basename(data_dir):
            #     if pair_data_path == img_path:
            #         continue
            #     pair_data = cv2.imread(pair_data_path)
            #     pair.append(np.concatenate(pair_data, pbib_data))   # [Ia,Pb,Ib]でコンキャットできたばず　あとは例外処理
            #     print("pair appended")

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        Ia_path = self.pair[idx][0]
        Ib_path = self.pair[idx][1]
        print(Ia_path, Ib_path)
        Ia_data = cv2.imread(Ia_path)
        Ib_data, Pb_data = keypoints_from_images.return_Pb_Ib(Ib_path)
        Pb_data = np.array(Pb_data, dtype=np.float32)
        Ia_data = cv2.resize(Ia_data, (Ib_data.shape[1], Ib_data.shape[0]))
        Pb_data = Pb_data.transpose((1, 2, 0)).astype(np.float32)
        Ia_data = cv2.resize(Ia_data, (256, 256))
        Ib_data = cv2.resize(Ib_data, (256, 256))
        Pb_data = cv2.resize(Pb_data, (256, 256))
        Ia_data = Ia_data.transpose((2, 0, 1)).astype(np.float32)
        Ib_data = Ib_data.transpose((2, 0, 1)).astype(np.float32)
        Pb_data = Pb_data.transpose((2, 0, 1)).astype(np.float32)
        Ia_data /= 255
        Ib_data /= 255
        Pb_data /= 255

        if self.transform:
            Ia_data = self.transform(Ia_data)
            Ib_data = self.transform(Ib_data)
            Pb_data = self.transform(Pb_data)

        return Ia_data, Pb_data, Ib_data


if __name__ == '__main__':
    mydataset = MyDataset()
    Ia, Pb, Ib = mydataset.__getitem__(0)
    print(Ia.shape)
    print(Pb.shape)
    print(Ib.shape)
    print(Ia.dtype)
    print(Pb.dtype)
    print(Ib.dtype)
