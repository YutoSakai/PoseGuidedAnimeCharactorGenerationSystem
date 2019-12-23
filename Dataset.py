import torch
import torch.utils.data
import numpy as np
import os
from PIL import Image
import glob
import keypoints_from_images
import cv2
import torchvision.utils as vutils
import subprocess

img_folder_path = "/img_highres"
delete_args = ['rm']


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.img_paths = [img_path for img_path in glob.glob("/img_highres/**/*.jpg", recursive=True)]
        self.pair = []
        self.keypoints_estimate = keypoints_from_images.Keypoints_from_images()  # Keypoints_from_imagesクラスをインスタンス化
        for i, img_path in enumerate(self.img_paths):
            # pbib_data = keypoints_from_images.return_keypoints(img_path)  # 姿勢推定出来ないデータセットを削除する際使用
            # keypoints = self.keypoints_estimate.return_Pb_Ib(img_path)
            # if keypoints[0] is None:
            #     print("None and delete image in dataset")
            #     try:
            #         delete_args.append(img_path)
            #         print(delete_args)
            #         subprocess.check_call(delete_args)
            #         delete_args.pop()
            #     except:
            #         print("Can not delete image file.")
            #     continue
            # if len(keypoints[1]) != 25:
            #     print("25以外なので delete image in dataset")
            #     try:
            #         delete_args.append(img_path)
            #         subprocess.check_call(delete_args)
            #         delete_args.pop()
            #     except:
            #         print("Can not delete image file.")
            #     continue
            data_dir = os.path.dirname(img_path)
            basename = os.path.basename(img_path)
            id_str = basename.split('_')[0]
            same_id_paths = [i for i in glob.glob(str(data_dir) + "/" + str(id_str) + "*.jpg", recursive=True) if i != img_path]
            for same_id_path in same_id_paths:
                self.pair.append((same_id_path, img_path))
            if i == 1:
                break

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, idx):
        Ia_path = self.pair[idx][0]
        Ib_path = self.pair[idx][1]
        # print(Ia_path, Ib_path)
        Ia_data = cv2.imread(Ia_path)
        Ib_data, Pb_data = self.keypoints_estimate.return_Ib_Pb(Ib_path)
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
    vutils.save_image(torch.from_numpy(Ia), 'out/Ia_test.png',
                      normalize=True)
    vutils.save_image(torch.from_numpy(Ib), 'out/Ib_test.png',
                      normalize=True)
