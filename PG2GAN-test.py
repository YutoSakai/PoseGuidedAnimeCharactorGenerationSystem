from __future__ import print_function
import argparse
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torch
from torch.autograd import Variable

from Dataset import MyDataset
import PG2GAN
import keypoints_from_images
import cv2
import numpy as np

image_path = "imagedir/image.png"
target_image_path = "imagedir/target_image.png"

def test(Ia, Pb):
    netG1 = PG2GAN.NetG1()
    netG1.load_state_dict(torch.load('outpth/netG1_epoch_%d.pth'))
    netG2 = PG2GAN.NetG2()
    netG2.load_state_dict(torch.load('outpth/netG2_epoch_%d.pth'))

    G1_x = torch.cat((Ia, Pb), 1)
    G1_out = netG1(G1_x)
    G2_x = torch.cat((Ia, G1_out), 1)
    G2_out = netG2(G2_x)

    vutils.save_image(G1_out, 'out/test_G1_out.png', normalize=True)
    vutils.save_image(G2_out, 'out/test_G2_out.png', normalize=True)


if __name__ == "__main__":
    keypoints_estimate = keypoints_from_images.Keypoints_from_images()
    Ia = cv2.imread(image_path)
    Ib, Pb = keypoints_estimate.return_Ib_Pb(target_image_path)
    Pb = np.array(Pb, dtype=np.float32)
    Ia = cv2.resize(Ia, (Ib.shape[1], Ib.shape[0]))
    Pb = Pb.transpose((1, 2, 0)).astype(np.float32)
    Ia = cv2.resize(Ia, (256, 256))
    Pb = cv2.resize(Pb, (256, 256))
    Ia = Ia.transpose((2, 0, 1)).astype(np.float32)
    Pb = Pb.transpose((2, 0, 1)).astype(np.float32)
    Ia /= 255
    Pb /= 255

    test(Ia, Pb)


