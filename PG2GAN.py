from __future__ import print_function
import argparse
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torch
import cv2
import numpy as np
from torch.autograd import Variable

from Dataset import MyDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--niterG1', type=int, default=10, help='number of epochs to train for G1')
parser.add_argument('--niterG2', type=int, default=10, help='number of epochs to train for G2')
parser.add_argument('--L1_lambda', type=int, default=1, help='L1_lambda for G2')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    # torch.cuda.manual_seed_all(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

data_set = MyDataset()
data_loader = torch.utils.data.DataLoader(data_set, batch_size=opt.batchSize, shuffle=True)
print(len(data_set))


cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


class NetG1(nn.Module):
    """Generator at stage-1 (G1)"""
    def __init__(self):
        super(NetG1, self).__init__()
        self.dim_list = [28, 64, 128, 256, 384, 512, 640, 64]
        # [入力, 1層目, 2層目, 3層目, 4層目, 5層目, 6層目, 全結合層]
        # input bs x 28 x 256 x256
        self.conv_1 = nn.Conv2d(self.dim_list[0], self.dim_list[1], kernel_size=3, stride=1, padding=1)
        # state bs x 64 x 256 x 256
        self.e_block_1 = ResBlock(self.dim_list[1])
        self.conv_2 = nn.Conv2d(self.dim_list[1], self.dim_list[2], 3, 2, 1)
        # state bs x 128 x 128 x128
        self.e_block_2 = ResBlock(self.dim_list[2])
        self.conv_3 = nn.Conv2d(self.dim_list[2], self.dim_list[3], 3, 2, 1)
        # state bs x 256 x 64 x64
        self.e_block_3 = ResBlock(self.dim_list[3])
        self.conv_4 = nn.Conv2d(self.dim_list[3], self.dim_list[4], 3, 2, 1)
        # state bs x 384 x 32 x32
        self.e_block_4 = ResBlock(self.dim_list[4])
        self.conv_5 = nn.Conv2d(self.dim_list[4], self.dim_list[5], 3, 2, 1)
        # state bs x 512 x 16 x16
        self.e_block_5 = ResBlock(self.dim_list[5])
        self.conv_6 = nn.Conv2d(self.dim_list[5], self.dim_list[6], 3, 2, 1)
        # state bs x 640 x 8 x 8
        self.e_block_6 = ResBlock(self.dim_list[6])
        self.conv_7 = nn.Conv2d(self.dim_list[6], self.dim_list[6], 3, 1, 1)
        # state bs x 1280 x 8 x 8
        # have to do view
        self.fc_1 = nn.Linear(self.dim_list[6] * 8 * 8, self.dim_list[7])
        self.fc_2 = nn.Linear(self.dim_list[7], self.dim_list[6] * 8 * 8)
        # have to do view
        # state bs x 1280 x 8x 8
        self.de_block_1 = ResBlock(self.dim_list[6])
        self.deconv_1 = nn.ConvTranspose2d(self.dim_list[6], self.dim_list[5], kernel_size=3, stride=2, padding=1)
        # state bs x 640 x 16 x 16
        self.de_block_2 = ResBlock(self.dim_list[5])
        self.deconv_2 = nn.ConvTranspose2d(self.dim_list[5], self.dim_list[4], kernel_size=3, stride=2, padding=1)
        # state bs x 512 x 32 x 32
        self.de_block_3 = ResBlock(self.dim_list[4])
        self.deconv_3 = nn.ConvTranspose2d(self.dim_list[4], self.dim_list[3], kernel_size=3, stride=2, padding=1)
        # state bs x 384 x 64 x64
        self.de_block_4 = ResBlock(self.dim_list[3])
        self.deconv_4 = nn.ConvTranspose2d(self.dim_list[3], self.dim_list[2], kernel_size=3, stride=2, padding=1)
        # state bs x 256 x 128 x 128
        self.de_block_5 = ResBlock(self.dim_list[2])
        self.deconv_5 = nn.ConvTranspose2d(self.dim_list[2], self.dim_list[1], kernel_size=3, stride=2, padding=1)
        # state bs x 128 x 256 x 256
        self.de_block_6 = ResBlock(self.dim_list[1])
        self.deconv_6 = nn.ConvTranspose2d(self.dim_list[1], 3, kernel_size=3, stride=1, padding=1)
        # state bs x 3 x 256 x 256

    def forward(self, x):
        # encoding
        out_from_e_1 = self.e_block_1(self.conv_1(x))
        out_from_e_2 = self.e_block_2(self.conv_2(out_from_e_1))
        out_from_e_3 = self.e_block_3(self.conv_3(out_from_e_2))
        out_from_e_4 = self.e_block_4(self.conv_4(out_from_e_3))
        out_from_e_5 = self.e_block_5(self.conv_5(out_from_e_4))
        out_from_e_6 = self.e_block_6(self.conv_6(out_from_e_5))
        out_from_e = self.conv_7(out_from_e_6)
        # view and fullchain
        out_from_e = out_from_e.view(-1, self.dim_list[6] * 8 * 8)
        out_from_fc = self.fc_2(self.fc_1(out_from_e))
        out_from_fc = out_from_fc.view(-1, self.dim_list[6], 8, 8)
        # decording and skip connection
        # input bs x 1280 x 8 x 8
        out_from_de_1 = self.deconv_1(self.de_block_1(out_from_fc + out_from_e_6), output_size=out_from_e_5.size())
        out_from_de_2 = self.deconv_2(self.de_block_2(out_from_de_1 + out_from_e_5), output_size=out_from_e_4.size())
        out_from_de_3 = self.deconv_3(self.de_block_3(out_from_de_2 + out_from_e_4), output_size=out_from_e_3.size())
        out_from_de_4 = self.deconv_4(self.de_block_4(out_from_de_3 + out_from_e_3), output_size=out_from_e_2.size())
        out_from_de_5 = self.deconv_5(self.de_block_5(out_from_de_4 + out_from_e_2), output_size=out_from_e_1.size())
        out_from_de_6 = self.deconv_6(self.de_block_6(out_from_de_5 + out_from_e_1))

        return nn.functional.sigmoid(out_from_de_6)


class NetG2(nn.Module):
    """Generator at stage-2 (G2)"""
    def __init__(self):
        super(NetG2, self).__init__()
        # encoder
        # input bs x 6 x 256 x256
        self.conv_1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1)
        # state bs x 64 x 256 x 256
        self.e_block_1 = ResBlock(64)
        self.conv_2 = nn.Conv2d(64, 128, 3, 2, 1)
        # state bs x 128 x 128 x128
        self.e_block_2 = ResBlock(128)
        self.conv_3 = nn.Conv2d(128, 256, 3, 2, 1)
        # state bs x 256 x 64 x64
        self.e_block_3 = ResBlock(256)
        self.conv_4 = nn.Conv2d(256, 384, 3, 2, 1)
        # state bs x 384 x 32 x32
        self.e_block_4 = ResBlock(384)
        # state bs x 384 x 32 x 32

        # decoder
        self.de_block_1 = ResBlock(384)
        self.deconv_1 = nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1)
        # state bs x 384 x 64 x 64
        self.de_block_2 = ResBlock(256)
        self.deconv_2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        # state bs x 256 x 128 x 128
        self.de_block_3 = ResBlock(128)
        self.deconv_3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        # state bs x 128 x 256 x 256
        self.de_block_4 = ResBlock(64)
        self.deconv_4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # encoding
        out_from_e_1 = self.e_block_1(self.conv_1(x))
        out_from_e_2 = self.e_block_2(self.conv_2(out_from_e_1))
        out_from_e_3 = self.e_block_3(self.conv_3(out_from_e_2))
        out_from_e_4 = self.e_block_4(self.conv_4(out_from_e_3))
        # decoding
        out_from_de_1 = self.deconv_1(self.de_block_1(out_from_e_4 + out_from_e_4), output_size=out_from_e_3.size())
        out_from_de_2 = self.deconv_2(self.de_block_2(out_from_de_1 + out_from_e_3), output_size=out_from_e_2.size())
        out_from_de_3 = self.deconv_3(self.de_block_3(out_from_de_2 + out_from_e_2), output_size=out_from_e_1.size())
        out_from_de_4 = self.deconv_4(self.de_block_4(out_from_de_3 + out_from_e_1))

        return nn.functional.sigmoid(out_from_de_4)


class ResBlock(nn.Module):
    """Residual Block (-Conv-ReLU-Conv-ReLU-(+shortcut)-)"""
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        # ch has no change
        self.res_conv_1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.res_relu_1 = nn.ReLU()
        self.res_conv_2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.res_relu_2 = nn.ReLU()

    def forward(self, x):
        out = self.res_relu_1(self.res_conv_1(x))
        out = self.res_relu_2(self.res_conv_2(out))
        out += x

        return out


class NetD(nn.Module):
    """Discriminator at stage-2 (D)"""
    def __init__(self):
        super(NetD, self).__init__()

        ndf = 64
        self.main = nn.Sequential(
            # input bs x 6 x 256 x 256
            nn.Conv2d(6, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # state bs x (ndf) x 128 x 128
            # nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
            # nn.BatchNorm2d(ndf * 2),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2),
            # state bs x (ndf*2) x 64 x 64
            # nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
            # nn.BatchNorm2d(ndf * 4),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2),
            # state bs x (ndf*4) x 32 x 32
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2),
            # state bs x (ndf*8) x 16 x 16
            # nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 16),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2),
            # state bs x (ndf*16) x 8 x 8
            # nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 32),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2),
            # state bs x (ndf) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)


def weights_init(m):
    """custom weights initialization called on netG and netD"""     # なくてもいい
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


'''setup network and initialize weights'''
netG1 = NetG1()
netG1.apply(weights_init)
netG2 = NetG2()
netG2.apply(weights_init)
netD = NetD()
netD.apply(weights_init)

'''criterion'''
L1_criterion = nn.L1Loss()
BCE_criterion = nn.BCELoss()

real_label = 1.0
fake_label = 0.0

'''using cuda'''
if opt.cuda:
    netG1.cuda()
    netG2.cuda()
    netD.cuda()
    L1_criterion.cuda()
    BCE_criterion.cuda()

'''setup optimizer'''
optimizerG1 = optim.Adam(netG1.parameters(), lr=2e-5, betas=(0.5, 0.999))
optimizerG2 = optim.Adam(netG2.parameters(), lr=2e-5, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=2e-5, betas=(0.5, 0.999))

'''training G1'''
for epoch in range(opt.niterG1):
    for i, data in enumerate(data_loader):
        condition_Ia, target_Pb, target_Ib = data
        netG1.zero_grad()
        if opt.cuda:
            condition_Ia = condition_Ia.cuda()
            target_Pb = target_Pb.cuda()
            target_Ib = target_Ib.cuda()
        input_G1 = torch.cat((condition_Ia, target_Pb), 1)  # input_G1 bs x 21 x 256 x256
        pred_Ib = netG1(input_G1)
        errG1 = L1_criterion(pred_Ib, target_Ib)  # this is not pose-mask-loss
        errG1.backward()
        optimizerG1.step()

        if i % 10 == 0:
            print(f'[{epoch:2d}/{opt.niterG1:2d}][{i:3d}/{len(data_loader):3d}] '
                  f'Loss_G1: {errG1.data.item():7.4f}')

    if epoch % 1 == 0:
        # cv2.imwrite(f'out/condition_Ia_trainingG1_epoch_%03d.png' % epoch,
        #             cv2.cvtColor(condition_Ia, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(f'out/target_Ib_trainingG1_epoch_%03d.png' % epoch,
        #             cv2.cvtColor(target_Ib, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(f'out/pred_Ib_trainingG1_epoch_%03d.png' % epoch,
        #             cv2.cvtColor(pred_Ib, cv2.COLOR_RGB2BGR))
        vutils.save_image(condition_Ia[:, [2, 1, 0], :, :], 'out_lossD_0.5/condition_Ia_trainingG1_epoch_%03d.png' % epoch,
                          normalize=True)
        vutils.save_image(target_Ib[:, [2, 1, 0], :, :], 'out_lossD_0.5/target_Ib_trainingG1_epoch_%03d.png' % epoch,
                          normalize=True)
        vutils.save_image(pred_Ib[:, [2, 1, 0], :, :], 'out_lossD_0.5/pred_Ib_trainingG1_epoch_%03d.png' % epoch,
                          normalize=True)
    # do checkpointing
    if epoch % 1 == 0:
        torch.save(netG1.state_dict(), 'outpth/netG1_epoch_%d.pth' % epoch)

'''training Adversarial net (G2 and D)'''
for epoch in range(opt.niterG2):
    for i, data in enumerate(data_loader):
        condition_Ia, target_Pb, target_Ib = data
        netG2.zero_grad()
        netD.zero_grad()

        # label = torch.tensor([real_label for _ in range(condition_Ia.shape[0])])
        label = torch.tensor([random.uniform(0.7, 1.2) for _ in range(condition_Ia.shape[0])])
        # これを作る時点でメモリオーバー

        if opt.cuda:
            condition_Ia = condition_Ia.cuda()
            target_Pb = target_Pb.cuda()
            target_Ib = target_Ib.cuda()
            label = label.cuda()

        input_G1 = torch.cat((condition_Ia, target_Pb), 1)  # input_G1 bs x 28 x 256 x256
        pred_Ib = netG1(input_G1).detach()

        input_G2 = torch.cat((condition_Ia, pred_Ib), 1)  # input_G2 bs x 6 x 256 x256
        refined_pred_Ib = pred_Ib + netG2(input_G2)

        real_pair = torch.cat((condition_Ia, target_Ib), 1)  # input_D bs x 6 x256 x 256
        fake_pair = torch.cat((condition_Ia, refined_pred_Ib), 1)  # input_D bs x 6 x256 x 256

        # train D with pairs
        output_real = netD(real_pair)
        output_real = torch.squeeze(output_real, 1)
        errD_real = BCE_criterion(output_real, label)

        output_fake = netD(fake_pair.detach())  # detach
        output_fake = torch.squeeze(output_fake, 1)
        # label.data.fill_(fake_label)
        label = torch.tensor([random.uniform(0.0, 0.3) for _ in range(condition_Ia.shape[0])])
        errD_fake = BCE_criterion(output_fake, label)

        errD = errD_real + errD_fake
        errD.backward()

        optimizerD.step()

        # train G with pairs
        output_fake = netD(fake_pair)
        output_fake = torch.squeeze(output_fake, 1)
        # label.data.fill_(real_label)  # fake labels are real for generator cost
        label = torch.tensor([random.uniform(0.7, 1.2) for _ in range(condition_Ia.shape[0])])
        errG2BCE = BCE_criterion(output_fake, label)
        errG2L1 = L1_criterion(refined_pred_Ib, target_Ib)
        errG2 = errG2BCE + opt.L1_lambda * errG2L1
        errG2.backward()

        optimizerG2.step()

        if i % 1 == 0:
            print(f'[{epoch:2d}/{opt.niterG2:2d}][{i:3d}/{len(data_loader):3d}] '
                  f'Loss_G2: {errG2.item():7.4f} '
                  f'Loss_G2BCE: {errG2BCE.item():7.4f} '
                  f'Loss_G2L1: {errG2L1.item():7.4f} '
                  f'Loss_D: {errD.item():7.4f} '
                  f'Loss_D_real: {errD_real.item():7.4f} '
                  f'Loss_D_fake: {errD_fake.item():7.4f} ')

    if epoch % 1 == 0:
        # cv2.imwrite(f'out/condition_Ia_trainingG2_epoch_%03d.png' % epoch, condition_Ia)
        # cv2.imwrite(f'out/target_Ib_trainingG2_epoch_%03d.png' % epoch, target_Ib)
        # cv2.imwrite(f'out/refined_pred_Ib_trainingG2_epoch_%03d.png' % epoch, refined_pred_Ib)
        vutils.save_image(condition_Ia[:, [2, 1, 0], :, :], 'out_lossD_0.5/condition_Ia_trainingG2_epoch_%03d.png' % epoch,
                          normalize=True)
        vutils.save_image(target_Ib[:, [2, 1, 0], :, :], 'out_lossD_0.5/target_Ib_trainingG2_epoch_%03d.png' % epoch,
                          normalize=True)
        vutils.save_image(refined_pred_Ib[:, [2, 1, 0], :, :], 'out_lossD_0.5/refined_pred_Ib_trainingG2_epoch_%03d.png' % epoch,
                          normalize=True)

    # do checkpointing
    if epoch % 1 == 0:
        torch.save(netG2.state_dict(), 'outpth/netG2_epoch_%d.pth' % epoch)
        torch.save(netD.state_dict(), 'outpth/netD_epoch_%d.pth' % epoch)
