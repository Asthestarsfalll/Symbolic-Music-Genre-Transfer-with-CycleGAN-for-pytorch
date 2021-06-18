import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy


def kernel_initializer(w, mean=0., std=0.02):
    return nn.init.normal_(w, mean, std)


def padding(x, p=3):
    return F.pad(x, (p, p, p, p), mode='reflect')


def cycle_loss(real_a, cycle_a, real_b, cycle_b):
    return F.l1_loss(cycle_a, real_a, reduction='mean') + F.l1_loss(cycle_b, real_b, size_average='mean')


class Generator(nn.Module):
    def __init__(self, dim=64):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, dim, 7, 1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            # (bs,64,64,84)
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1, bias=False),
            nn.InstanceNorm2d(2 * dim, affine=True),
            nn.ReLU(inplace=True),
            # (bs,128,32,42)
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1, bias=False),
            nn.InstanceNorm2d(4 * dim, affine=True),
            nn.ReLU(inplace=True)
            # (bs,256,16,21)
        )

        self.ResNet = nn.ModuleList(
            [ResNetBlock(4 * dim, 3, 1) for i in range(10)])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 3, 2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(2 * dim, affine=True),
            nn.ReLU(inplace=True),
            # (bs,128,32,42)
            nn.ConvTranspose2d(2 * dim, dim, 3, 2, padding=1,
                               output_padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True)
            # (bs,64,64,84)
        )
        self.output = nn.Conv2d(dim, 1, 7, 1, bias=False)
        # (bs,1,64,84)

        for i in range(3):
            self.encoder[3 *
                         i].weight = kernel_initializer(self.encoder[3 * i].weight)
            self.encoder[1 + 3 * i].weight = kernel_initializer(
                self.encoder[1 + 3 * i].weight, 1., 0.02)

        for i in range(2):
            self.decoder[3 *
                         i].weight = kernel_initializer(self.decoder[3 * i].weight)
            self.decoder[1 + 3 *
                         i].weight = kernel_initializer(self.decoder[1 + 3 * i].weight, 1., 0.02)

        self.output.weight = kernel_initializer(self.output.weight)

    def forward(self, x):
        x = padding(x)
        x = self.encoder(x)
        # (bs, 256, 16, 21)
        for i in range(10):
            x = self.ResNet[i](x)
        # (bs,256,16,21)
        x = self.decoder(x)
        x = padding(x)
        x = self.output(x)
        return torch.sigmoid(x)


class ResNetBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1):
        super(ResNetBlock, self).__init__()
        self.dim = dim
        self.ks = kernel_size
        self.s = stride
        self.p = (kernel_size - 1) // 2
        self.Conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, stride, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True)
        )
        self.Conv1[0].weight = kernel_initializer(self.Conv1[0].weight)

        self.Conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, stride, bias=False),
            nn.InstanceNorm2d(dim, affine=True)
        )
        self.Conv2[0].weight = kernel_initializer(self.Conv2[0].weight)

    def __call__(self, x):
        y = padding(x, self.p)
        # (bs,256,18,23)
        y = self.Conv1(y)
        # (bs,256,16,21)
        y = padding(y, self.p)
        # (bs,256,18,23)
        y = self.Conv2(y)
        # (vs,256,16,21)
        out = torch.relu(x + y)
        return out


class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator, self).__init__()
        self.discri = nn.Sequential(
            nn.Conv2d(1, dim, 7, 2, padding=3, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,64,32,42)
            nn.Conv2d(dim, 4 * dim, 7, 2, padding=3, bias=False),
            nn.InstanceNorm2d(4 * dim, affine=True),
            # (bs,256,16,21)
            nn.Conv2d(4 * dim, 1, 7, 1, padding=3, bias=False)
            # (bs,1,16,21)
        )

        for i in [0, 2, 4]:
            self.discri[i].weight = kernel_initializer(self.discri[i].weight)
        self.discri[3].weight = kernel_initializer(
            self.discri[3].weight, 1., 0.02)

    def forward(self, x):
        x = self.discri(x)
        return x


class Classifier(nn.Module):
    def __init__(self, dim=64):
        super(Classifier, self).__init__()
        self.cla = nn.Sequential(
            nn.Conv2d(1, dim, (1, 12), (1, 12), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,64,64,7)
            nn.Conv2d(dim, 2 * dim, (4, 1), (4, 1), bias=False),
            nn.InstanceNorm2d(2 * dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,128,16,7)
            nn.Conv2d(2 * dim, 4 * dim, (2, 1), (2, 1), bias=False),
            nn.InstanceNorm2d(4 * dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,256,8,7)
            nn.Conv2d(4 * dim, 8 * dim, (8, 1), (8, 1), bias=False),
            nn.InstanceNorm2d(8 * dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,512,1,7)
            nn.Conv2d(8 * dim, 2, (1, 7), (1, 7), bias=False)
            # (bs,2,1,1)
        )
        self.softmax = nn.Softmax(dim=1)

        for i in [0, 2, 5, 8, 11]:
            self.cla[i].weight = kernel_initializer(self.cla[i].weight)
        for i in [3, 6, 9]:
            self.cla[i].weigth = kernel_initializer(
                self.cla[i].weight, 1., 0.02)

    def forward(self, x):
        x = self.cla(x)
        # x.squeeze(-1).squeeze(-1)
        x = self.softmax(x)
        return x.squeeze(-1).squeeze(-1)


class Sampler(object):
    def __init__(self, max_length=50):
        self.maxsize = max_length
        self.num = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num < self.maxsize:
            self.images.append(image)
            self.num += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand() * self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand() * self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


class CycleGAN(nn.Module):
    def __init__(self, sigma=0.01, sample_size=50, lamb=10, mode='train'):
        super(CycleGAN, self).__init__()
        assert mode in ['train', '']
        self.G_A2B = Generator(64)
        self.G_B2A = Generator(64)
        self.D_A = Discriminator(64)
        self.D_B = Discriminator(64)
        self.D_A_all = Discriminator(64)
        self.D_B_all = Discriminator(64)
        self.sigma = sigma
        self.mode = mode
        self.sampler = Sampler(sample_size)
        self.lamb = lamb

    def forward(self, real_A, real_B, x_m):
        # blue line
        fake_B = self.G_A2B(real_A)
        cycle_A = self.G_B2A(fake_B)

        # red line
        fake_A = self.G_B2A(real_B)
        cycle_B = self.G_A2B(fake_A)
        if self.mode == 'train':

            [sample_fake_A, sample_fake_B] = self.sampler([fake_A, fake_B])
            gauss_noise = kernel_initializer(real_A, mean=0, std=self.sigma)

            DA_real = self.D_A(real_A + gauss_noise)
            DB_real = self.D_B(real_B + gauss_noise)

            DA_fake = self.D_A(fake_A + gauss_noise)
            DB_fake = self.D_B(fake_B + gauss_noise)

            DA_fake_sample = self.D_A(sample_fake_A + gauss_noise)
            DB_fake_sample = self.D_B(sample_fake_B + gauss_noise)

            DA_real_all = self.D_A_all(x_m + gauss_noise)
            DB_real_all = self.D_B_all(x_m + gauss_noise)

            DA_fake_all = self.D_A_all(sample_fake_A + gauss_noise)
            DB_fake_all = self.D_B_all(sample_fake_B + gauss_noise)

            c_loss = cycle_loss(real_A, cycle_A, real_B, cycle_B)

            # there are to many to return , maybe i should cur
            return self.lamb * c_loss, DA_real, DB_real, DA_fake, DB_fake, DA_fake_sample, DB_fake_sample, DA_real_all, DB_real_all, DA_fake_all, DB_fake_all

        elif self.mode == 'A2B':
            return fake_A
        elif self.mode == 'B2A':
            return fake_B
