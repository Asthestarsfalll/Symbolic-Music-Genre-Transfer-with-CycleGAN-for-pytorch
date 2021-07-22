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
    return F.l1_loss(cycle_a, real_a, reduction='mean') + F.l1_loss(cycle_b, real_b, reduction='mean')


class InstanceNorm2d(nn.InstanceNorm2d):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = False,
                 track_running_stats: bool = False
    ) -> None:
        super(InstanceNorm2d, self).__init__(num_features, affine)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            self.weight = kernel_initializer(self.weight, 1., 0.02)
            nn.init.zeros_(self.bias)

class ResNetBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1):
        super(ResNetBlock, self).__init__()
        self.dim = dim
        self.ks = kernel_size
        self.s = stride
        self.p = (kernel_size - 1) // 2
        self.layer1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, stride, bias=False),
            InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True)
        )
        self.layer1[0].weight = kernel_initializer(self.layer1[0].weight)

        self.layer2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, stride, bias=False),
            InstanceNorm2d(dim, affine=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer2[0].weight = kernel_initializer(self.layer2[0].weight)

    def forward(self, x):
        y = padding(x, self.p)
        # (bs,256,18,23)
        y = self.layer1(y)
        # (bs,256,16,21)
        y = padding(y, self.p)
        # (bs,256,18,23)
        y = self.layer2(y)
        # (bs,256,16,21)
        out = self.relu(x + y)
        return out


class Discriminator(nn.Module):
    def __init__(self, dim=64):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, dim, 7, 2, padding=3, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,64,32,42)
            nn.Conv2d(dim, 4 * dim, 7, 2, padding=3, bias=False),
            InstanceNorm2d(4 * dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,256,16,21)
            nn.Conv2d(4 * dim, 1, 7, 1, padding=3, bias=False)
            # (bs,1,16,21)
        )

        for i in [0, 2, 5]:
            self.discriminator[i].weight = kernel_initializer(self.discriminator[i].weight)

    def forward(self, x):
        x = self.discriminator(x)
        return x


class Generator(nn.Module):
    def __init__(self, dim=64):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, dim, 7, 1, bias=False),
            InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            # (bs,64,64,84)
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1, bias=False),
            InstanceNorm2d(2 * dim, affine=True),
            nn.ReLU(inplace=True),
            # (bs,128,32,42)
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1, bias=False),
            InstanceNorm2d(4 * dim, affine=True),
            nn.ReLU(inplace=True)
            # (bs,256,16,21)
        )

        self.ResNet = nn.Sequential(
            *[ResNetBlock(4 * dim, 3, 1) for i in range(10)])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 3, 2, padding=1,
                               output_padding=1, bias=False),
            InstanceNorm2d(2 * dim, affine=True),
            nn.ReLU(inplace=True),
            # (bs,128,32,42)
            nn.ConvTranspose2d(2 * dim, dim, 3, 2, padding=1,
                               output_padding=1, bias=False),
            InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True)
            # (bs,64,64,84)
        )
        self.output = nn.Conv2d(dim, 1, 7, 1, bias=False)
        # (bs,1,64,84)

        for i in range(3):
            self.encoder[3 *
                         i].weight = kernel_initializer(self.encoder[3 * i].weight)

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

class Classifier(nn.Module):
    def __init__(self, dim=64):
        super(Classifier, self).__init__()
        self.cla = nn.Sequential(
            nn.Conv2d(1, dim, (1, 12), (1, 12), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,64,64,7)
            nn.Conv2d(dim, 2 * dim, (4, 1), (4, 1), bias=False),
            InstanceNorm2d(2 * dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,128,16,7)
            nn.Conv2d(2 * dim, 4 * dim, (2, 1), (2, 1), bias=False),
            InstanceNorm2d(4 * dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,256,8,7)
            nn.Conv2d(4 * dim, 8 * dim, (8, 1), (8, 1), bias=False),
            InstanceNorm2d(8 * dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,512,1,7)
            nn.Conv2d(8 * dim, 2, (1, 7), (1, 7), bias=False)
            # (bs,2,1,1)
        )
        self.softmax = nn.Softmax(dim=1)

        for i in [0, 2, 5, 8, 11]:
            self.cla[i].weight = kernel_initializer(self.cla[i].weight)

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
        assert mode in ['train', 'A2B', 'B2A']
        self.G_A2B = Generator(64)
        self.G_B2A = Generator(64)
        self.D_A = Discriminator(64)
        self.D_B = Discriminator(64)
        self.D_A_all = Discriminator(64)
        self.D_B_all = Discriminator(64)
        self.l2loss = nn.MSELoss(reduction='mean')
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
            gauss_noise = torch.ones_like(real_A)
            gauss_noise = torch.abs(kernel_initializer(gauss_noise, mean=0, std=self.sigma))

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

            # Cycle loss
            c_loss = self.lamb * cycle_loss(real_A, cycle_A, real_B, cycle_B)

            # Generator losses
            g_A2B_loss = self.l2loss(DB_fake, torch.ones_like(DB_fake)) + c_loss
            g_B2A_loss = self.l2loss(DA_fake, torch.ones_like(DA_fake)) + c_loss

            # Discriminator losses
            d_A_loss_real = self.l2loss(DA_real, torch.ones_like(DA_real))
            d_A_loss_fake = self.l2loss(DA_fake_sample, torch.zeros_like(DA_fake_sample))
            d_A_loss = (d_A_loss_real + d_A_loss_fake) / 2
            d_B_loss_real = self.l2loss(DB_real, torch.ones_like(DB_real))
            d_B_loss_fake = self.l2loss(DB_fake_sample, torch.zeros_like(DB_fake_sample))
            d_B_loss = (d_B_loss_real + d_B_loss_fake) / 2

            # All losses
            d_A_all_loss_real = self.l2loss(DA_real_all, torch.ones_like(DA_real_all))
            d_A_all_loss_fake = self.l2loss(DA_fake_all, torch.zeros_like(DA_fake_all))
            d_A_all_loss = (d_A_all_loss_real + d_A_all_loss_fake) / 2
            d_B_all_loss_real = self.l2loss(DB_real_all, torch.ones_like(DB_real_all))
            d_B_all_loss_fake = self.l2loss(DB_fake_all, torch.zeros_like(DB_fake_all))
            d_B_all_loss = (d_B_all_loss_real + d_B_all_loss_fake) / 2

            return (c_loss, g_A2B_loss, g_B2A_loss, d_A_loss, d_B_loss,
                    d_A_all_loss, d_B_all_loss)

        elif self.mode == 'A2B':
            return fake_B, cycle_A
        elif self.mode == 'B2A':
            return fake_A, cycle_B
