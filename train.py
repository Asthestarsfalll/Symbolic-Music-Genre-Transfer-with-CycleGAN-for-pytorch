import os
import glob
from torch import optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import CycleGAN
from dataloader import MusicDataset, ToTensor

# ------- 1. define loss function --------

l1loss = nn.L1Loss(reduction='mean')
l2loss = nn.MSELoss(reduction='mean')

# ------- 2. set the directory of training dataset --------

model_name = 'JC'  # J       C PC JP

data_dir = os.path.join(os.getcwd(), 'traindata' + os.sep)
tra_a_dir = os.path.join('JC_C', 'train' + os.sep)
tra_b_dir = os.path.join('JC_J', 'train' + os.sep)

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
if not os.path.exists(model_dir + model_name):
    os.makedirs(model_dir + model_name)

epoch_num = 30
batch_size_train = 1
batch_size_val = 1
val_num = 0
gamma = 1

tra_a_name_list = glob.glob(data_dir + tra_a_dir + '*.*')
tra_b_name_list = glob.glob(data_dir + tra_b_dir + '*.*')

print('--' * 20)
print('A: ', len(tra_a_name_list))
print('B: ', len(tra_b_name_list))
train_num = len(tra_a_name_list)
print('--' * 20)

music_dataset = MusicDataset(
    a_dir_list=tra_a_name_list,
    b_dir_list=tra_b_name_list,
    transform=ToTensor())
music_dataloader = DataLoader(
    music_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

# ------- 3. define model --------
net = CycleGAN(sigma=0.01, sample_size=50, lamb=10, mode='train')

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    net.cuda()

print("---define optimizer...")
optimizer_GA2B = optim.Adam(net.G_A2B.parameters(), lr=0.0002, betas=(
    0.5, 0.999), eps=1e-08, weight_decay=0)
optimizer_GB2A = optim.Adam(net.G_B2A.parameters(), lr=0.0002, betas=(
    0.5, 0.999), eps=1e-08, weight_decay=0)
optimizer_DA = optim.Adam(net.D_A.parameters(), lr=0.0002, betas=(
    0.5, 0.999), eps=1e-08, weight_decay=0)
optimizer_DB = optim.Adam(net.D_B.parameters(), lr=0.0002, betas=(
    0.5, 0.999), eps=1e-08, weight_decay=0)
optimizer_DA_all = optim.Adam(net.D_A_all.parameters(), lr=0.0002, betas=(
    0.5, 0.999), eps=1e-08, weight_decay=0)
optimizer_DB_all = optim.Adam(net.D_B_all.parameters(), lr=0.0002, betas=(
    0.5, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite = 0
g_running_loss = 0.0
d_running_loss = 0.0
ite_num4val = 0
save_frq = 5000

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(music_dataloader):
        ite = ite + 1
        ite_num4val = ite_num4val + 1
        real_a, real_b, real_mixed = data['bar_a'], data['bar_b'], data['bar_mixed']
        real_a = torch.FloatTensor(real_a)
        real_b = torch.FloatTensor(real_b)
        real_mixed = torch.FloatTensor(real_mixed)

        if torch.cuda.is_available():
            real_a.cuda()
            real_b.cuda()
            real_mixed.cuda()

        # zero the parameter gradients
        optimizer_GA2B.zero_grad()
        optimizer_GB2A.zero_grad()
        optimizer_DA.zero_grad()
        optimizer_DB.zero_grad()
        optimizer_DA_all.zero_grad()
        optimizer_DB_all.zero_grad()

        cycle_loss, DA_real, DB_real, DA_fake, DB_fake, DA_fake_sample, DB_fake_sample, DA_real_all, DB_real_all, DA_fake_all, DB_fake_all = net(
            real_a, real_b, real_mixed)

        # Generator loss
        g_A2B_loss = l1loss(DB_fake, torch.ones_like(DB_fake)) + cycle_loss
        g_B2A_loss = l1loss(DA_fake, torch.ones_like(DA_fake)) + cycle_loss
        g_loss = g_A2B_loss + g_B2A_loss - cycle_loss

        # Discriminator loss
        d_A_loss_real = l2loss(DA_real, torch.ones_like(DA_real))
        d_A_loss_fake = l2loss(
            DA_fake_sample, torch.zeros_like(DA_fake_sample))
        d_A_loss = (d_A_loss_real + d_A_loss_fake) / 2
        d_B_loss_real = l2loss(DB_real, torch.ones_like(DB_real))
        d_B_loss_fake = l2loss(
            DB_fake_sample, torch.zeros_like(DB_fake_sample))
        d_B_loss = (d_B_loss_real + d_B_loss_fake) / 2
        d_loss = d_A_loss + d_B_loss

        d_A_all_loss_real = l2loss(DA_real_all, torch.ones_like(DA_real_all))
        d_A_all_loss_fake = l2loss(DA_fake_all, torch.zeros_like(DA_fake_all))
        d_A_all_loss = (d_A_all_loss_real + d_A_all_loss_fake) / 2
        d_B_all_loss_real = l2loss(DB_real_all, torch.ones_like(DB_real_all))
        d_B_all_loss_fake = l2loss(DB_fake_all, torch.zeros_like(DB_fake_all))
        d_B_all_loss = (d_B_all_loss_real + d_B_all_loss_fake) / 2
        d_all_loss = d_A_all_loss + d_B_all_loss
        D_loss = d_loss + gamma * d_all_loss

        g_A2B_loss.backward(retain_graph=True)
        g_B2A_loss.backward(retain_graph=True)

        d_A_loss.backward(retain_graph=True)
        d_B_loss.backward(retain_graph=True)

        d_A_all_loss.backward(retain_graph=True)
        d_B_all_loss.backward()

        optimizer_GA2B.step()
        optimizer_GB2A.step()
        optimizer_DA.step()
        optimizer_DB.step()
        optimizer_DA_all.step()
        optimizer_DB_all.step()

        g_running_loss += g_loss.data.item()
        d_running_loss += D_loss.data.item()

        del DA_real, DB_real, DA_fake, DB_fake, DA_fake_sample, DB_fake_sample, DA_real_all, DB_real_all, DA_fake_all, DB_fake_all
        del g_A2B_loss, g_B2A_loss, g_loss, d_A_loss_real, d_A_loss_fake, d_A_loss, d_B_loss_real, d_B_loss_fake, d_B_loss, d_loss, d_A_all_loss_real, d_A_all_loss_fake, d_A_all_loss, d_B_all_loss_real, d_B_all_loss_fake, d_B_all_loss, d_all_loss, D_loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] Generator : %3f, Discriminator : %3f " % (
            epoch +
            1, epoch_num, (i + 1) *
            batch_size_train, train_num, ite, g_running_loss / ite_num4val,
            d_running_loss / ite_num4val))

        if ite % save_frq == 0:
            torch.save(net, model_dir + model_name + "_itr_%d_G_%3f_D_%3f.pth" % (
                ite, g_running_loss / ite_num4val, d_running_loss / ite_num4val))
            g_running_loss = 0.0
            d_running_loss = 0.0
            net.train()
            ite_num4val = 0
