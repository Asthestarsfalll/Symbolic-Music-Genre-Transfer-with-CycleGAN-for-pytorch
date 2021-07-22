import os
import time
import argparse
from tensorboardX import SummaryWriter
from torch import optim
import torch
from torch.utils.data import DataLoader
from model import CycleGAN
from dataloader import MusicDataset
import logging
from logger import setup_logger


logger = logging.getLogger()

def make_parses():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument(
        '--epoch',
        default=30,
        type=int
    )
    parser.add_argument(
        '--start-epoch',
        default=0,
        type=int
    )
    parser.add_argument(
        '--batch-size',
        default=2,
        type=int
    )
    parser.add_argument(
        '--model-name',
        default='CP',
        type=str,
        help='Optional: CP, JC, JP'
    )
    parser.add_argument(
        '--resume',
        default='',
        type=str
    )
    parser.add_argument(
        '--gamma',
        default=1,
        type=float
    )
    parser.add_argument(
        '--sigma',
        default=0.01,
        type=float
    )
    parser.add_argument(
        '--lamb',
        default=10,
        type=float
    )
    parser.add_argument(
        '--sample_size',
        default=50,
        type=int
    )
    parser.add_argument(
        '--save-frq',
        default=1000,
        type=int
    )
    parser.add_argument(
        '--log_frq',
        default=200,
        type=int
    )
    parser.add_argument(
        '--lr',
        default=0.002,
        type=float
    )
    parser.add_argument(
        '--wd', '--weight-decay',
        default=0,
        type=float
    )
    parser.add_argument(
        '--data-mode',
        default='full',
        type=str,
        help='Optional: full, '
    )
    return parser.parse_args()


def train():
    # ------- set the directory of training dataset --------
    args = make_parses()
    model_name = args.model_name  # JC CP JP.
    writer = SummaryWriter(comment=model_name+str(time.time()))

    data_dir = os.path.join(os.getcwd(), 'data' + os.sep)


    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    epoch_num = args.epoch
    batch_size_train = args.batch_size
    gamma = args.gamma
    save_frq = args.save_frq
    log_frq = args.log_frq

    music_dataset = MusicDataset(data_dir, train_mode=model_name, data_mode=args.data_mode, is_train='train')
    train_num = len(music_dataset)
    logger.info("train data contains 2*{} items".format(train_num))
    music_dataloader = DataLoader(
        music_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)
    setup_logger(model_dir)

    logger.info("load model with mode {}".format(model_name))

    # ------- define model --------
    model = CycleGAN(sigma=args.sigma, sample_size=args.sample_size, lamb=args.lamb, mode='train')
    if args.resume:
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        if 'model_name' in checkpoint.keys():
            assert model_name == checkpoint['model_name']
        model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        model.cuda()

    logger.info("---define optimizer---")
    lr = args.lr
    wd = args.wd
    optimizer_GA2B = optim.Adam(model.G_A2B.parameters(), lr=lr, betas=(
        0.5, 0.999), eps=1e-08, weight_decay=wd)
    optimizer_GB2A = optim.Adam(model.G_B2A.parameters(), lr=lr, betas=(
        0.5, 0.999), eps=1e-08, weight_decay=wd)
    optimizer_DA = optim.Adam(model.D_A.parameters(), lr=lr, betas=(
        0.5, 0.999), eps=1e-08, weight_decay=wd)
    optimizer_DB = optim.Adam(model.D_B.parameters(), lr=lr, betas=(
        0.5, 0.999), eps=1e-08, weight_decay=wd)
    optimizer_DA_all = optim.Adam(model.D_A_all.parameters(), lr=lr, betas=(
        0.5, 0.999), eps=1e-08, weight_decay=wd)
    optimizer_DB_all = optim.Adam(model.D_B_all.parameters(), lr=lr, betas=(
        0.5, 0.999), eps=1e-08, weight_decay=wd)

    # ------- training process --------
    logger.info("---start training---")
    ite = 0
    g_running_loss = 0.0
    d_running_loss = 0.0
    ite_num4val = 0

    start = time.time()
    for epoch in range(args.start_epoch, epoch_num):
        model.train()
        for i, data in enumerate(music_dataloader):
            ite = ite + 1
            ite_num4val = ite_num4val + 1
            real_a, real_b, real_mixed = data['bar_a'], data['bar_b'], data['bar_mixed']
            real_a = torch.FloatTensor(real_a)
            real_b = torch.FloatTensor(real_b)
            real_mixed = torch.FloatTensor(real_mixed)

            if torch.cuda.is_available():
                real_a = real_a.cuda()
                real_b = real_b.cuda()
                real_mixed = real_mixed.cuda()
            # zero the parameter gradients
            optimizer_GA2B.zero_grad()
            optimizer_GB2A.zero_grad()
            optimizer_DA.zero_grad()
            optimizer_DB.zero_grad()
            optimizer_DA_all.zero_grad()
            optimizer_DB_all.zero_grad()

            cycle_loss, g_A2B_loss, g_B2A_loss, d_A_loss, d_B_loss, \
            d_A_all_loss, d_B_all_loss = model(real_a, real_b, real_mixed)
            # Generator loss
            g_loss = g_A2B_loss + g_B2A_loss - cycle_loss

            # Discriminator loss
            d_loss = d_A_loss + d_B_loss

            d_all_loss = d_A_all_loss + d_B_all_loss
            D_loss = d_loss + gamma * d_all_loss

            g_A2B_loss.backward(retain_graph=True)
            g_B2A_loss.backward(retain_graph=True)

            d_A_loss.backward(retain_graph=True)
            d_B_loss.backward(retain_graph=True)

            d_A_all_loss.backward(retain_graph=True)
            d_B_all_loss.backward(retain_graph=True)

            optimizer_GA2B.step()
            optimizer_GB2A.step()
            optimizer_DA.step()
            optimizer_DB.step()
            optimizer_DA_all.step()
            optimizer_DB_all.step()

            g_running_loss += g_loss.data.item()
            d_running_loss += D_loss.data.item()

            writer.add_scalar('cycle_loss', cycle_loss, global_step=ite)
            writer.add_scalar('g_A2B_loss', g_A2B_loss, global_step=ite)
            writer.add_scalar('g_B2A_loss', g_B2A_loss, global_step=ite)
            writer.add_scalar('d_A_loss', d_A_loss, global_step=ite)
            writer.add_scalar('d_B_loss', d_B_loss, global_step=ite)
            writer.add_scalar('d_A_all_loss', g_loss, global_step=ite)
            writer.add_scalar('d_B_all_loss', g_loss, global_step=ite)
            writer.add_scalar('d_all_loss', g_loss, global_step=ite)
            writer.add_scalar('D_loss', D_loss, global_step=ite)

            del g_A2B_loss, g_B2A_loss, g_loss, d_A_loss, d_B_loss, d_loss, \
                d_A_all_loss, d_B_all_loss, d_all_loss, D_loss
            if i % log_frq == 0:
                end = time.time()
                logger.info("[epoch: %3d/%3d, "
                            "batch: %5d/%5d, "
                            "ite: %d, "
                            "time: %3f] "
                            "g_loss : %3f, "
                            "d_loss : %3f " % (
                    epoch + 1, epoch_num,
                    (i) * batch_size_train, train_num,
                    ite,
                    end - start,
                    g_running_loss / ite_num4val,
                    d_running_loss / ite_num4val))
                start = end

            if ite % save_frq == 0:
                saved_model_name = model_dir + model_name + "_itr_%d_G_%3f_D_%3f.pth" % (
                    ite, g_running_loss / ite_num4val, d_running_loss / ite_num4val)
                torch.save({
                    'epoch': epoch,
                    'model_name': model_name,
                    'state_dict': model.state_dict()},
                    saved_model_name)
                logger.info("saved model {}".format(saved_model_name))
                g_running_loss = 0.0
                d_running_loss = 0.0
                model.train()
                ite_num4val = 0

if __name__ =="__main__":
    train()
