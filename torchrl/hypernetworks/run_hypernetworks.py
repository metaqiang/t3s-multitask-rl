# UPDATE: 2022/4/28
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from modules.soft_module import SoftModule
import time
import os
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast as autocast, GradScaler
from utils import *
import modules
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())
set_single_cpu()


class SADataset(Dataset):
    def __init__(self, n=10000):
        self.s_0 = torch.randn((n, 9))
        self.a_0 = torch.sum(self.s_0, axis=-1, keepdim=True).repeat(1, 8)
        self.task_em_0 = torch.zeros((n, 10))
        self.task_em_0[..., 0] = 1

        self.s_1 = torch.randn((n, 9))
        self.a_1 = torch.mean(self.s_1, axis=-1, keepdim=True).repeat(1, 8)
        self.task_em_1 = torch.zeros((n, 10))
        self.task_em_1[..., 1] = 1

        self.s_0 = torch.randn((n, 9))
        self.a_0 = torch.sum(self.s_0, axis=-1, keepdim=True).repeat(1, 8) / 2
        self.task_em_0 = torch.zeros((n, 10))
        self.task_em_0[..., 2] = 1

        self.s_1 = torch.randn((n, 9))
        self.a_1 = torch.mean(self.s_1, axis=-1, keepdim=True).repeat(1, 8) / 2
        self.task_em_1 = torch.zeros((n, 10))
        self.task_em_1[..., 3] = 1

        self.s_0 = torch.randn((n, 9))
        self.a_0 = torch.sum(self.s_0, axis=-1, keepdim=True).repeat(1, 8) * 2
        self.task_em_0 = torch.zeros((n, 10))
        self.task_em_0[..., 4] = 1

        self.s_1 = torch.randn((n, 9))
        self.a_1 = torch.mean(self.s_1, axis=-1, keepdim=True).repeat(1, 8) * 2
        self.task_em_1 = torch.zeros((n, 10))
        self.task_em_1[..., 5] = 1

        self.s_0 = torch.randn((n, 9))
        self.a_0 = torch.sum(self.s_0, axis=-1, keepdim=True).repeat(1, 8) / 3
        self.task_em_0 = torch.zeros((n, 10))
        self.task_em_0[..., 6] = 1

        self.s_1 = torch.randn((n, 9))
        self.a_1 = torch.mean(self.s_1, axis=-1, keepdim=True).repeat(1, 8) / 3
        self.task_em_1 = torch.zeros((n, 10))
        self.task_em_1[..., 7] = 1

        self.s_0 = torch.randn((n, 9))
        self.a_0 = torch.sum(self.s_0, axis=-1, keepdim=True).repeat(1, 8) * 3
        self.task_em_0 = torch.zeros((n, 10))
        self.task_em_0[..., 8] = 1

        self.s_1 = torch.randn((n, 9))
        self.a_1 = torch.mean(self.s_1, axis=-1, keepdim=True).repeat(1, 8) * 3
        self.task_em_1 = torch.zeros((n, 10))
        self.task_em_1[..., 9] = 1

        self.s = torch.cat([self.s_0, self.s_1], 0)
        self.a = torch.cat([self.a_0, self.a_1], 0)
        self.task_em = torch.cat([self.task_em_0, self.task_em_1], 0)

    def __getitem__(self, index):
        return self.s[index, ...], self.a[index, ...], self.task_em[index, ...]

    def __len__(self):
        return self.s.shape[0]


def get_all_param_num():
    models = [modules.MLP(), modules.HyperMLP(), modules.HyperMLPMH(), modules.MMoE(),
              modules.SoftModule(), modules.HyperMTANPro()]
    data = []

    for i, model in enumerate(models):
        print(i, model)
        number = get_parameter_number(model)
        data.append(number['Trainable'])
        print(number)
        print('*' * 50)

    plt.bar(range(len(data)), data, tick_label=[
            'MT MLP', 'Single MLP', 'Multi Head MLP', 'MMoE', 'Soft Module', 'Ours'])
    plt.show()


def run_overfit_demo():
    time_stamp = time.strftime("%F-%H-%M-%S")
    log_dir = os.path.join('output', args.name, time_stamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # model = modules.MLP(mlp_input_dim=19).to(device)
    # model = modules.HyperMLP().to(device)
    # model = modules.HyperMLPMH().to(device)
    # model = modules.MMoE().to(device)
    # model = modules.SoftModule().to(device)
    model = modules.HyperMTANPro().to(device)

    print(get_parameter_number(model))

    sa_dataset = SADataset()

    train_loader = DataLoader(
        dataset=sa_dataset, batch_size=1280, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # scaler = GradScaler()

    count = 0
    for epoch in range(100):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            s, a, task_em = data

            s = Variable(s).to(device)
            a = Variable(a).to(device)
            task_em = Variable(task_em).to(device)

            # with autocast():
            outputs = model(task_em, s)
            # outputs = model(task_em, torch.cat([s, task_em], 1))
            # outputs = model(torch.cat([s, task_em], 1))
            labels = a
            loss = criterion(outputs, labels).mean()

            loss.backward()
            # scaler.scale(loss).backward()

            # nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()
            # scaler.step(optimizer)
            # scaler.update()

            writer.add_scalar('loss', loss.mean().item(), count)
            count += 1
            if i % 10 == 0:
                print('epoch:{}, loss: {}'.format(epoch, loss.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run flags")
    parser.add_argument('--name', default='dev')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    apply_seed(args.seed)

    # get_all_param_num()
    run_overfit_demo()
