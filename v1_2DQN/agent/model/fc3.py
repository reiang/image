#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dim, output_dim,chkpt_dir='./chkpt'):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = self.checkpoint_dir + '/models'

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)

        self.flatten = nn.Flatten()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions
    
    def save_checkpoint(self, save_file):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        print('... saving checkpoint ...')
        T.save(self.state_dict(), save_file)

    def load_checkpoint(self, load_file):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(load_file))

