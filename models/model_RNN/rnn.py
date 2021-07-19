import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn, optim

from dataset import AGCDataset, agc_dataloader
from utils import *


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bias, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bias=bias, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, state):
        output, state = self.lstm(x, state)
        result = self.fc(output)
        return result, state

    def init_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))


class RNN:
    def __init__(self, input_size=37, hidden_size=128, output_size=22, num_layers=4, bias=True, dropout=0.2,
                 lr=0.0001, batch_size=64):
        self.batch_size = batch_size
        self.lr = lr

        self.net = RNNModel(input_size, hidden_size, output_size, num_layers, bias, dropout)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)

    def optimize(self, x, y):
        criterion = nn.L1Loss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        state = self.net.init_state(self.batch_size)
        x = x.to(self.device)
        y = y.to(self.device)

        optimizer.zero_grad()
        y_pred, state = self.net(x, state)

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        return loss.detach().cpu().numpy()

    def predict(self, x):
        self.net.eval()

        state = self.net.init_state(1)

        x = x[:, np.newaxis]
        y_pred, state = self.net(x, state)
        y_pred = y_pred.detach().cpu().numpy().squeeze()

        return y_pred

    def save_state_dict(self, model_save_path='./model/trained_model/rnn.pth'):
        verify_output_path(model_save_path)
        torch.save(self.net.state_dict(), model_save_path)

    def load_state_dict(self, model_load_path='./model/trained_model/rnn.pth'):
        state_dict = torch.load(model_load_path)
        self.net.load_state_dict(state_dict)


def train(num_epoch=100):
    rnn = RNN()
    dataloader = agc_dataloader(rnn.batch_size)

    print('training...')

    for epoch in range(num_epoch):
        loss = 0.0
        for batch, (x, y) in enumerate(dataloader):
            cur_loss = rnn.optimize(x, y)
            loss += cur_loss / len(dataloader)
            print(f'epoch: {epoch:03}, bathc: {batch:03}, loss: {cur_loss}')

    rnn.save_state_dict()


def test(seed=42):
    np.random.seed(seed)

    rnn = RNN()
    rnn.load_state_dict()
    dataset = AGCDataset()
    xs, ys = dataset[np.random.randint(len(dataset))]
    y_pred = rnn.predict(xs)

    ys = dataset.denormalize_target(ys)
    y_pred = dataset.denormalize_target(y_pred)

    plt.figure(dpi=300)
    plt.plot(ys[:, -4], label='truth')
    plt.plot(y_pred[:, -4], label='predicted')
    plt.title('headFW')
    plt.legend()
    plt.show()
    plt.clf()

    plt.plot(ys[:, -3], label='truth')
    plt.plot(y_pred[:, -3], label='predicted')
    plt.title('shootDryMatterContent')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()
    test()

