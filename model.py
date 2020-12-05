import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        output = self.act(self.bn1(self.fc1(x)))
        # output = self.act(self.fc1(x))
        output = self.act(self.bn2(self.fc2(output)))
        # output = self.act(self.fc2(output))
        output = self.act(self.bn3(self.fc3(output)))
        # output = self.act(self.fc3(output))
        return output


class GestureDetector(nn.Module):
    def __init__(self, frames, nf):
          super(GestureDetector, self).__init__()
          block = MLPBlock
          self.mlp = block(frames*21*3, nf)
          self.fc = nn.Linear(nf, 5) # our gesture dataset is consisted of 5 classes

    def forward(self, x):
        # print(x.view(x.size()[0], -1).shape)
        # print(torch.flatten(torch.flatten(x, start_dim=1), start_dim=0).shape)
        output = self.mlp(x.view(x.size()[0], -1))
        output = self.fc(output)
        return output


def get_hand_gesture(model, input):
    input = torch.FloatTensor(input).to('cuda')
    input = input.unsqueeze(0)
    # print(input.shape)
    output = model(input)
    # print(output.shape)
    pred = output.argmax(dim=1).data[0]
    # print(pred == 3)
    prob = output[0][pred].data
    return prob, pred

