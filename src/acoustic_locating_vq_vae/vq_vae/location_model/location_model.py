import torch
from torch import nn


class LocationModule(nn.Module):

    def __init__(self, encoder_output_dim: int, num_hiddens: int, output_dim: int):
        super(LocationModule, self).__init__()
        self.encoder_output_dim = encoder_output_dim
        self.fc_1 = nn.Linear(encoder_output_dim * num_hiddens, 1024)
        self.relu1 = nn.ReLU()
        self.fc_2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.fc_3 = nn.Linear(512, 512)
        self.relu3 = nn.ReLU()
        self.fc_4 = nn.Linear(512, 64)
        self.relu4 = nn.ReLU()
        self.fc_5 = nn.Linear(64, output_dim)

    def forward(self, x):
         z = self.fc_1(torch.flatten(x, start_dim=1))
         z = self.relu1(z)
         z = self.fc_2(z)
         z = self.relu2(z)
         z = self.fc_3(z)
         z = self.relu3(z)
         z = self.fc_4(z)
         z = self.relu4(z)
         return self.fc_5(z)
