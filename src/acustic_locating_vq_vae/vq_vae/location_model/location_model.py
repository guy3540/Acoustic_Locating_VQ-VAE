from torch import nn


class LocationModule(nn.Module):

    def __init__(self, latent_dim: int, num_hiddens: int, output_dim: int):
        super(LocationModule, self).__init__()
        self.fc_1 = nn.Linear(latent_dim * num_hiddens, 1000)
        self.relu1 = nn.ReLU()
        self.fc_2 = nn.Linear(1000, 100)
        self.relu2 = nn.ReLU()
        self.fc_3 = nn.Linear(100, output_dim)

    def forward(self, x):
         z = self.fc_1(x)
         z = self.relu1(z)
         z = self.fc_2(z)
         z = self.relu2(z)
         return self.fc_3(z)
