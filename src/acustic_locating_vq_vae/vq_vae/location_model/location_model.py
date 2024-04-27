from torch import nn


class LocationModule(nn.Module):

    def __init__(self, latent_dim: int, num_hiddens: int, output_dim: int):
        super(LocationModule, self).__init__()
        self.fc_1 = nn.Linear(latent_dim * num_hiddens, 100)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(100, output_dim)

    def forward(self, x):
         z = self.fc_1(x)
         z = self.relu(z)
         return self.fc_2(z)
