import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, stride=1, dims=2):
        super(Residual, self).__init__()
        conv_func = nn.Conv2d if dims == 2 else nn.Conv1d
        self._block = nn.Sequential(
            nn.ReLU(True),
            conv_func(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(True),
            conv_func(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, dims=2):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens, dims=dims)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Conv1DResidualModel(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(Conv1DResidualModel, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Define convolutional layers
        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
             for _ in range(num_layers)])
        self.relu = nn.ReLU()

        self.conv_end = nn.Conv1d(in_channels=hidden_size, out_channels=output_size, kernel_size=3, padding=1)

    def forward(self, x):
        # Apply initial convolutional layer
        residual = x
        out = self.conv_layers[0](x)
        out = self.relu(out)

        # Apply residual blocks
        for i in range(1, self.num_layers):
            residual = out
            out = self.conv_layers[i](out)
            out = self.relu(out + residual)  # Residual connection

            # Apply time stride of 2 in the 3rd layer
            if i == 2:
                out = F.max_pool1d(out, kernel_size=2, stride=2)

        return self.conv_end(out)

