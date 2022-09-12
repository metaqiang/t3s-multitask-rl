import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, mlp_input_dim=19, mlp_output_dim=8, output_hidden_dim=400):
        super().__init__()

        self.fc_1 = nn.Linear(mlp_input_dim, output_hidden_dim)
        self.fc_2 = nn.Linear(output_hidden_dim, output_hidden_dim)
        self.fc_3 = nn.Linear(output_hidden_dim, output_hidden_dim)
        self.fc_4 = nn.Linear(output_hidden_dim, mlp_output_dim)

    def forward(self, x):
        x = torch.relu(self.fc_1(x))
        x = torch.relu(self.fc_2(x))
        x = torch.relu(self.fc_3(x))
        x = self.fc_4(x)

        return x
