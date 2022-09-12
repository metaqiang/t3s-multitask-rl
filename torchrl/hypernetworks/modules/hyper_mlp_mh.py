import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperMLPMH(torch.nn.Module):
    def __init__(self, hyper_input_dim=10, mlp_input_dim=19, mlp_output_dim=8):
        super().__init__()

        self.hyper_input_dim = hyper_input_dim
        self.mlp_input_dim = mlp_input_dim
        self.mlp_output_dim = mlp_output_dim

        self.fc_1 = nn.Linear(mlp_input_dim + hyper_input_dim, 400)
        self.fc_2 = nn.Linear(400, 400)
        self.fc_3 = nn.Linear(400, 400)
        # self.fc_4 = nn.Linear(400, 400)

        # self.head_W_1 = nn.Linear(hyper_input_dim, 400 * 400)
        # self.head_b_1 = nn.Linear(hyper_input_dim, 400)

        self.head_W_2 = nn.Linear(hyper_input_dim, 400 * mlp_output_dim)
        self.head_b_2 = nn.Linear(hyper_input_dim, mlp_output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, -0.05, 0.05)
                torch.nn.init.zeros_(m.bias)

                # m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()

    def forward(self, hyper_x, mlp_x):
        base_shape = x.shape[:-1]

        x = x.reshape(-1, x.shape[-1])
        mlp_x = mlp_x.reshape(-1, mlp_x.shape[-1])

        out = F.relu(self.fc_1(mlp_x))
        out = F.relu(self.fc_2(out))
        out = F.relu(self.fc_3(out))
        # out = F.relu(self.fc_4(out))
        out = out.reshape(out.shape[0], 1, out.shape[1])

        # W_1 = self.head_W_1(x).reshape(out.shape[0], 400, 400)
        # b_1 = self.head_b_1(x)

        W_2 = self.head_W_2(x).reshape(out.shape[0], 400, self.mlp_output_dim)
        b_2 = self.head_b_2(x)

        # out = F.relu(torch.bmm(out, W_1).reshape(out.shape[0], 400) + b_1)
        # out = out.reshape(out.shape[0], 1, out.shape[1])
        out = torch.bmm(out, W_2).reshape(
            out.shape[0], self.mlp_output_dim) + b_2

        return out.reshape(base_shape + torch.Size([-1]))
