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


class MMoE(torch.nn.Module):
    def __init__(self, mlp_input_dim=9, hyper_input_dim=10, mlp_output_dim=8, output_hidden_dim=400, export_num=3):
        super().__init__()

        self.expert_module_list = nn.ModuleList()
        for _ in range(export_num):
            self.expert_module_list.append(MLP(
                mlp_input_dim=mlp_input_dim, mlp_output_dim=mlp_output_dim, output_hidden_dim=output_hidden_dim))

        self.gate_param = nn.Parameter(
            torch.zeros(hyper_input_dim, export_num))

    def forward(self, hyper_x, mlp_x):
        base_shape = hyper_x.shape[:-1]
        hyper_x = hyper_x.reshape(-1, hyper_x.shape[-1])
        mlp_x = mlp_x.reshape(-1, mlp_x.shape[-1])

        non_zero = hyper_x.nonzero()[:, 1]
        gates = torch.softmax(self.gate_param[non_zero, :], axis=1)

        expert_output = []
        for expert_net in self.expert_module_list:
            expert_output.append(expert_net(mlp_x))

        output = sum([(torch.unsqueeze(gates[:, i], 1) * expert_output[i])
                     for i in range(len(expert_output))])
        return output.reshape(base_shape + torch.Size([-1]))
