import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperMLP(torch.nn.Module):
    def __init__(self, hyper_input_dim=10, mlp_input_dim=9, mlp_output_dim=8,
                 hyper_hidden_dim=40, output_hidden_dim=400, hyper_layer=1, output_layer=3):
        super().__init__()
        self.hyper_input_dim = hyper_input_dim
        self.mlp_input_dim = mlp_input_dim
        self.mlp_output_dim = mlp_output_dim
        self.hyper_layer = hyper_layer
        self.output_layer = output_layer
        self.output_hidden_dim = output_hidden_dim

        self.hyper_parameter_dict = nn.ModuleDict({k: nn.ModuleList() for k in [f"w{i}" for i in range(output_layer)] +
                                                   [f"b{i}" for i in range(output_layer)]})
        self.use_bias = True

        for i in range(output_layer):
            for j in range(hyper_layer):
                if hyper_layer == 1:
                    if i == 0:
                        w_to_append = nn.Linear(
                            hyper_input_dim, mlp_input_dim * output_hidden_dim, bias=self.use_bias)
                        b_to_append = nn.Linear(
                            hyper_input_dim, output_hidden_dim, bias=self.use_bias)
                    elif i < output_layer - 1:
                        w_to_append = nn.Linear(
                            hyper_input_dim, output_hidden_dim * output_hidden_dim, bias=self.use_bias)
                        b_to_append = nn.Linear(
                            hyper_input_dim, output_hidden_dim, bias=self.use_bias)
                    else:
                        w_to_append = nn.Linear(
                            hyper_input_dim, output_hidden_dim * mlp_output_dim, bias=self.use_bias)
                        b_to_append = nn.Linear(
                            hyper_input_dim, mlp_output_dim, bias=self.use_bias)
                else:
                    if j == 0:
                        w_to_append = nn.Linear(
                            hyper_input_dim, hyper_hidden_dim, bias=self.use_bias)
                        b_to_append = nn.Linear(
                            hyper_input_dim, hyper_hidden_dim, bias=self.use_bias)
                    elif j < hyper_layer - 1:
                        w_to_append = nn.Linear(
                            hyper_hidden_dim, hyper_hidden_dim, bias=self.use_bias)
                        b_to_append = nn.Linear(
                            hyper_hidden_dim, hyper_hidden_dim, bias=self.use_bias)
                    else:
                        if i == 0:
                            w_to_append = nn.Linear(hyper_hidden_dim, mlp_input_dim * output_hidden_dim,
                                                    bias=self.use_bias)
                            b_to_append = nn.Linear(
                                hyper_hidden_dim, output_hidden_dim, bias=self.use_bias)
                        elif i < output_layer - 1:
                            w_to_append = nn.Linear(hyper_hidden_dim, output_hidden_dim * output_hidden_dim,
                                                    bias=self.use_bias)
                            b_to_append = nn.Linear(
                                hyper_hidden_dim, output_hidden_dim, bias=self.use_bias)
                        else:
                            w_to_append = nn.Linear(hyper_hidden_dim, output_hidden_dim * mlp_output_dim,
                                                    bias=self.use_bias)
                            b_to_append = nn.Linear(
                                hyper_hidden_dim, mlp_output_dim, bias=self.use_bias)

                self.hyper_parameter_dict['w' + str(i)].append(w_to_append)
                self.hyper_parameter_dict['b' + str(i)].append(b_to_append)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, -0.05, 0.05)
                torch.nn.init.zeros_(m.bias)

    def forward(self, hyper_x, mlp_x):
        x_shape = hyper_x.shape
        hyper_x = hyper_x.reshape(-1, hyper_x.shape[-1])

        # x = self.fc_task_em(x.reshape(-1, x.shape[-1]))
        base_shape = hyper_x.shape[:-1]

        out = mlp_x.reshape(base_shape + torch.Size([1, -1]))

        for i in range(self.output_layer):
            w, b = hyper_x, hyper_x
            for j in range(self.hyper_layer):
                w = self.hyper_parameter_dict['w' + str(i)][j](w)
                b = self.hyper_parameter_dict['b' + str(i)][j](b)

                if j != self.hyper_layer - 1:
                    w = F.relu(w, inplace=True)
                    b = F.relu(b, inplace=True)

            if i == 0:
                w = w.reshape(
                    base_shape + torch.Size([self.mlp_input_dim, self.output_hidden_dim]))
                b = b.reshape(
                    base_shape + torch.Size([1, self.output_hidden_dim]))
            elif i < self.output_layer - 1:
                w = w.reshape(
                    base_shape + torch.Size([self.output_hidden_dim, self.output_hidden_dim]))
                b = b.reshape(
                    base_shape + torch.Size([1, self.output_hidden_dim]))
            else:
                w = w.reshape(
                    base_shape + torch.Size([self.output_hidden_dim, self.mlp_output_dim]))
                b = b.reshape(
                    base_shape + torch.Size([1, self.mlp_output_dim]))

            out = torch.bmm(out, w) + b
            if i != self.output_layer - 1:
                out = F.relu(out, inplace=True)

        if len(x_shape) == 2:
            out = out.reshape(base_shape + torch.Size([-1]))
        elif len(x_shape) == 3:
            out = out.reshape(x_shape[:2] + torch.Size([-1]))
        else:
            raise ValueError

        return out
