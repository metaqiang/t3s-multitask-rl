import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperMTANPro(torch.nn.Module):
    def __init__(self, hyper_input_dim=10, mlp_input_dim=9, mlp_output_dim=8, attention_block=1,
                 global_linear_hidden_size=256, global_linear_hidden_block=0,
                 mask_liner_hidden_sise=128, mask_liner_hidden_block=2, extractor_liner_hidden_size=64):
        super().__init__()

        self.hyper_input_dim = hyper_input_dim
        self.mlp_input_dim = mlp_input_dim
        self.mlp_output_dim = mlp_output_dim
        self.global_linear_hidden_size = global_linear_hidden_size
        self.global_linear_hidden_block = global_linear_hidden_block
        self.mask_liner_hidden_sise = mask_liner_hidden_sise
        self.mask_liner_hidden_block = mask_liner_hidden_block
        self.extractor_liner_hidden_size = extractor_liner_hidden_size
        self.attention_block = attention_block

        self.func = nn.ReLU(inplace=True)

        # part 1
        self.global_linears = nn.ModuleDict()

        for i in range(self.attention_block):
            self.global_linears[str(i)] = nn.ModuleList()
            if i == 0:
                self.global_linears[str(0)].append(
                    nn.Linear(self.mlp_input_dim, self.global_linear_hidden_size))
            else:
                self.global_linears[str(i)].append(
                    nn.Linear(self.global_linear_hidden_size, self.global_linear_hidden_size))

            for _ in range(self.global_linear_hidden_block):
                self.global_linears[str(i)].append(
                    nn.Linear(self.global_linear_hidden_size, self.global_linear_hidden_size))

            self.global_linears[str(i)].append(
                nn.Linear(self.global_linear_hidden_size, self.global_linear_hidden_size))

        # part 2
        self.mask_linears_w = nn.ModuleDict()
        self.mask_linears_b = nn.ModuleDict()

        for i in range(self.attention_block):
            self.mask_linears_w[str(i)] = nn.ModuleList()
            self.mask_linears_b[str(i)] = nn.ModuleList()

            if i == 0:
                self.mask_linears_w[str(i)].append(nn.Linear(
                    self.hyper_input_dim, self.global_linear_hidden_size * self.mask_liner_hidden_sise))
            else:
                self.mask_linears_w[str(i)].append(nn.Linear(self.hyper_input_dim, (
                    self.global_linear_hidden_size + self.extractor_liner_hidden_size) * self.mask_liner_hidden_sise))

            self.mask_linears_b[str(i)].append(
                nn.Linear(self.hyper_input_dim, self.mask_liner_hidden_sise))

            for _ in range(self.mask_liner_hidden_block):
                self.mask_linears_w[str(i)].append(
                    nn.Linear(self.hyper_input_dim, self.mask_liner_hidden_sise ** 2))
                self.mask_linears_b[str(i)].append(
                    nn.Linear(self.hyper_input_dim, self.mask_liner_hidden_sise))

            self.mask_linears_w[str(i)].append(nn.Linear(
                self.hyper_input_dim, self.mask_liner_hidden_sise * self.global_linear_hidden_size))
            self.mask_linears_b[str(i)].append(
                nn.Linear(self.hyper_input_dim, self.global_linear_hidden_size))

        # part 3
        # shared
        # self.extractor_shared_networks = nn.ModuleDict()
        # for i in range(self.attention_block):
        #     self.extractor_shared_networks[str(i)] = nn.ModuleList()
        #     self.extractor_shared_networks[str(i)].append(nn.Linear(self.global_linear_hidden_size, self.extractor_liner_hidden_size))

        # non-shared
        self.extractor_linears_w = nn.ModuleDict()
        self.extractor_linears_b = nn.ModuleDict()

        for i in range(self.attention_block):
            self.extractor_linears_w[str(i)] = nn.ModuleList()
            self.extractor_linears_b[str(i)] = nn.ModuleList()

            self.extractor_linears_w[str(i)].append(nn.Linear(
                self.hyper_input_dim, self.global_linear_hidden_size * self.extractor_liner_hidden_size))
            self.extractor_linears_b[str(i)].append(
                nn.Linear(self.hyper_input_dim, self.extractor_liner_hidden_size))

        # part 4
        self.last_linears_w = nn.Linear(
            self.hyper_input_dim, self.extractor_liner_hidden_size * self.mlp_output_dim)
        self.last_linears_b = nn.Linear(
            self.hyper_input_dim, self.mlp_output_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, -0.05, 0.05)
                torch.nn.init.zeros_(m.bias)

    def forward(self, hyper_x, mlp_x):
        base_shape = hyper_x.shape[:-1]
        hyper_x = hyper_x.reshape(-1, hyper_x.shape[-1])
        mlp_x = mlp_x.reshape(-1, mlp_x.shape[-1])

        batch_size = hyper_x.shape[0]
        global_results = []

        for i in range(self.attention_block):
            for j, liner in enumerate(self.global_linears[str(i)]):
                if i == 0 and j == 0:
                    global_results.append(self.func(liner(mlp_x)))
                else:
                    global_results.append(self.func(liner(global_results[-1])))

        for i in range(self.attention_block):
            # part 1
            if i == 0:
                out = global_results[0].reshape(
                    batch_size, 1, self.global_linear_hidden_size)
            else:
                out = torch.cat([out, global_results[(self.global_linear_hidden_block+2) * (i)]], dim=1).reshape(
                    batch_size, 1, self.global_linear_hidden_size + self.extractor_liner_hidden_size)

            if i == 0:
                w = self.mask_linears_w[str(i)][0](hyper_x).reshape(
                    batch_size, self.global_linear_hidden_size, self.mask_liner_hidden_sise)
            else:
                w = self.mask_linears_w[str(i)][0](hyper_x).reshape(
                    batch_size, self.global_linear_hidden_size + self.extractor_liner_hidden_size, self.mask_liner_hidden_sise)

            b = self.mask_linears_b[str(i)][0](hyper_x).reshape(
                batch_size, 1, self.mask_liner_hidden_sise)
            out = self.func(torch.bmm(out, w) + b)

            # part 2
            for j, (linear_w, linear_b) in enumerate(zip(self.mask_linears_w[str(i)], self.mask_linears_b[str(i)])):
                if j != 0 and j != len(self.mask_linears_w[str(i)])-1:
                    w = linear_w(hyper_x).reshape(
                        batch_size, self.mask_liner_hidden_sise, self.mask_liner_hidden_sise)
                    b = linear_b(hyper_x).reshape(
                        batch_size, 1, self.mask_liner_hidden_sise)
                    out = self.func(torch.bmm(out, w) + b)

            w = self.mask_linears_w[str(i)][-1](hyper_x).reshape(
                batch_size, self.mask_liner_hidden_sise, self.global_linear_hidden_size)
            b = self.mask_linears_b[str(
                i)][-1](hyper_x).reshape(batch_size, 1, self.global_linear_hidden_size)

            mask = torch.sigmoid(
                torch.bmm(out, w) + b).reshape(batch_size, self.global_linear_hidden_size)
            out = (mask * global_results[(self.global_linear_hidden_block+2) * (
                i+1) - 1]).reshape(batch_size, 1, self.global_linear_hidden_size)

            # shared
            # out = out.reshape(batch_size, -1)
            # out = self.func(self.extractor_shared_networks[str(i)][0](out))

            # non-shared
            w = self.extractor_linears_w[str(i)][-1](hyper_x).reshape(
                batch_size, self.global_linear_hidden_size, self.extractor_liner_hidden_size)
            b = self.extractor_linears_b[str(
                i)][-1](hyper_x).reshape(batch_size, 1, self.extractor_liner_hidden_size)
            out = self.func(torch.bmm(out, w) + b).reshape(batch_size, -1)

        out = out.reshape(batch_size, 1, self.extractor_liner_hidden_size)

        w = self.last_linears_w(hyper_x).reshape(
            batch_size, self.extractor_liner_hidden_size, self.mlp_output_dim)
        b = self.last_linears_b(hyper_x).reshape(
            batch_size, 1, self.mlp_output_dim)

        out = (torch.bmm(out, w) + b).reshape(base_shape + torch.Size([-1]))

        return out
