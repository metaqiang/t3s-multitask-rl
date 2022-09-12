import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl.hypernetworks.modules.init as init


def null_activation(x):
    return x


class MLPBase(nn.Module):
    def __init__(self, input_shape, hidden_shapes, activation_func=F.relu, init_func=init.basic_init,
                 last_activation_func=None):
        super().__init__()

        self.activation_func = activation_func
        self.fcs = []
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = activation_func
        input_shape = np.prod(input_shape)

        self.output_shape = input_shape
        for i, next_shape in enumerate(hidden_shapes):
            fc = nn.Linear(input_shape, next_shape)
            init_func(fc)
            self.fcs.append(fc)
            # set attr for pytorch to track parameters( device )
            self.__setattr__("fc{}".format(i), fc)

            input_shape = next_shape
            self.output_shape = next_shape

    def forward(self, x):

        out = x
        for fc in self.fcs[:-1]:
            out = fc(out)
            out = self.activation_func(out)
        out = self.fcs[-1](out)
        out = self.last_activation_func(out)
        return out


class ModularGatedCascadeCondNet(nn.Module):
    def __init__(self, output_shape,
                 base_type, em_input_shape, input_shape,
                 em_hidden_shapes,
                 hidden_shapes,

                 num_layers, num_modules,

                 module_hidden,

                 gating_hidden, num_gating_layers,

                 # gated_hidden
                 add_bn=True,
                 pre_softmax=False,
                 cond_ob=True,
                 module_hidden_init_func=init.basic_init,
                 last_init_func=init.uniform_init,
                 activation_func=F.relu,
                 **kwargs):

        super().__init__()

        self.base = base_type(
            last_activation_func=null_activation,
            input_shape=input_shape,
            activation_func=activation_func,
            hidden_shapes=hidden_shapes,
            **kwargs)
        self.em_base = base_type(
            last_activation_func=null_activation,
            input_shape=em_input_shape,
            activation_func=activation_func,
            hidden_shapes=em_hidden_shapes,
            **kwargs)

        self.activation_func = activation_func

        module_input_shape = self.base.output_shape
        self.layer_modules = []

        self.num_layers = num_layers
        self.num_modules = num_modules

        for i in range(num_layers):
            layer_module = []
            for j in range(num_modules):
                fc = nn.Linear(module_input_shape, module_hidden)
                module_hidden_init_func(fc)
                if add_bn:
                    module = nn.Sequential(
                        nn.BatchNorm1d(module_input_shape),
                        fc,
                        nn.BatchNorm1d(module_hidden)
                    )
                else:
                    module = fc

                layer_module.append(module)
                self.__setattr__("module_{}_{}".format(i, j), module)

            module_input_shape = module_hidden
            self.layer_modules.append(layer_module)

        self.last = nn.Linear(module_input_shape, output_shape)
        last_init_func(self.last)

        assert self.em_base.output_shape == self.base.output_shape, \
            "embedding should has the same dimension with base output for gated"
        gating_input_shape = self.em_base.output_shape
        self.gating_fcs = []
        for i in range(num_gating_layers):
            gating_fc = nn.Linear(gating_input_shape, gating_hidden)
            module_hidden_init_func(gating_fc)
            self.gating_fcs.append(gating_fc)
            self.__setattr__("gating_fc_{}".format(i), gating_fc)
            gating_input_shape = gating_hidden

        self.gating_weight_fcs = []
        self.gating_weight_cond_fcs = []

        self.gating_weight_fc_0 = nn.Linear(gating_input_shape,
                                            num_modules * num_modules)
        last_init_func(self.gating_weight_fc_0)
        # self.gating_weight_fcs.append(self.gating_weight_fc_0)

        for layer_idx in range(num_layers - 2):
            gating_weight_cond_fc = nn.Linear((layer_idx + 1) *
                                              num_modules * num_modules,
                                              gating_input_shape)
            module_hidden_init_func(gating_weight_cond_fc)
            self.__setattr__("gating_weight_cond_fc_{}".format(layer_idx + 1),
                             gating_weight_cond_fc)
            self.gating_weight_cond_fcs.append(gating_weight_cond_fc)

            gating_weight_fc = nn.Linear(gating_input_shape,
                                         num_modules * num_modules)
            last_init_func(gating_weight_fc)
            self.__setattr__("gating_weight_fc_{}".format(layer_idx + 1),
                             gating_weight_fc)
            self.gating_weight_fcs.append(gating_weight_fc)

        self.gating_weight_cond_last = nn.Linear((num_layers - 1) *
                                                 num_modules * num_modules,
                                                 gating_input_shape)
        module_hidden_init_func(self.gating_weight_cond_last)

        self.gating_weight_last = nn.Linear(gating_input_shape, num_modules)
        last_init_func(self.gating_weight_last)

        self.pre_softmax = pre_softmax
        self.cond_ob = cond_ob

    def forward(self, x, embedding_input, return_weights=False):
        # Return weights for visualization
        out = self.base(x)
        embedding = self.em_base(embedding_input)

        if self.cond_ob:
            embedding = embedding * out

        out = self.activation_func(out)

        if len(self.gating_fcs) > 0:
            embedding = self.activation_func(embedding)
            for fc in self.gating_fcs[:-1]:
                embedding = fc(embedding)
                embedding = self.activation_func(embedding)
            embedding = self.gating_fcs[-1](embedding)

        base_shape = embedding.shape[:-1]

        weights = []
        flatten_weights = []

        raw_weight = self.gating_weight_fc_0(
            self.activation_func(embedding))  # torch.Size([128, 10, 4])

        weight_shape = base_shape + torch.Size([self.num_modules,
                                                self.num_modules])
        flatten_shape = base_shape + torch.Size([self.num_modules *
                                                 self.num_modules])

        raw_weight = raw_weight.view(weight_shape)

        # torch.Size([128, 10, 2, 2])
        softmax_weight = F.softmax(raw_weight, dim=-1)
        weights.append(softmax_weight)
        if self.pre_softmax:
            flatten_weights.append(raw_weight.view(flatten_shape))
        else:
            flatten_weights.append(softmax_weight.view(flatten_shape))

        for gating_weight_fc, gating_weight_cond_fc in zip(self.gating_weight_fcs, self.gating_weight_cond_fcs):
            cond = torch.cat(flatten_weights, dim=-1)
            if self.pre_softmax:
                cond = self.activation_func(cond)
            cond = gating_weight_cond_fc(cond)
            cond = cond * embedding
            cond = self.activation_func(cond)

            raw_weight = gating_weight_fc(cond)
            raw_weight = raw_weight.view(weight_shape)
            softmax_weight = F.softmax(raw_weight, dim=-1)
            weights.append(softmax_weight)
            if self.pre_softmax:
                flatten_weights.append(raw_weight.view(flatten_shape))
            else:
                flatten_weights.append(softmax_weight.view(flatten_shape))

        cond = torch.cat(flatten_weights, dim=-1)
        if self.pre_softmax:
            cond = self.activation_func(cond)
        cond = self.gating_weight_cond_last(cond)
        cond = cond * embedding
        cond = self.activation_func(cond)

        raw_last_weight = self.gating_weight_last(cond)
        last_weight = F.softmax(raw_last_weight, dim=-1)

        module_outputs = [(layer_module(out)).unsqueeze(-2)
                          for layer_module in self.layer_modules[0]]

        # torch.Size([128, 10, 2, 256])
        module_outputs = torch.cat(module_outputs, dim=-2)

        # [TODO] Optimize using 1 * 1 convolution.

        for i in range(self.num_layers - 1):
            new_module_outputs = []
            for j, layer_module in enumerate(self.layer_modules[i + 1]):
                module_input = (module_outputs *
                                weights[i][..., j, :].unsqueeze(-1)).sum(dim=-2)

                module_input = self.activation_func(module_input)
                new_module_outputs.append((
                    layer_module(module_input)
                ).unsqueeze(-2))

            module_outputs = torch.cat(new_module_outputs, dim=-2)

        out = (module_outputs * last_weight.unsqueeze(-1)).sum(-2)
        out = self.activation_func(out)
        out = self.last(out)

        if return_weights:
            return out, weights, last_weight
        return out


class FlattenModularGatedCascadeCondNet(ModularGatedCascadeCondNet):
    def forward(self, input, embedding_input, return_weights=False):
        return super().forward(input, embedding_input, return_weights=return_weights)


class SoftModule(torch.nn.Module):
    def __init__(self, hyper_input_dim=10, mlp_input_dim=9, mlp_output_dim=8):
        super().__init__()
        params = {'hidden_shapes': [400, 400], 'em_hidden_shapes': [400], 'num_layers': 2, 'num_modules': 2,
                  'module_hidden': 256, 'num_gating_layers': 2, 'gating_hidden': 256, 'add_bn': False,
                  'pre_softmax': False}
        params['base_type'] = MLPBase

        self.qf1 = FlattenModularGatedCascadeCondNet(
            input_shape=mlp_input_dim,
            em_input_shape=hyper_input_dim,
            output_shape=mlp_output_dim,
            **params)

    def forward(self, x, mlp_x):
        return self.qf1(input=mlp_x, embedding_input=x)


if __name__ == '__main__':
    soft_module = SoftModule()
    print(soft_module(torch.randn((128, 10, 10)), torch.randn((128, 10, 9))).shape)
