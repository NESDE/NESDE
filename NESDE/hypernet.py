import torch
import torch.nn as nn

class ConfNet(nn.Module):
    # implements a linear nn with adjustable weights, to be used by a HyperNet
    def __init__(self, input_dim, output_dim, bias=True):
        super(ConfNet, self).__init__()
        self.bias = bias
        self.par_shape = torch.Tensor([input_dim, output_dim])
        self.__par = torch.zeros([input_dim, output_dim])
        if bias:
            self.__bias = torch.zeros(output_dim)

    def set_pars(self, par):
        if self.bias:
            if torch.equal(torch.Tensor(list(par[0].shape)), self.par_shape) and (self.par_shape[1] == par[1].shape[0]):
                self.__par = par[0]
                self.__bias = par[1]
            else:
                raise ValueError("Passed wrong shape parameter!")
        else:
            if torch.equal(torch.Tensor(list(par.shape)), self.par_shape):
                self.__par = par
            else:
                raise ValueError("Passed wrong shape parameter!")

    def get_pars(self):
        if self.bias:
            return [self.__par, self.__bias]
        else:
            return self.__par

    def forward(self, input):
        if self.bias:
            return input @ self.__par + self.__bias
        else:
            return input @ self.__par

class SeqHyper(nn.Sequential):
    # implements a sequence on nn (similar to Sequential)
    # mod_list is a list, not *list !
    def __init__(self, mod_list):
        super(SeqHyper, self).__init__(*mod_list)
        i = 0
        self.key_list = []
        for key, m in self._modules.items():
            if isinstance(m, ConfNet):
                self.key_list.append(key)
            i += 1

    def get_pars(self):
        pars = []
        for key in self.key_list:
            pars.append(self._modules[key].get_pars())
        return pars

    def set_pars(self, pars):
        if len(pars) != len(self.key_list):
            raise ValueError("Wrong length of parameters list!")
        for i, key in enumerate(self.key_list):
            self._modules[key].set_pars(pars[i])

class HyperNet(nn.Module):
    # takes control over the weights of a given ConfNet
    # decouples the input of itself from the ConfNet input
    def __init__(self, net, input_dim, hidden=[64, 64], nonlinearity=nn.ReLU):
        super(HyperNet, self).__init__()
        pars = net.get_pars()
        output_dim = 0
        self.ids_list = []
        self.bias_list = []
        self.shape_list = []
        self.n_layers = 0
        self.net = net
        if isinstance(net, SeqHyper):
            for par in pars:
                self.n_layers += 1
                if isinstance(par, list):
                    par_size = par[0].shape[0] * par[0].shape[1]
                    bias_size = par[1].shape[0]
                    self.ids_list.append([[output_dim, output_dim + par_size], [output_dim + par_size, output_dim + par_size + bias_size]])
                    self.bias_list.append(True)
                    self.shape_list.append(par[0].shape)
                    output_dim += par_size + bias_size
                else:
                    par_size = par.shape[0] * par.shape[1]
                    self.ids_list.append([output_dim, output_dim + par_size])
                    self.bias_list.append(False)
                    self.shape_list.append(par.shape)
                    output_dim += par_size

        elif isinstance(net, ConfNet):
            self.n_layers += 1
            if isinstance(pars, list):
                par_size = pars[0].shape[0] * pars[0].shape[1]
                bias_size = pars[1].shape[0]
                self.ids_list.append(
                    [[output_dim, output_dim + par_size], [output_dim + par_size, output_dim + par_size + bias_size]])
                self.bias_list.append(True)
                self.shape_list.append(pars[0].shape)
                output_dim += par_size + bias_size
            else:
                par_size = pars.shape[0] * pars.shape[1]
                self.ids_list.append([output_dim, output_dim + par_size])
                self.bias_list.append(False)
                self.shape_list.append(pars.shape)
                output_dim += par_size
        else:
            raise ValueError("net variable is of wrong type!")

        layer_sizes = [input_dim] + hidden
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i+1]), nonlinearity()])
        layers.append(nn.Linear(layer_sizes[-1], output_dim))
        self.hnet = nn.Sequential(*layers)

    def forward(self, input):
        out = self.hnet(input).view(-1)
        par_list = []
        for i in range(self.n_layers):
            if self.bias_list[i]:
                par = out[self.ids_list[i][0][0]:self.ids_list[i][0][1]]
                bias = out[self.ids_list[i][1][0]:self.ids_list[i][1][1]]
                par_list.append([par.view(self.shape_list[i]), bias])
            else:
                par = out[self.ids_list[i][0]:self.ids_list[i][1]]
                par_list.append(par.view(self.shape_list[i]))
        if self.n_layers == 1:
            par_list = par_list[0]
        self.net.set_pars(par_list)
        return
