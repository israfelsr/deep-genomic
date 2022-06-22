import torch
import torch.nn as nn
from torch.nn import functional as F

from modeling.genomic_model import GenomicModelConfig


def reparametrize(mu, logvar):
    std = torch.exp(logvar * 0.5)
    eps = torch.rand_like(std)
    return eps.mul(std).add_(mu)


class Encoder(nn.Module):

    def __init__(self, config: GenomicModelConfig):
        super().__init__()
        self.conditional = config.conditional
        init_dim = [config.x_dim]
        if self.conditional:
            init_dim[0] += config.c_embedded
        layers_dim = init_dim + config.encoder_dims

        self.architecture = nn.Sequential()
        for t, (in_dim,
                out_dim) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            self.architecture.add_module(name=f"Linear:{t}",
                                         module=nn.Linear(in_dim, out_dim))
            self.architecture.add_module(name=f"Relu:{t}", module=nn.ReLU())
        self.linear_mu = nn.Linear(layers_dim[-1], config.z_dim)
        self.linear_logvar = nn.Linear(layers_dim[-1], config.z_dim)

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=1)
            h = self.architecture(x)
            #h = h + torch.mean(c, axis=1).unsqueeze(dim=1)
        else:
            h = self.architecture(x)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        return mu, logvar


class Decoder(nn.Module):

    def __init__(self, config: GenomicModelConfig):
        super().__init__()
        self.conditional = config.conditional
        init_dim = [config.z_dim]
        if self.conditional:
            init_dim[0] += config.c_embedded
        layers_dim = init_dim + config.decoder_dims

        self.architecture = nn.Sequential()
        for t, (in_dim,
                out_dim) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            self.architecture.add_module(name=f"Linear:{t}",
                                         module=nn.Linear(in_dim, out_dim))
            self.architecture.add_module(name=f"Relu:{t}", module=nn.ReLU())
        self.last_linear = nn.Linear(layers_dim[-1], config.x_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, c=None):
        if self.conditional:
            z = torch.cat((z, c), dim=1)
            h = self.architecture(z)
            #h = h + torch.mean(c, axis=1).unsqueeze(dim=1)
        else:
            h = self.architecture(z)
        h = self.last_linear(h)
        x_hat = self.sigmoid(h)
        return x_hat


class ConditionContext(nn.Module):

    def __init__(self, config: GenomicModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.c_dim, 128)
        self.fc2 = nn.Linear(128, config.c_embedded)

    def forward(self, c):
        h = F.relu(self.fc1(c))
        h = self.fc2(h)
        return h


class Prior(nn.Module):

    def __init__(self, config: GenomicModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.c_embedded, 128)
        self.fc2 = nn.Linear(128, 128)
        self.linear_mu = nn.Linear(128, config.z_dim)
        self.linear_logvar = nn.Linear(128, config.z_dim)

    def forward(self, c):
        h = F.relu(self.fc1(c))
        h = F.relu(self.fc2(h))
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        return mu, logvar
