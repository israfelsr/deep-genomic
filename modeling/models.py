from modeling.blocks import *
from modeling.genomic_model import GenomicModel, GenomicModelConfig, Models
from utils.logging import get_logger

LOG = get_logger(__name__)


def build_model(config: GenomicModelConfig):
    LOG.info(f'Initializing {config.model} model')
    if config.model == Models.SIMPLE_VAE:
        model = SimpleVariationalModel(config)
    return model


class SimpleVariationalModel(GenomicModel):

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x, c=None):
        mu, logvar = self.encoder(x, c)
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(z, c)
        return {'x_hat': x_hat, 'x': x, 'mu': mu, 'logvar': logvar}

    def generate(self, c=None):
        num_samples = c.shape[0] if c is not None else 64
        with torch.no_grad():
            z = torch.randn(num_samples, self.config.z_dim)
            x_hat = self.decoder(z, c)
        return x_hat
