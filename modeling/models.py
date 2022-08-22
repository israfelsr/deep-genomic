from email.generator import DecodedGenerator
from modeling.blocks import *
from modeling.genomic_model import GenomicModel, GenomicModelConfig, Models
from utils.logging import get_logger

LOG = get_logger(__name__)


def build_model(config: GenomicModelConfig):
    LOG.info(f'Initializing {config.model} model')
    if config.model == Models.SIMPLE_VAE:
        model = SimpleVariationalModel(config)
    if config.model == Models.PRIOR_VAE:
        assert config.conditional
        model = StudentTeacherModel(config)
    if config.model == Models.AUTOENCODER:
        model = Autoencoder(config)
    return model


class Autoencoder(GenomicModel):

    def __init__(self, config: GenomicModelConfig):
        super().__init__(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x, c=None):
        z, logvar = self.encoder(x, c)
        x_hat = self.decoder(z, c)
        return {'x_hat': x_hat}

    def generate(self, c=None):
        pass


class SimpleVariationalModel(GenomicModel):

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        if config.use_context:
            self.context = ConditionContext(config)

    def forward(self, x, c=None):
        if self.config.use_context:
            c = self.context(c)
        mu, logvar = self.encoder(x, c)
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(z, c)
        return {'x_hat': x_hat, 'mu': mu, 'logvar': logvar}

    def generate(self, c=None):
        if self.config.use_context:
            c = self.context(c)
        num_samples = c.shape[0] if c is not None else 64
        with torch.no_grad():
            z = torch.randn(num_samples, self.config.z_dim)
            x_hat = self.decoder(z, c)
        return x_hat


class StudentTeacherModel(GenomicModel):

    def __init__(self, config):
        super().__init__(config)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.prior = Prior(config)
        if config.use_context:
            self.context = ConditionContext(config)

    def forward(self, x, c):
        if self.config.use_context:
            c = self.context(c)
        mu, logvar = self.encoder(x, c)
        prior_mu, prior_logvar = self.prior(c)
        z = reparametrize(prior_mu, prior_logvar)
        x_hat = self.decoder(z, c)
        return {
            'x_hat': x_hat,
            'mu': mu,
            'logvar': logvar,
            'prior_mu': prior_mu,
            'prior_logvar': prior_logvar
        }

    def generate(self, c):
        with torch.no_grad():
            if self.config.use_context:
                c = self.context(c)
            mu, logvar = self.prior(c)
            z = reparametrize(mu, logvar)
            generated = self.decoder(z, c)
        return generated
