import torchmetrics
import torch
import torch.nn.functional as F


class GenomicGenerationMetrics:

    def __init__(self, prefix):
        self.metrics = torchmetrics.MetricCollection({
            'acc':
            torchmetrics.Accuracy(),
            'f1_score':
            torchmetrics.F1Score()
        })
        self.running_loss = torchmetrics.MeanMetric()
        self.examples_count = 0
        self.prefix = prefix

    #TODO: make the loss with variable parameters
    def update(self, outputs, loss_function):
        for name, metric in self.metrics.items():
            metric.update(outputs['x_hat'], outputs['x'].long())
        samples = outputs['x'].shape[0]
        self.running_loss.update(loss_function(outputs) / samples)
        self.examples_count += samples

    def compute_and_reset(self):
        computed_metrics = {}
        for name, metric in self.metrics.items():
            value = metric.compute()
            computed_metrics[f"{self.prefix}{name}"] = value
        loss = self.running_loss.compute()
        computed_metrics[f"{self.prefix}loss"] = loss
        self.running_loss.reset()
        computed_metrics[f"{self.prefix}total_examples"] = self.examples_count
        self.examples_count = 0
        return {
            k: v.item() if torch.is_tensor(v) else v
            for k, v in computed_metrics.items()
        }

    def call_prepare(self, accelerator):
        self.running_loss = accelerator.prepare(self.running_loss)
        self.metrics = {
            k: accelerator.prepare(metrics_collection)
            for k, metrics_collection in self.metrics.items()
        }


def ce_elbo_loss(x_hat, x, mu, logvar):
    recon_loss = F.cross_entropy(x_hat, x.long(), reduction='sum') / x.shape[0]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def mse_elbo_loss(x_hat, x, mu, logvar):
    recon_loss = F.mse_loss(x_hat, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def bce_elbo_loss(outputs):
    recon_loss = F.binary_cross_entropy(outputs['x_hat'],
                                        outputs['x'],
                                        reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu'].pow(2) -
                               outputs['logvar'].exp())
    return recon_loss + kl_loss


def bce_prior_loss(outputs):
    elbo = bce_elbo_loss(outputs)
    q = torch.distributions.MultivariateNormal(
        outputs['mu'], torch.diag_embed(torch.exp(outputs['logvar'] * 0.5)))
    p = torch.distributions.MultivariateNormal(
        outputs['prior_mu'],
        torch.diag_embed(torch.exp(outputs['prior_logvar'] * 0.5)))
    prior_kl = torch.sum(torch.distributions.kl_divergence(q, p))
    return elbo + prior_kl


def regression_elbo_loss(x_hat, x, mu, logvar, prior_mu, prior_logvar, c,
                         c_pred):
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kl_loss = (
        1 + logvar - prior_logvar -
        torch.divide(torch.square(mu - prior_mu), torch.exp(prior_logvar)) -
        torch.divide(torch.exp(logvar), torch.exp(prior_logvar)))
    kl_loss = -0.5 * torch.sum(kl_loss, axis=0)
    #label_loss = torch.divide(0.5*torch.square(r_mean - inputs_r), K.exp(r_log_var)) +  0.5 * r_log_var
    label_loss = F.mse_loss(c_pred, c)
    return torch.mean(recon_loss + kl_loss + label_loss)


CRITERION = {
    "ce_elbo": ce_elbo_loss,
    "mse_elbo": mse_elbo_loss,
    "bce_elbo": bce_elbo_loss,
    "priorbce_elbo": bce_prior_loss
}