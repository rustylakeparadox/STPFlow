import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 限制 logvar 范围，避免数值过大或过小
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


def vae_loss(
    x_hat,
    x,
    mu,
    logvar,
    beta: float = 1e-5,
    recon_loss_type: str = "mse",
):
    if recon_loss_type == "mse":
        recon = F.mse_loss(x_hat, x, reduction="mean")
    elif recon_loss_type == "smooth_l1":
        recon = F.smooth_l1_loss(x_hat, x, reduction="mean")
    else:
        raise ValueError(f"Unsupported recon_loss_type: {recon_loss_type}")

    # 按 latent dim 求平均的 KL，更稳定一些
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl = kl.mean()

    loss = recon + beta * kl
    return loss, recon, kl

def corr_loss(pred, target, eps=1e-8):
    pred_c = pred - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)

    pred_z = pred_c / (pred_c.std(dim=0, keepdim=True) + eps)
    target_z = target_c / (target_c.std(dim=0, keepdim=True) + eps)

    corr = (pred_z * target_z).mean(dim=0)
    return 1.0 - corr.mean()