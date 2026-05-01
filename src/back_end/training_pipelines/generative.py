from __future__ import annotations

import io
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.back_end.config.settings import RANDOM_STATE


# ---
# Shared helpers
# ---

def _to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.tensor(X, dtype=torch.float32)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _scale(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


# ---
# 1. Autoencoder
# ---

class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64),         nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def train_autoencoder(
    X: np.ndarray,
    latent_dim: int = 8,
    epochs: int = 200,
    lr: float = 1e-3,
) -> tuple[_Autoencoder, StandardScaler, list[float]]:
    """Train autoencoder, return (model, scaler, loss_history)."""
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    X_scaled, scaler = _scale(X)
    tensor = _to_tensor(X_scaled)

    model     = _Autoencoder(X.shape[1], latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        recon = model(tensor)
        loss  = criterion(recon, tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, scaler, losses


def generate_autoencoder(
    model: _Autoencoder,
    scaler: StandardScaler,
    n_samples: int,
    latent_dim: int = 8,
) -> np.ndarray:
    """Sample from the latent space and decode to feature space."""
    model.eval()
    with torch.no_grad():
        z             = torch.randn(n_samples, latent_dim)
        synthetic_sc  = _to_numpy(model.decode(z))
    return scaler.inverse_transform(synthetic_sc)


# ---
# 2. GAN
# ---

class _Generator(nn.Module):
    def __init__(self, noise_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 64),   nn.LeakyReLU(0.2),
            nn.Linear(64, 128),         nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(64, 1),          nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_gan(
    X: np.ndarray,
    noise_dim: int = 16,
    epochs: int = 300,
    lr: float = 2e-4,
) -> tuple[_Generator, StandardScaler, list[float], list[float]]:
    """Train GAN, return (generator, scaler, g_losses, d_losses)."""
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    X_scaled, scaler = _scale(X)
    real_tensor = _to_tensor(X_scaled)
    n_features  = X.shape[1]

    gen = _Generator(noise_dim, n_features)
    disc = _Discriminator(n_features)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    g_losses: list[float] = []
    d_losses: list[float] = []
    for _ in range(epochs):
        batch_size = real_tensor.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Discriminator step
        z    = torch.randn(batch_size, noise_dim)
        fake = gen(z).detach()
        d_loss = (
            criterion(disc(real_tensor), real_labels)
            + criterion(disc(fake), fake_labels)
        )
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # Generator step
        z    = torch.randn(batch_size, noise_dim)
        fake = gen(z)
        g_loss = criterion(disc(fake), real_labels)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    return gen, scaler, g_losses, d_losses


def generate_gan(
    gen: _Generator,
    scaler: StandardScaler,
    n_samples: int,
    noise_dim: int = 16,
) -> np.ndarray:
    """Generate synthetic samples with a trained GAN generator."""
    gen.eval()
    with torch.no_grad():
        z            = torch.randn(n_samples, noise_dim)
        synthetic_sc = _to_numpy(gen(z))
    return scaler.inverse_transform(synthetic_sc)


# ---
# 3. Diffusion Model (simplified DDPM for tabular data)
# ---

class _DenoisingNet(nn.Module):
    """Predicts noise given a noisy sample and a timestep embedding."""

    def __init__(self, input_dim: int, time_emb_dim: int = 16):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim), nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, 128), nn.SiLU(),
            nn.Linear(128, 128),                       nn.SiLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        return self.net(torch.cat([x, t_emb], dim=-1))


def _linear_beta_schedule(timesteps: int = 100) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, timesteps)


def train_diffusion(
    X: np.ndarray,
    timesteps: int = 100,
    epochs: int = 200,
    lr: float = 1e-3,
) -> tuple[_DenoisingNet, StandardScaler, torch.Tensor, list[float]]:
    """Train diffusion model, return (denoising_net, scaler, betas, loss_history)."""
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    X_scaled, scaler = _scale(X)
    x0         = _to_tensor(X_scaled)
    n_features = X.shape[1]

    betas     = _linear_beta_schedule(timesteps)
    alphas    = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    model     = _DenoisingNet(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        t      = torch.randint(0, timesteps, (x0.size(0),))
        ab     = alpha_bar[t].unsqueeze(-1)
        noise  = torch.randn_like(x0)
        x_noisy = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

        pred_noise = model(x_noisy, t.float() / timesteps)
        loss = criterion(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, scaler, betas, losses


def generate_diffusion(
    model: _DenoisingNet,
    scaler: StandardScaler,
    betas: torch.Tensor,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    """Generate samples via iterative denoising (reverse process)."""
    timesteps = len(betas)
    alphas    = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    model.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, n_features)
        for i in reversed(range(timesteps)):
            t          = torch.full((n_samples,), i, dtype=torch.float32) / timesteps
            pred_noise = model(x, t)
            alpha_t    = alphas[i]
            ab_t       = alpha_bar[i]
            x = (1 / torch.sqrt(alpha_t)) * (
                x - (betas[i] / torch.sqrt(1 - ab_t)) * pred_noise
            )
            if i > 0:
                x += torch.sqrt(betas[i]) * torch.randn_like(x)

    return scaler.inverse_transform(_to_numpy(x))


# ---
# Quality metrics
# ---

def wasserstein_distances(
    real: np.ndarray,
    synthetic: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Normalised 1-D Wasserstein distance per feature (lower = better)."""
    scaler = StandardScaler()
    real_norm = scaler.fit_transform(real)
    std = scaler.scale_
    std_safe = np.where(std == 0, 1.0, std)
    synth_norm = (synthetic - scaler.mean_) / std_safe

    rows = []
    for i, name in enumerate(feature_names):
        dist = wasserstein_distance(real_norm[:, i], synth_norm[:, i])
        rows.append({"feature": name, "wasserstein_distance": round(dist, 4)})
    return pd.DataFrame(rows).sort_values("wasserstein_distance")


def compare_distributions(
    real: np.ndarray,
    synthetic: np.ndarray,
    method: Literal["pca", "tsne"] = "pca",
) -> pd.DataFrame:
    """Project real and synthetic data into 2-D for visual comparison."""
    combined = np.vstack([real, synthetic])
    labels   = ["Real"] * len(real) + ["Synthetic"] * len(synthetic)

    scaler         = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    if method == "pca":
        proj = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(combined_scaled)
    else:
        perplexity = min(30, len(combined_scaled) - 1)
        proj = TSNE(
            n_components=2, random_state=RANDOM_STATE, perplexity=perplexity
        ).fit_transform(combined_scaled)

    return pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "source": labels})


# ---
# Export helpers
# ---

def synthetic_to_dataframe(
    synthetic: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Wrap a synthetic array into a named DataFrame."""
    return pd.DataFrame(synthetic, columns=feature_names)


def synthetic_to_csv_bytes(
    synthetic: np.ndarray,
    feature_names: list[str],
) -> bytes:
    """Serialise synthetic data to CSV bytes (for Streamlit download button)."""
    df  = synthetic_to_dataframe(synthetic, feature_names)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()
