import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm
from metrics import empirical_wasserstein_1

# Build the toy dataset
N = 1024
X, color = make_swiss_roll(n_samples=N, noise=0.4)
X = X[:, [0, 2]].astype(np.float32)        # 2D projection

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
X_tensor = torch.from_numpy(X_scaled)

# Build the data loader
batch_size = 256
dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_tensor),
    batch_size=batch_size, shuffle=True, drop_last=True,
)

# Define generator architecture
class Generator(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=64, data_dim=2):
        super().__init__()
        
        class ResBlock(nn.Module):
            def __init__(self,dim):
                super().__init__()

                # Linear layer
                self.l = nn.Linear(dim,dim)

                # ReLU
                self.act = nn.ReLU()

            def forward(self, z):
                # skip connection
                return z + self.act(self.l(z))
        
        # Linear transform - 3 residual blocks - projection to data space
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, z):
        return self.net(z)

# Define descriminator architecture
class Discriminator(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=64):
        super().__init__()
        
        class ResBlock(nn.Module):
            def __init__(self,dim):
                super().__init__()
                # Spectrally normalized linear layer
                self.l = spectral_norm(nn.Linear(dim,dim))

                # Leaky relu activation
                self.act = nn.LeakyReLU(0.2)

            def forward(self, z):
                # Skip connection
                return z + self.act(self.l(z))

        # Linear transform - 3 residual blocks - projection to logit
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        return self.net(z)

# Instantiate the generator and discriminator
latent_dim = 2
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator().to(device)

# Some consistent query samples for visualization
z_0 = torch.randn(1000,2).to(device)
sample_list = []
class GANTrainer:
    def __init__(self, generator, discriminator, latent_dim, lr=1e-3):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

        # Note non-standard ADAM parameters!
        self.opt_g = torch.optim.Adam(generator.parameters(), lr=0.2*lr,betas=(0.5,0.999))
        self.opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr,betas=(0.5,0.999))

    def train(self, dataloader, epochs=5000):
        for epoch in range(epochs):
            for (real,) in dataloader:
                B = real.size(0)
                real = real.to(device)

                # --- Discriminator step ---
                z = torch.randn(B, self.latent_dim).to(device)
                fake = self.generator(z).detach()

                a_pred_real = self.discriminator(real)
                a_pred_fake = self.discriminator(fake)

                y_pred_real = torch.sigmoid(a_pred_real)
                y_pred_fake = torch.sigmoid(a_pred_fake)

                loss_d = (
                    - (0.9*torch.log(y_pred_real) + 0.1*torch.log(1 - y_pred_real)).mean()
                    - (torch.log(1 - y_pred_fake)).mean()
                )
                self.opt_d.zero_grad(); loss_d.backward(); self.opt_d.step()

                # --- Generator step ---
                z = torch.randn(B, self.latent_dim).to(device)
                fake = self.generator(z)

                a_pred_fake = self.discriminator(fake)
                y_pred_fake = torch.sigmoid(a_pred_fake)

                loss_g = -torch.log(y_pred_fake).mean()
                self.opt_g.zero_grad(); loss_g.backward(); self.opt_g.step()

            if (epoch) % 20 == 0:
  
                dist = empirical_wasserstein_1(real,fake)
                samples = self.generator(z_0)
                sample_list.append(np.array(samples.detach().cpu()))
                print(f"Epoch {epoch+1:4d} | loss_d={loss_d.item():.3f} | loss_g={loss_g.item():.3f} | W_dist={dist}:.3f")

# Train
trainer = GANTrainer(generator, discriminator, latent_dim)
trainer.train(dl)

# Plot
from plotting import *
animate_gan(X_tensor,sample_list)
plot_discriminator_vector_field(discriminator, sample_list[-1], None, xlim=(-3, 3), ylim=(-3, 3), n=50, device=device)

