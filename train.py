from re import I
import torch 
from torch import optim 
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.utils as vutils 
import numpy as np 
import os
import time
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional, Union, List

from model import Generator, Discriminator, add_spectral_norm
from utils.utils import weights_init_normal

class Trainer:
    """
    Trainer class for BiGAN-based anomaly detection.
    
    This class handles the training and evaluation of the BiGAN model
    with configurable parameters for different types of images.
    """
    def __init__(self,  
                 train_data,
                 test_data,
                 generator=Generator,
                 discriminator=Discriminator,  
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 num_epochs=200, 
                 lr_adam=1e-4, 
                 lr_rmsprop=1e-4, 
                 batch_size=64, 
                 latent_dim=256,
                 image_size=28,
                 input_channels=1,
                 feature_maps_gen=32,
                 feature_maps_disc=32,
                 use_spectral_norm=False,
                 dropout_rate=0.0,
                 use_tanh=True,
                 save_dir="./results"):
        """
        Initialize the Trainer.
        
        Args:
            train_data: DataLoader for training data
            test_data: DataLoader for testing data
            generator: Generator class
            discriminator: Discriminator class
            device: Device to run on (cuda/mps/cpu)
            num_epochs: Number of training epochs
            lr_adam: Learning rate for Adam optimizer
            lr_rmsprop: Learning rate for RMSprop optimizer
            batch_size: Batch size for training
            latent_dim: Dimension of the latent space
            image_size: Size of the input images (assuming square)
            input_channels: Number of channels in the input image
            feature_maps_gen: Base feature maps for generator
            feature_maps_disc: Base feature maps for discriminator
            use_spectral_norm: Whether to use spectral normalization
            dropout_rate: Dropout rate for regularization
            use_tanh: Whether to use tanh in generator output
            save_dir: Directory to save outputs
        """
        self.generator_class = generator 
        self.discriminator_class = discriminator
        self.num_epochs = num_epochs
        self.lr_adam = lr_adam
        self.lr_rmsprop = lr_rmsprop
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.train_loader = train_data
        self.test_loader = test_data
        self.device = device
        self.image_size = image_size
        self.input_channels = input_channels
        self.feature_maps_gen = feature_maps_gen
        self.feature_maps_disc = feature_maps_disc
        self.use_spectral_norm = use_spectral_norm
        self.dropout_rate = dropout_rate
        self.use_tanh = use_tanh
        self.save_dir = save_dir
        
        # Create save directories
        self.create_directories()
        
        # Initialize statistics tracking
        self.train_losses = {"g_loss": [], "d_loss": [], "epochs": []}
        self.test_losses = {"g_loss": [], "d_loss": [], "epochs": []}
    
    def create_directories(self):
        """Create necessary directories for saving outputs."""
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/models", exist_ok=True)
        os.makedirs(f"{self.save_dir}/reconstruction/fake", exist_ok=True)
        os.makedirs(f"{self.save_dir}/reconstruction/true", exist_ok=True)
        os.makedirs(f"{self.save_dir}/reconstruction/comparison", exist_ok=True)
        os.makedirs(f"{self.save_dir}/reconstruction/test", exist_ok=True)
        os.makedirs(f"{self.save_dir}/stats", exist_ok=True)

    def get_models(self):
        """Create and initialize the generator and discriminator models."""
        # Initialize generator with appropriate parameters
        G = self.generator_class(
            z_dim=self.latent_dim, 
            input_channels=self.input_channels, 
            image_size=self.image_size,
            feature_maps_gen=self.feature_maps_gen,
            use_tanh=self.use_tanh
        ).to(self.device)
        
        # Initialize discriminator with appropriate parameters
        D = self.discriminator_class(
            z_dim=self.latent_dim, 
            input_channels=self.input_channels, 
            image_size=self.image_size,
            feature_maps_disc=self.feature_maps_disc,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Apply spectral normalization if requested
        if self.use_spectral_norm:
            G = add_spectral_norm(G)
            D = add_spectral_norm(D)
            
        # Initialize weights
        G.apply(weights_init_normal)
        D.apply(weights_init_normal)
        
        return G, D

    def training(self, epoch):
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
        """
        # Get models
        G, D = self.get_models()
        G.train()
        D.train()
        
        # Optimizers
        optimizer_g = optim.Adam(G.parameters(), lr=self.lr_adam, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(D.parameters(), lr=self.lr_adam, betas=(0.5, 0.999))
        
        epoch_start_time = time.time()
        i = 0
        g_loss_sum = 0
        d_loss_sum = 0
        
        try:
            # Debug information for first epoch
            if epoch == 1:
                print(f"Device being used: {self.device}")
                print(f"Model parameters - Image size: {self.image_size}, Channels: {self.input_channels}, Latent dim: {self.latent_dim}")
                # Print summary of model architecture
                print(f"\nGenerator architecture:")
                print(G)
                print(f"\nDiscriminator architecture:")
                print(D)
                
            # Main training loop
            for x, _ in self.train_loader:
                batch_size = x.size(0)
                
                # Debug info for first batch of first epoch
                if epoch == 1 and i == 0:
                    print(f"Input batch shape: {x.shape}")
                
                #  Train Discriminator
                D.zero_grad()
                
                # Setup real data
                y_true = torch.ones(batch_size, 1).to(self.device)
                y_fake = torch.zeros(batch_size, 1).to(self.device)
                
                # Label smoothing for better stability
                y_true = y_true * 0.9 + 0.1 * torch.rand_like(y_true)
                
                # Add noise to improve robustness (decreases with epoch)
                noise_factor = 0.1 * (self.num_epochs - epoch) / self.num_epochs
                
                # Process real data
                x_true = x.float().to(self.device)
                
                # Add noise to real data (matching dimensions)
                noise1 = torch.randn_like(x_true) * noise_factor
                
                z_true = G.encoding(x_true)
                z_true = z_true.view(batch_size, self.latent_dim, 1, 1)
                
                # Debug only for first batch of first epoch
                if epoch == 1 and i == 0:
                    print(f"x_true shape: {x_true.shape}")
                    print(f"z_true shape: {z_true.shape}")
                
                out_true = D(x_true + noise1, z_true)
                
                # Debug discriminator output shape
                if epoch == 1 and i == 0:
                    print(f"Discriminator out_true shape: {out_true.shape}")
                    print(f"y_true shape: {y_true.shape}")
                
                # Real loss
                loss_d_real = F.binary_cross_entropy(out_true, y_true)
                loss_d_real.backward(retain_graph=True)
                
                # Process fake data
                z_fake = torch.randn((batch_size, self.latent_dim, 1, 1)).to(self.device)
                x_fake = G.decoding(z_fake)
                
                # Add noise to fake data (matching dimensions)
                noise2 = torch.randn_like(x_fake) * noise_factor
                
                out_fake = D(x_fake.detach() + noise2, z_fake)
                
                # Fake loss
                loss_d_fake = F.binary_cross_entropy(out_fake, y_fake)
                loss_d_fake.backward()
                
                # Combined discriminator loss
                loss_d = 0.5 * (loss_d_real + loss_d_fake)
                
                # Update discriminator
                optimizer_d.step()
                
                # Train Generator
                G.zero_grad()
                
                # Generator wants discriminator to think fake is real
                out_fake = D(x_fake + noise2, z_fake)
                loss_gdecoding = F.binary_cross_entropy(out_fake, y_true)
                loss_gdecoding.backward(retain_graph=True)
                
                # Encoder wants discriminator to think encoded is real
                out_true = D(x_true + noise1, z_true)
                loss_gencoding = F.binary_cross_entropy(out_true, y_true)
                loss_gencoding.backward()
                
                # Combined generator loss
                loss_g = 0.5 * (loss_gdecoding + loss_gencoding)
                
                # Update generator
                optimizer_g.step()
                
                # Track losses
                d_loss_sum += loss_d.item()
                g_loss_sum += loss_g.item()
                
                # Log progress
                if i % 10 == 0:
                    print(f"Epoch:{epoch}, Iter:{i}, D_Loss:{loss_d.item():.5f}, G_Loss:{loss_g.item():.5f}, "
                          f"D(x):{out_true.mean().item():.5f}, D(G(z)):{out_fake.mean().item():.5f}")
                
                # Save sample images
                if i % 100 == 0:
                    self.save_images(x_fake, x_true, epoch, i)
                    avg_d_loss = d_loss_sum / (i + 1)
                    avg_g_loss = g_loss_sum / (i + 1)
                    print(f"Training Epoch {epoch}, Avg Discriminator Loss: {avg_d_loss:.5f}, "
                          f"Avg Generator Loss: {avg_g_loss:.5f}")
                i += 1
            
            # Track statistics for plotting
            self.train_losses["epochs"].append(epoch)
            self.train_losses["d_loss"].append(d_loss_sum / len(self.train_loader))
            self.train_losses["g_loss"].append(g_loss_sum / len(self.train_loader))
            
            # Save model and comparison at end of epoch
            if epoch % 50 == 0:
                self.save_comparison_grid(G, epoch)
                self.save_models(G, D, epoch)
                self.plot_losses()
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s. "
                  f"Avg D_Loss: {self.train_losses['d_loss'][-1]:.5f}, "
                  f"Avg G_Loss: {self.train_losses['g_loss'][-1]:.5f}")
                
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def test(self, epoch):
        """
        Test the model and compute metrics.
        
        Args:
            epoch: Current epoch number
        """
        G, D = self.get_models()
        G.eval()
        D.eval()
        
        d_loss = 0
        g_loss = 0
        
        try:
            with torch.no_grad():
                for x, _ in self.test_loader:
                    batch_size = x.size(0)
                    
                    y_true = torch.ones(batch_size, 1).to(self.device)
                    y_fake = torch.zeros(batch_size, 1).to(self.device)
                    
                    # Noise for testing (smaller than training)
                    noise_factor = 0.05 * (self.num_epochs - epoch) / self.num_epochs
                    
                    # Process real and fake data
                    z_fake = torch.randn((batch_size, self.latent_dim, 1, 1)).to(self.device)
                    x_fake = G.decoding(z_fake)
                    
                    x_true = x.float().to(self.device)
                    
                    # Create noise with proper dimensions
                    noise1 = torch.randn_like(x_true) * noise_factor
                    noise2 = torch.randn_like(x_fake) * noise_factor
                    
                    z_true = G.encoding(x_true).view(batch_size, self.latent_dim, 1, 1)
                    
                    out_true = D(x_true + noise1, z_true)
                    out_fake = D(x_fake + noise2, z_fake)
                    
                    # Calculate losses
                    criterion = nn.BCELoss()
                    loss_d_real = criterion(out_true, y_true)
                    loss_d_fake = criterion(out_fake, y_fake)
                    
                    # Metrics
                    loss_d = 0.5 * (loss_d_real + loss_d_fake)
                    loss_gencoding = F.binary_cross_entropy(out_true, y_true)
                    loss_gdecoding = F.binary_cross_entropy(out_fake, y_true)
                    loss_g = 0.5 * (loss_gdecoding + loss_gencoding)
                    
                    d_loss += loss_d.item()
                    g_loss += loss_g.item()
                
                # Print test results
                if epoch % 10 == 0:
                    avg_d_loss = d_loss / len(self.test_loader)
                    avg_g_loss = g_loss / len(self.test_loader)
                    
                    print(f"Test Epoch {epoch}, Avg D_Loss: {avg_d_loss:.5f}, "
                          f"Avg G_Loss: {avg_g_loss:.5f}, "
                          f"D(x): {out_true.mean().item():.5f}, "
                          f"D(G(z)): {out_fake.mean().item():.5f}")
                
                # Save test statistics
                self.test_losses["epochs"].append(epoch)
                self.test_losses["d_loss"].append(d_loss / len(self.test_loader))
                self.test_losses["g_loss"].append(g_loss / len(self.test_loader))
                
                # Save test reconstructions
                if epoch % 30 == 0:
                    test_samples = torch.cat([x_true.data[:16], x_fake.data[:16]])
                    vutils.save_image(test_samples, f'{self.save_dir}/reconstruction/test/{epoch}.png')
                    
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            raise
    
    def save_images(self, x_fake, x_true, epoch, iteration):
        """Save generated and real images during training."""
        try:
            vutils.save_image(x_fake.data[:10], 
                              f'{self.save_dir}/reconstruction/fake/fake_{epoch}_{iteration}.png')
            vutils.save_image(x_true.data[:10], 
                              f'{self.save_dir}/reconstruction/true/true_{epoch}_{iteration}.png')
        except Exception as e:
            print(f"Error saving images: {str(e)}")
    
    def save_comparison_grid(self, G, epoch):
        """Save a grid comparing real vs generated images."""
        try:
            # Get some test data
            test_batch = next(iter(self.test_loader))[0][:16].to(self.device)
            # Generate reconstructions
            with torch.no_grad():
                reconstructed, _ = G(test_batch)
            # Concatenate real and reconstructed
            comparison = torch.cat([test_batch.data, reconstructed.data])
            vutils.save_image(comparison, 
                              f'{self.save_dir}/reconstruction/comparison/{epoch}.png', 
                              nrow=16)
        except Exception as e:
            print(f"Error saving comparison grid: {str(e)}")
    
    def save_models(self, G, D, epoch):
        """Save model checkpoints."""
        try:
            torch.save(G.state_dict(), f"{self.save_dir}/models/generator_{epoch}.pth")
            torch.save(D.state_dict(), f"{self.save_dir}/models/discriminator_{epoch}.pth")
        except Exception as e:
            print(f"Error saving models: {str(e)}")
    
    def plot_losses(self):
        """Plot and save loss curves."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Training losses
            plt.subplot(2, 1, 1)
            plt.plot(self.train_losses["epochs"], self.train_losses["g_loss"], label="Generator Loss")
            plt.plot(self.train_losses["epochs"], self.train_losses["d_loss"], label="Discriminator Loss")
            plt.title('Training Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Test losses
            plt.subplot(2, 1, 2)
            plt.plot(self.test_losses["epochs"], self.test_losses["g_loss"], label="Generator Loss")
            plt.plot(self.test_losses["epochs"], self.test_losses["d_loss"], label="Discriminator Loss")
            plt.title('Test Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/stats/losses.png")
            plt.close()
        except Exception as e:
            print(f"Error plotting losses: {str(e)}")

    def anomaly_detector(self, x):
        """
        Compute anomaly score for input data.
        
        This method measures how anomalous a data point x is.
        Samples with larger values of A are more likely to be anomalous.
        For perfectly non-anomalous data point x: A ~ .5 * log(D(x)) which
        ranges in (0, 1).
        
        Args:
            x: Input data to evaluate
            
        Returns:
            Anomaly score
        """
        G, D = self.get_models()
        G.eval()
        D.eval()

        try:
            with torch.no_grad():
                batch_size = x.size(0)
                y_true = torch.ones(batch_size, 1).to(self.device)
                x_true = x.float().to(self.device)
                
                # Encode the input
                z_true = G.encoding(x_true).view(batch_size, self.latent_dim, 1, 1)
                out_true = D(x_true, z_true)
                
                # Generate reconstruction
                reconstructed, _ = G(x_true)
                
                # Compute reconstruction error
                Lg = F.mse_loss(reconstructed, x_true)
                
                # Compute discriminator score
                Ld = F.binary_cross_entropy(out_true, y_true)
                
                # Combined anomaly score
                A = 0.5 * (Lg + Ld)
                
                print(f"Anomaly score: {A:.5f} | Reconstruction loss: {Lg:.5f} | Discriminator score: {Ld:.5f}")
                return A, Lg, Ld
                
        except Exception as e:
            print(f"Error in anomaly detection: {str(e)}")
            raise



        

