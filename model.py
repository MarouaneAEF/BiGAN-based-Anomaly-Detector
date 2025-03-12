import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional, Union

class Discriminator(nn.Module):
    """
    BiGAN Discriminator that processes both input (x) and latent code (z).
    
    The architecture is dynamic and adapts to different input image sizes.
    It has three components:
    1. x_graph: Processes the input image through convolutional layers
    2. z_graph: Processes the latent code
    3. xz_graph: Combines and processes the features from x_graph and z_graph
    """
    def __init__(self, 
                 z_dim: int, 
                 input_channels: int = 1, 
                 image_size: int = 28,
                 feature_maps_disc: int = 32,
                 dropout_rate: float = 0.0):
        """
        Initialize the Discriminator.
        
        Args:
            z_dim: Dimension of the latent space
            input_channels: Number of channels in the input image (1 for grayscale, 3 for RGB)
            image_size: Width/height of the input image (assuming square images)
            feature_maps_disc: Base number of feature maps in convolutional layers
            dropout_rate: Dropout rate for regularization (0.0 to disable dropout)
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.input_channels = input_channels
        self.image_size = image_size
        self.dropout_rate = dropout_rate
        self.feature_maps = feature_maps_disc
        
        # Calculate output size of x_graph
        # This dynamically adapts to different input image sizes
        num_layers = int(np.log2(image_size)) - 2  # At least 2 layers
        num_layers = max(3, min(num_layers, 5))  # Between 3 and 5 layers
        
        self.final_size = max(1, image_size // (2 ** num_layers))
        
        # Build x_graph (image processing)
        x_layers = []
        current_channels = input_channels
        for i in range(num_layers):
            output_channels = self.feature_maps * (2 ** i)
            x_layers.extend([
                nn.Conv2d(current_channels, output_channels, 
                          kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(output_channels) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            if dropout_rate > 0:
                x_layers.append(nn.Dropout2d(dropout_rate))
            current_channels = output_channels
            
        # Final layer to reach feature_dim
        feature_dim = self.feature_maps * (2 ** (num_layers - 1))
        x_layers.extend([
            nn.Conv2d(current_channels, feature_dim, 
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        self.xStack = nn.Sequential(*x_layers)
        self.feature_dim = feature_dim
        
        # z_graph (latent code processing)
        self.zStack = nn.Sequential(
            nn.Conv2d(z_dim, feature_dim // 2, 1, stride=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_dim // 2, feature_dim, 1, stride=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # joint (x,z) graph with global average pooling to ensure single output value
        self.xzStack = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim * 2, 1, stride=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(feature_dim * 2, feature_dim, 1, stride=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(feature_dim, feature_dim, 1, stride=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final projection to single value with global average pooling
        self.final_layer = nn.Sequential(
            nn.Conv2d(feature_dim, 1, 1, stride=1, bias=True)
        )

    def x_graph(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input image."""
        return self.xStack(x)
    
    def z_graph(self, z: torch.Tensor) -> torch.Tensor:
        """Process the latent code."""
        return self.zStack(z)
    
    def xz_graph(self, xz: torch.Tensor) -> torch.Tensor:
        """Process the concatenated features."""
        return self.xzStack(xz)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input image [batch_size, channels, height, width]
            z: Latent code [batch_size, z_dim, 1, 1]
            
        Returns:
            Discriminator output (probability of real)
        """
        x_features = self.x_graph(x)
        z_features = self.z_graph(z)
        
        # Fix dimension mismatch by resizing z_features to match x_features spatial dimensions
        if x_features.shape[2:] != z_features.shape[2:]:
            # Get the target spatial dimensions from x_features
            target_height, target_width = x_features.shape[2], x_features.shape[3]
            
            # Resize z_features to match x_features spatial dimensions
            z_features = F.interpolate(
                z_features, 
                size=(target_height, target_width),
                mode='bilinear', 
                align_corners=False
            )
            
        # Now concatenate along channel dimension
        xz_features = torch.cat((x_features, z_features), dim=1)
        
        # Process through xz_graph
        features = self.xz_graph(xz_features)
        
        # Apply final 1x1 convolution
        logits = self.final_layer(features)
        
        # Global average pooling to ensure [B, 1, 1, 1] output regardless of spatial dimensions
        logits = F.adaptive_avg_pool2d(logits, 1)
        
        # Reshape to [B, 1]
        logits = logits.view(logits.size(0), -1)
        
        return torch.sigmoid(logits)


class Generator(nn.Module):
    """
    BiGAN Generator that includes both an encoder (x → z) and decoder (z → x).
    
    The architecture is dynamic and adapts to different image sizes.
    """
    def __init__(self, 
                 z_dim: int, 
                 input_channels: int = 1, 
                 image_size: int = 28,
                 feature_maps_gen: int = 32,
                 use_tanh: bool = True):
        """
        Initialize the Generator.
        
        Args:
            z_dim: Dimension of the latent space
            input_channels: Number of channels in the input/output image
            image_size: Width/height of the input image (assuming square images)
            feature_maps_gen: Base number of feature maps in layers
            use_tanh: Whether to use tanh activation for the output (to maintain -1 to 1 range)
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.input_channels = input_channels
        self.image_size = image_size
        self.use_tanh = use_tanh
        self.feature_maps = feature_maps_gen
        
        # Calculate intermediate sizes for encoder/decoder
        # This adapts to different image sizes dynamically
        num_encoder_layers = int(np.log2(image_size)) - 2
        num_encoder_layers = max(3, min(num_encoder_layers, 5))  # Between 3 and 5 layers
        
        self.initial_size = max(2, image_size // (2 ** num_encoder_layers))
        
        # Encoder Stack (x → z)
        enc_layers = []
        in_channels = input_channels
        for i in range(num_encoder_layers):
            out_channels = self.feature_maps * (2 ** i)
            enc_layers.extend([
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
        
        # Final encoder layer to produce 2*z_dim (for mu and logvar)
        enc_layers.extend([
            nn.Conv2d(in_channels, 512, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 2*self.z_dim, 
                      kernel_size=self.initial_size, stride=1, padding=0, bias=True)
        ])
        
        self.enc_stack = nn.Sequential(*enc_layers)
        
        # Decoder Stack (z → x)
        # Start with a dense layer to convert z to initial volume
        dec_layers = []
        latent_size = self.initial_size
        
        # First layer that expands z to the initial volume
        dec_layers.extend([
            nn.ConvTranspose2d(z_dim, self.feature_maps * (2 ** (num_encoder_layers - 1)), 
                              kernel_size=latent_size, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.feature_maps * (2 ** (num_encoder_layers - 1))),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Upsampling layers
        for i in range(num_encoder_layers - 1, 0, -1):
            dec_layers.extend([
                nn.ConvTranspose2d(self.feature_maps * (2 ** i), self.feature_maps * (2 ** (i-1)), 
                                  kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.feature_maps * (2 ** (i-1))),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Final layer to output the image
        dec_layers.extend([
            nn.ConvTranspose2d(self.feature_maps, input_channels, 
                              kernel_size=4, stride=2, padding=1, bias=True)
        ])
        
        # Output activation based on config
        if use_tanh:
            dec_layers.append(nn.Tanh())
        else:
            dec_layers.append(nn.Sigmoid())
            
        self.dec_stack = nn.Sequential(*dec_layers)

    def decoding(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector z to image space.
        
        Args:
            z: Latent vector [batch_size, z_dim, 1, 1]
            
        Returns:
            Reconstructed image
        """
        return self.dec_stack(z)
    
    def reparametrize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE component.
        
        Args:
            z: Encoder output with 2*z_dim features (mu and logvar)
            
        Returns:
            Sampled latent vector
        """
        mu, log_var = z[:, :self.z_dim], z[:, self.z_dim:]
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        sample = mu + epsilon * std
        return sample
    
    def encoding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space.
        
        Args:
            x: Input image [batch_size, channels, height, width]
            
        Returns:
            Latent vector
        """
        x = self.enc_stack(x)
        x = x.view(x.size(0), -1)  # Flatten
        z_sample = self.reparametrize(x)
        return z_sample
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the generator.
        
        Args:
            x: Input image [batch_size, channels, height, width]
            
        Returns:
            Tuple of (reconstructed_image, latent_vector)
        """
        z = self.encoding(x)
        z = z.view(-1, self.z_dim, 1, 1)
        x_reconstructed = self.decoding(z)
        return x_reconstructed, z


# Spectral Normalization for improved stability (optional)
def add_spectral_norm(model):
    """
    Add spectral normalization to all conv layers in the model.
    This improves training stability.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            module = torch.nn.utils.spectral_norm(module)
    return model