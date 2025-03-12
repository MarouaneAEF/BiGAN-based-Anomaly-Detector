import torch 
import argparse
import os

from train import Trainer
from model import Discriminator, Generator
from dataloader import get_datasets


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BiGAN-based Anomaly Detector')
    
    # Basic parameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    
    # Model parameters
    parser.add_argument('--image_size', type=int, default=28, help='Size of input images (assumed square)')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--feature_maps_gen', type=int, default=32, help='Base number of feature maps in generator')
    parser.add_argument('--feature_maps_disc', type=int, default=32, help='Base number of feature maps in discriminator')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for regularization')
    parser.add_argument('--use_tanh', action='store_true', help='Use tanh activation in generator output')
    parser.add_argument('--use_spectral_norm', action='store_true', help='Use spectral normalization')
    
    # Dataset parameters
    parser.add_argument('--dataset_type', type=str, default='mnist', choices=['mnist', 'fashion_mnist', 'cifar10'], 
                        help='Dataset to use: mnist, fashion_mnist, cifar10')
    parser.add_argument('--inlier', type=int, default=0, help='Which digit to use as inlier class (0-9)')
    parser.add_argument('--outlier_portion', type=float, default=0.2, help='Portion of outliers in test set')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--save_interval', type=int, default=10, help='Epoch interval for saving models')
    
    # Hardware parameters
    parser.add_argument('--use_mps', action='store_true', help='Use MPS (Apple Silicon GPU) if available')
    parser.add_argument('--use_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    
    return parser.parse_args()


def main():
    """Main function for BiGAN training and evaluation."""
    # Get command line arguments
    args = parse_args()
    
    # Setup device
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Force GPU use if available
    if args.use_cpu:
        device = torch.device("cpu")
        print("Forced CPU usage")
    elif args.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    
    print(f"Using device: {device}")
    
    # Get datasets
    train_data, test_data = get_datasets(
        inlier=args.inlier, 
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        image_size=args.image_size,
        outlier_portion=args.outlier_portion
    )
    
    # Output directory creation
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize trainer
    bigan = Trainer(
        train_data=train_data, 
        test_data=test_data, 
        generator=Generator,
        discriminator=Discriminator,
        device=device,
        num_epochs=args.epochs,
        lr_adam=args.lr,
        lr_rmsprop=args.lr,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        input_channels=args.input_channels,
        feature_maps_gen=args.feature_maps_gen,
        feature_maps_disc=args.feature_maps_disc,
        use_spectral_norm=args.use_spectral_norm,
        dropout_rate=args.dropout_rate,
        use_tanh=args.use_tanh,
        save_dir=args.save_dir
    )
    
    print("Starting training...")
    
    # Training and testing loop
    for epoch in range(1, args.epochs + 1):
        bigan.training(epoch)
        bigan.test(epoch)
        
        # Print epoch divider for readability
        if epoch % 10 == 0:
            print(f"{'-'*40}\nCompleted {epoch} / {args.epochs} epochs\n{'-'*40}")
    
    print(f"Training completed. Results saved in {args.save_dir}")


if __name__ == "__main__":
    main()
    
