# Enhanced BiGAN-based Anomaly Detector

This repository contains an improved implementation of the Bidirectional Generative Adversarial Network (BiGAN) for anomaly detection, as described in the paper ["Efficient GAN-Based Anomaly Detection"](https://arxiv.org/abs/1802.06222) by Zenati et al.

## Features

- **Dynamic architecture** that adapts to different image sizes and channel counts
- **Multiple dataset support**: MNIST, Fashion-MNIST, CIFAR-10, and custom datasets
- **Hardware acceleration** support: CUDA, MPS (Apple Silicon), and CPU
- **Comprehensive visualization** of training progress and anomaly detection results
- **Flexible configuration** through command-line arguments
- **Improved stability** with spectral normalization, label smoothing, and proper weight initialization

## Installation

1. Clone the repository:

```bash
git clone https://github.com/MarouaneAEF/BiGAN-based-Anomaly-Detector.git
cd BiGAN-based-Anomaly-Detector
```

2. Install the required dependencies:

```bash
pip install torch torchvision matplotlib numpy pillow
```

## Usage

### Basic Training

To train the model with default settings (MNIST, digit 0 as inlier):

```bash
python main.py
```

### Using MPS (Apple Silicon GPU)

For Mac users with M1/M2/M3 chips:

```bash
python main.py --use_mps
```

### Training on Different Datasets

```bash
# Fashion-MNIST with class 1 as inlier
python main.py --dataset_type fashion_mnist --inlier 1

# CIFAR-10 with class 0 (airplanes) as inlier
python main.py --dataset_type cifar10 --inlier 0 --input_channels 3 --image_size 32
```

### Full Parameter List

```
Basic parameters:
  --epochs INT         Number of training epochs (default: 200)
  --batch_size INT     Batch size for training (default: 64)
  --latent_dim INT     Dimension of latent space (default: 128)
  --lr FLOAT           Learning rate (default: 2e-4)

Model parameters:
  --image_size INT     Size of input images (default: 28)
  --input_channels INT Number of input channels (1 for grayscale, 3 for RGB) (default: 1)
  --feature_maps_gen INT  Base number of feature maps in generator (default: 32)
  --feature_maps_disc INT Base number of feature maps in discriminator (default: 32)
  --dropout_rate FLOAT    Dropout rate for regularization (default: 0.2)
  --use_tanh           Use tanh activation in generator output
  --use_spectral_norm  Use spectral normalization for stability

Dataset parameters:
  --inlier INT         Which digit/class to use as inlier (0-9 for MNIST) (default: 0)
  --outlier_portion FLOAT  Portion of outliers in test set (default: 0.2)
  --dataset_type STR   Dataset to use: mnist, fashion_mnist, cifar10 (default: mnist)

Output parameters:
  --save_dir STR       Directory to save results (default: ./results)
  --save_interval INT  Epoch interval for saving models (default: 10)

Hardware parameters:
  --use_mps            Use MPS (Apple Silicon GPU) if available
  --use_cpu            Force CPU usage even if GPU is available
```

## Model Architecture

The model consists of three main components:

1. **Encoder (E)**: Maps real data `x` to latent code `z`.
2. **Generator (G)**: Maps latent code `z` to generated data `G(z)`.
3. **Discriminator (D)**: Discriminates between the pairs `(x, E(x))` and `(G(z), z)`.

The architecture is dynamic and adapts to different image sizes automatically.

## Anomaly Detection

After training, the model computes an anomaly score for each input image based on:
1. The reconstruction error `||x - G(E(x))||`.
2. The discriminator's assessment of whether the encoded pair `(x, E(x))` is real.

Anomalous samples will have higher anomaly scores than normal samples.

## Example Outputs

The training process generates several visualizations in the `results` directory:

- `reconstruction/comparison/`: Shows original and reconstructed images side by side
- `reconstruction/fake/`: Shows samples from the generator
- `reconstruction/true/`: Shows original samples from the dataset
- `reconstruction/test/`: Shows test set evaluations
- `stats/`: Contains loss curves and other statistics
- `models/`: Contains model checkpoints

## Customization

You can customize the model by editing:

- `model.py`: For architecture changes
- `train.py`: For training procedure changes
- `dataloader.py`: For custom datasets

## Citing

If you use this code for your research, please cite the original paper:

```
@inproceedings{zenati2018efficient,
  title={Efficient GAN-Based Anomaly Detection},
  author={Zenati, Houssam and Foo, Chuan Sheng and Lecouat, Bruno and Manek, Gaurav and Chandrasekhar, Vijay R},
  booktitle={International Conference on Learning Representations (ICLR) Workshop},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.