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
options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of training epochs
  --batch_size BATCH_SIZE
                        Batch size for training
  --latent_dim LATENT_DIM
                        Dimension of latent space
  --lr LR               Learning rate
  --image_size IMAGE_SIZE
                        Size of input images (assumed square)
  --input_channels INPUT_CHANNELS
                        Number of input channels (1 for grayscale, 3 for RGB)
  --feature_maps_gen FEATURE_MAPS_GEN
                        Base number of feature maps in generator
  --feature_maps_disc FEATURE_MAPS_DISC
                        Base number of feature maps in discriminator
  --dropout_rate DROPOUT_RATE
                        Dropout rate for regularization
  --use_tanh            Use tanh activation in generator output
  --use_spectral_norm   Use spectral normalization
  --dataset_type {mnist,fashion_mnist,cifar10}
                        Dataset to use: mnist, fashion_mnist, cifar10
  --inlier INLIER       Which digit to use as inlier class (0-9)
  --outlier_portion OUTLIER_PORTION
                        Portion of outliers in test set
  --save_dir SAVE_DIR   Directory to save results
  --save_interval SAVE_INTERVAL
                        Epoch interval for saving models
  --use_mps             Use MPS (Apple Silicon GPU) if available
  --use_cpu             Force CPU usage even if GPU is available
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

## Visualizing Training Evolution with Animations

You can create GIF animations to visualize how your model's reconstructions and generated samples evolve throughout the training process. The included `create_animation.py` script makes this easy.

### Animation: Learning the Data Distribution

The animations created by this script provide a powerful visual demonstration of how the BiGAN progressively learns the underlying probability distribution of the data:

- **Reconstruction Quality**: As training progresses, you can observe how the model's ability to reconstruct normal samples improves, showing its increasing understanding of the inlier class distribution.
- **Feature Learning**: The evolution of reconstructed images reveals which features the model learns first (typically coarse structures) and which require more training (fine details, textures).
- **Manifold Exploration**: Generated samples show how the model explores and maps the latent space to the data manifold, gradually producing more realistic examples.
- **Convergence Behavior**: Animations help identify if/when the model reaches convergence or if it experiences mode collapse or instability.

These visualizations are particularly valuable for anomaly detection, as they demonstrate the model's growing ability to represent normal data, which directly relates to its capability to identify anomalies as deviations from the learned distribution.

### Creating a Basic Animation

```bash
# Create animation of reconstruction comparisons
python create_animation.py --mode single --image_dir ./results/reconstruction/comparison
```

### Multi-Panel Animations

You can also create a side-by-side animation showing multiple aspects of training:

```bash
# Create multi-panel animation showing reconstructions, fake samples, and true samples
python create_animation.py --mode multi
```

### Animation Options

- Single directory mode shows evolution of a specific type of output
- Multi-panel mode shows synchronized view of different outputs at each epoch
- Adjustable frame rate and resolution
- Custom panel combinations to focus on specific aspects of learning

For more options and examples, see `animation_readme.md`.

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