# BiGAN Reconstruction Animation

The `create_animation.py` script allows you to visualize the evolution of your BiGAN model's reconstructions over the training epochs.

## Requirements

Make sure you have the required dependencies:
```bash
pip install matplotlib pillow numpy
```

## Usage

### Single Directory Animation

Create an animation from reconstructed images in a single directory:

```bash
python create_animation.py --mode single --image_dir ./results/reconstruction/comparison --output ./results/comparison_animation.gif
```

### Multi-Panel Animation

Create a side-by-side animation showing multiple aspects of the training process:

```bash
python create_animation.py --mode multi --output ./results/multi_animation.gif
```

By default, this will create an animation with three panels showing:
1. Original vs reconstructed images (comparison)
2. Generated images (fake)
3. Original images (true)

### Options

- `--mode`: Animation mode, either 'single' or 'multi' (default: 'single')
- `--image_dir`: Directory containing images for single mode (default: './results/reconstruction/comparison')
- `--dirs`: Directories to include in multi-panel mode (space-separated) (default: comparison, fake, and true directories)
- `--output`: Path to save the animation (default: './results/animation.gif')
- `--fps`: Frames per second for the animation (default: 2)
- `--dpi`: DPI for the output animation (default: 100)

## Examples

### Generate animation of test reconstructions

```bash
python create_animation.py --mode single --image_dir ./results/reconstruction/test --output ./results/test_animation.gif
```

### Generate animation of fake samples only

```bash
python create_animation.py --mode single --image_dir ./results/reconstruction/fake --output ./results/fake_animation.gif
```

### Customize multi-panel animation

```bash
python create_animation.py --mode multi --dirs ./results/reconstruction/comparison ./results/reconstruction/test --output ./results/custom_animation.gif --fps 1
```

## How It Works

The script reads PNG images from the specified directories, sorts them by epoch number, and creates an animation showing how the images evolve over the training process. For multi-panel mode, it synchronizes the frames across multiple directories to show different aspects of the model's behavior at the same training stage. 