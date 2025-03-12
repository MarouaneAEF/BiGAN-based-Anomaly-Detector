import os
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import re

def natural_sort_key(s):
    """Sort strings that contain numbers in a natural way."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def create_animation(image_dir, output_path, fps=5, dpi=100):
    """
    Creates an animation from a series of images in the specified directory.
    
    Args:
        image_dir (str): Directory containing the images to animate
        output_path (str): Path to save the animation file
        fps (int): Frames per second for the animation
        dpi (int): DPI for the output animation
    """
    # Get all image files and sort them by epoch number
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    # Sort files naturally so epochs are in order
    image_files.sort(key=natural_sort_key)
    
    # Create figure
    first_img = Image.open(image_files[0])
    fig, ax = plt.subplots(figsize=(first_img.width/dpi, first_img.height/dpi))
    plt.tight_layout()
    plt.axis('off')
    
    # First frame
    img = plt.imread(image_files[0])
    im = ax.imshow(img, animated=True)
    
    title = ax.text(0.5, 1.02, "Epoch: 0", size=12, ha="center", transform=ax.transAxes)
    
    def update(frame):
        """Update function for the animation"""
        # Extract epoch number from filename
        epoch_match = re.search(r'(\d+)', os.path.basename(image_files[frame]))
        epoch_num = epoch_match.group(1) if epoch_match else str(frame)
        
        # Update image
        img = plt.imread(image_files[frame])
        im.set_array(img)
        
        # Update title
        title.set_text(f"Epoch: {epoch_num}")
        return [im, title]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(image_files), interval=1000/fps, blit=True)
    
    # Save animation
    print(f"Creating animation with {len(image_files)} frames")
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    print(f"Animation saved to {output_path}")

def create_multi_animation(directories, output_path, fps=5, dpi=100):
    """
    Creates a multi-panel animation from images in multiple directories.
    
    Args:
        directories (list): List of directories containing images to animate
        output_path (str): Path to save the animation file
        fps (int): Frames per second for the animation
        dpi (int): DPI for the output animation
    """
    # Get all image files from all directories
    all_files = []
    dir_names = []
    
    for directory in directories:
        image_files = glob.glob(os.path.join(directory, "*.png"))
        if not image_files:
            print(f"No images found in {directory}, skipping")
            continue
            
        # Sort files naturally
        image_files.sort(key=natural_sort_key)
        all_files.append(image_files)
        dir_names.append(os.path.basename(directory))
    
    if not all_files:
        print("No images found in any directory")
        return
    
    # Get common epochs (based on filenames)
    common_epochs = set()
    for files in all_files:
        epochs = []
        for f in files:
            epoch_match = re.search(r'(\d+)', os.path.basename(f))
            if epoch_match:
                epochs.append(epoch_match.group(1))
        common_epochs.update(epochs)
    
    common_epochs = sorted([int(e) for e in common_epochs])
    
    # Create mapping from epoch to file for each directory
    epoch_to_file = []
    for files in all_files:
        mapping = {}
        for f in files:
            epoch_match = re.search(r'(\d+)', os.path.basename(f))
            if epoch_match:
                mapping[int(epoch_match.group(1))] = f
        epoch_to_file.append(mapping)
    
    # Create figure with subplots
    n_panels = len(all_files)
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels*4, 4))
    
    # If only one panel, make axes iterable
    if n_panels == 1:
        axes = [axes]
    
    plt.tight_layout()
    
    # Setup first frame
    images = []
    titles = []
    
    for i, ax in enumerate(axes):
        ax.axis('off')
        
        # Get first image that exists for this directory
        for epoch in common_epochs:
            if epoch in epoch_to_file[i]:
                first_img_path = epoch_to_file[i][epoch]
                break
        else:
            # No images found for this directory
            ax.text(0.5, 0.5, "No images", ha='center', va='center')
            images.append(None)
            titles.append(None)
            continue
        
        img = plt.imread(first_img_path)
        im = ax.imshow(img, animated=True)
        images.append(im)
        
        title = ax.text(0.5, 1.02, f"{dir_names[i]} - Epoch: {common_epochs[0]}", 
                         size=10, ha="center", transform=ax.transAxes)
        titles.append(title)
    
    def update(frame):
        """Update function for the animation"""
        epoch = common_epochs[frame % len(common_epochs)]
        updated = []
        
        for i, (ax, im, title) in enumerate(zip(axes, images, titles)):
            if im is None:
                continue
                
            if epoch in epoch_to_file[i]:
                img_path = epoch_to_file[i][epoch]
                img = plt.imread(img_path)
                im.set_array(img)
                title.set_text(f"{dir_names[i]} - Epoch: {epoch}")
                updated.extend([im, title])
        
        return updated
    
    # Create animation
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(common_epochs), 
        interval=1000/fps, 
        blit=True
    )
    
    # Save animation
    print(f"Creating multi-panel animation with {len(common_epochs)} frames")
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    print(f"Animation saved to {output_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create animation from BiGAN reconstructions')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'multi'],
                        help='Animation mode: single (one directory) or multi (multiple directories)')
    parser.add_argument('--image_dir', type=str, 
                        default='./results/reconstruction/comparison',
                        help='Directory containing the images to animate (for single mode)')
    parser.add_argument('--dirs', type=str, nargs='+',
                        default=[
                            './results/reconstruction/comparison',
                            './results/reconstruction/fake',
                            './results/reconstruction/true'
                        ],
                        help='Directories to include in multi-panel animation (for multi mode)')
    parser.add_argument('--output', type=str, 
                        default='./results/animation.gif',
                        help='Path to save the animation file')
    parser.add_argument('--fps', type=int, default=2, 
                        help='Frames per second for the animation')
    parser.add_argument('--dpi', type=int, default=100, 
                        help='DPI for the output animation')
    return parser.parse_args()

def main():
    """Main function to create the animation."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.mode == 'single':
        create_animation(args.image_dir, args.output, args.fps, args.dpi)
    else:
        create_multi_animation(args.dirs, args.output, args.fps, args.dpi)

if __name__ == '__main__':
    main() 