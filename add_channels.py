import os
import argparse
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import mmcv
import torch.nn.functional as F


def compute_gradients_and_laplacian(image, comp):
    with torch.no_grad():
        # Sobel filters for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Laplacian kernel
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Apply the convolution to each channel separately
        grad_x = []
        grad_y = []
        laplacian = []
        
        for i in range(image.size(0)):  # Loop over each channel
            grad_x.append(F.conv2d(image[i:i+1, :, :], sobel_x, padding=1))
            grad_y.append(F.conv2d(image[i:i+1, :, :], sobel_y, padding=1))
            laplacian.append(F.conv2d(image[i:i+1, :, :], laplacian_kernel, padding=1))
        
        if comp == 'True':
            # Concatenate the results and then average over the channel dimension
            grad_x = torch.cat(grad_x, dim=0).mean(dim=0)
            grad_y = torch.cat(grad_y, dim=0).mean(dim=0)
            laplacian = torch.cat(laplacian, dim=0).mean(dim=0)
            return torch.cat([grad_x[None, :, :], grad_y[None, :, :], laplacian[None, :, :]], dim=0)
        else:
            grad_x = torch.cat(grad_x, dim=0)
            grad_y = torch.cat(grad_y, dim=0)
            laplacian = torch.cat(laplacian, dim=0)
        return torch.cat([grad_x, grad_y, laplacian], dim=0)


def create_fourier_channels(img_tensor, nmin, nmax):
    with torch.no_grad():
        # Expand img_tensor from 0 to 1 to a more suitable range for Fourier transformations
        img_tensor = img_tensor * 2 * np.pi  # Scale pixel values for trigonometric transformations
        
        # Create empty lists to collect new channels
        sin_channels, cos_channels = [], []

        # Apply transformations
        for n in range(nmin, nmax + 1):
            sin_channel = torch.sin(2 ** n * img_tensor)
            cos_channel = torch.cos(2 ** n * img_tensor)
            sin_channels.append(sin_channel)
            cos_channels.append(cos_channel)

        # Concatenate original and new channels
        new_tensor = torch.cat([*sin_channels, *cos_channels], dim=0)
        return new_tensor

def process_and_save_images(input_dir, output_dir, nmin, nmax, size, arg_type, comp):
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List all PNG files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    # Transformation to tensor
    to_tensor = transforms.ToTensor()
    
    for filename in image_files:
        # Load image
        img_path = os.path.join(input_dir, filename)
        image = np.array(Image.open(img_path).convert('RGB'))

        img, scale_factor = mmcv.imrescale(
            image,
            (size, size),
            interpolation='bilinear',
            return_scale=True,
            backend='cv2')
        
        image = Image.fromarray(img)
        # Convert image to tensor
        img_tensor = to_tensor(image)
        
        if arg_type == 'fourier':
            # Append Fourier sinusoidal channels
            added_channels_tensor = create_fourier_channels(img_tensor, nmin, nmax)
        elif arg_type == 'grad':
            added_channels_tensor = compute_gradients_and_laplacian(img_tensor, comp)
        # Save the tensor to the disk
        if arg_type == 'grad':
            cmp = 'compress' if comp == 'True' else 'full'
            save_pt = f"{arg_type}_{cmp}_{filename.replace('.png', '.pt')}"
        else:
            save_pt = f"{arg_type}_{filename.replace('.png', '.pt')}"
        output_path = os.path.join(output_dir, save_pt)
        torch.save(added_channels_tensor, output_path)
        print(added_channels_tensor.shape)
        print(f"Processed and saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Append Fourier sinusoidal channels to images and save them.")
    parser.add_argument("--input_dir", default='/data/split_ss_dota/single_image', type=str, help="Directory containing the input images (.png files).")
    parser.add_argument("--output_dir", default='/data/split_ss_dota/single_image/fourier', type=str, help="Directory where the output tensors will be saved.")
    parser.add_argument("--nmin", default=1, type=int, help="Minimum power of 2 for the sinusoidal frequency.")
    parser.add_argument("--nmax", default=5, type=int, help="Maximum power of 2 for the sinusoidal frequency.")
    parser.add_argument("--size", default=512, type=int, help="size of training data")
    parser.add_argument("--type", default='grad', type=str, help="fourier/grad")
    parser.add_argument("--comp", default='False', type=str, help="compress grad or not")
    args = parser.parse_args()

    process_and_save_images(args.input_dir, args.output_dir, args.nmin, args.nmax, args.size, args.type, args.comp)

if __name__ == "__main__":
    main()
