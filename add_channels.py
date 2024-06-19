import os
import argparse
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import mmcv
import torch.nn.functional as F
import torch.nn as nn

sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

sobel_x_2 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1, bias=False)
sobel_x_2.weight.data = sobel_x_kernel

sobel_x_kernel = sobel_x_kernel.repeat(3, 1, 1, 1)
sobel_x = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, groups=3, bias=False)
sobel_x.weight.data = sobel_x_kernel


sobel_x_3 = nn.Conv2d(3, 1, kernel_size=(3, 3), padding=1, bias=False)
sobel_x_3.weight.data = sobel_x_kernel.reshape((1, 3, 3, 3))

# Sobel filter for detecting vertical gradients
sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

sobel_y_2 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1, bias=False)
sobel_y_2.weight.data = sobel_y_kernel

sobel_y_kernel = sobel_y_kernel.repeat(3, 1, 1, 1)
sobel_y = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, groups=3, bias=False)
sobel_y.weight.data = sobel_y_kernel


sobel_y_3 = nn.Conv2d(3, 1, kernel_size=(3, 3), padding=1, bias=False)
sobel_y_3.weight.data = sobel_y_kernel.reshape((1, 3, 3, 3))

# Laplacian kernel for detecting edges
laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

laplacian_2 = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1, bias=False)
laplacian_2.weight.data = laplacian_kernel

laplacian_kernel = laplacian_kernel.repeat(3, 1, 1, 1)
laplacian = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, groups=3, bias=False)
laplacian.weight.data = laplacian_kernel

laplacian_3 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False)
laplacian_3.weight.data = laplacian_kernel.reshape((1, 3, 3, 3))

def compute_gradients_and_laplacian(image, comp):
    with torch.no_grad():
        # Sobel filters for gradient computation
        global sobel_x , sobel_y, laplacian
        

        grad_x = sobel_x(image) 
        grad_y = sobel_y(image)
        laplacian_ = laplacian(image)
        if comp == 'True':
            # Concatenate the results and then average over the channel dimension
            grad_x = grad_x.mean(dim=1)
            grad_y = grad_y.mean(dim=1)
            laplacian_ = laplacian_.mean(dim=1)
            return torch.cat([grad_x[:, None, :, :], grad_y[:, None, :, :], laplacian_[:, None, :, :]], dim=1)

        return torch.cat([grad_x, grad_y, laplacian_], dim=1)


def compute_gradients_and_laplacian_v2(image, comp):
    with torch.no_grad():
        # Sobel filters for gradient computation
        global sobel_x_2 , sobel_y_2, laplacian_2
        
        image = image.reshape(12, 1, 512, 512)
        grad_x = sobel_x_2(image) 
        grad_y = sobel_y_2(image)
        laplacian_ = laplacian_2(image)

        grad_x = grad_x.reshape(4, 3, 512, 512)
        grad_y = grad_y.reshape(4, 3, 512, 512)
        laplacian_ = laplacian_.reshape(4, 3, 512, 512) 
        if comp == 'True':
            # Concatenate the results and then average over the channel dimension
            grad_x = grad_x.mean(dim=1)
            grad_y = grad_y.mean(dim=1)
            laplacian_ = laplacian_.mean(dim=1)
            return torch.cat([grad_x[:, None, :, :], grad_y[:, None, :, :], laplacian_[:, None, :, :]], dim=1)

        return torch.cat([grad_x, grad_y, laplacian_], dim=1)


def compute_gradients_and_laplacian_v3(image, comp):
    with torch.no_grad():
        # Sobel filters for gradient computation
        global sobel_x_3 , sobel_y_3, laplacian_3
    
        grad_x = sobel_x_3(image) 
        grad_y = sobel_y_3(image)
        laplacian_ = laplacian_3(image)

        return torch.cat([grad_x, grad_y, laplacian_], dim=1)


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
        new_tensor = torch.cat([*sin_channels, *cos_channels], dim=1)
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
        
        # image = Image.fromarray(img)
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
    parser.add_argument("--input_dir", default='/data/split_ss_dota/train/images', type=str, help="Directory containing the input images (.png files).")
    parser.add_argument("--output_dir", default='/data/split_ss_dota/single_image/fourier', type=str, help="Directory where the output tensors will be saved.")
    parser.add_argument("--nmin", default=1, type=int, help="Minimum power of 2 for the sinusoidal frequency.")
    parser.add_argument("--nmax", default=5, type=int, help="Maximum power of 2 for the sinusoidal frequency.")
    parser.add_argument("--size", default=512, type=int, help="size of training data")
    parser.add_argument("--type", default='grad', type=str, help="fourier/grad")
    parser.add_argument("--comp", default='False', type=str, help="compress grad or not")
    args = parser.parse_args()

    # process_and_save_images(args.input_dir, args.output_dir, args.nmin, args.nmax, args.size, args.type, args.comp)


    import time
    
    tens = [torch.rand((4, 3, 512, 512)) for _ in range(50)]

    start_fourier = time.time()
    for i in range(50):
        ch = create_fourier_channels(tens[i], 7, 8)
        res = torch.cat([tens[i],ch], dim=1)
        print(res.shape)
    end_fourier = time.time()
    print(f"avg time - {(end_fourier - start_fourier) / 50}")

    # start_grad = time.time()
    # for i in range(50):
    #     ch = compute_gradients_and_laplacian(tens[i], False)
    #     res = torch.cat([tens[i],ch], dim=1)
    # end_grad = time.time()
    # print(f"avg time - {(end_grad - start_grad) / 50}")

    # start_grad = time.time()
    # for i in range(50):
    #     ch = compute_gradients_and_laplacian(tens[i], 'True')
    #     res = torch.cat([tens[i],ch], dim=1)
    # end_grad = time.time()
    # print(f"avg time - {(end_grad - start_grad) / 50}")

    # start_grad = time.time()
    # for i in range(50):
    #     ch = compute_gradients_and_laplacian_v2(tens[i], False)
    #     res = torch.cat([tens[i],ch], dim=1)
    # end_grad = time.time()
    # print(f"avg time - {(end_grad - start_grad) / 50}")

    # start_grad = time.time()
    # for i in range(50):
    #     ch = compute_gradients_and_laplacian_v2(tens[i], 'True')
    #     res = torch.cat([tens[i],ch], dim=1)
    # end_grad = time.time()
    # print(f"avg time - {(end_grad - start_grad) / 50}")


    # start_grad = time.time()
    # for i in range(50):
    #     ch = compute_gradients_and_laplacian_v3(tens[i], False)
    #     res = torch.cat([tens[i],ch], dim=1)
    # end_grad = time.time()
    # print(f"avg time - {(end_grad - start_grad) / 50}")


if __name__ == "__main__":
    main()
