import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example tensor
def pca(tensor):
    with torch.no_grad():
        # Step 1: Reshape the tensor
        data = tensor.view(256, -1).T  # [16384, 256] where 16384 = 128x128

        # Step 2: Implement PCA using PyTorch
        # Center the data by subtracting the mean
        mean = torch.mean(data, dim=0)
        data_centered = data - mean

        # Compute the covariance matrix
        cov_matrix = torch.mm(data_centered.T, data_centered) / (data_centered.size(0) - 1)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)  # Use eigh for symmetric matrices, which are the covariance matrices

        # Sort eigenvectors by eigenvalues in descending order
        idxs = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, idxs]

        # Select the top 3 eigenvectors
        projection_matrix = eigenvectors[:, :3]

        # Project the data onto the top 3 principal components
        data_reduced = torch.mm(data_centered, projection_matrix)

        # Step 3: Visualize the Result
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = data_reduced[:, 0]  # Color by the first principal component
        scatter = ax.scatter(data_reduced[:, 0].cpu().numpy(), data_reduced[:, 1].cpu().numpy(), c=colors.cpu().numpy(), cmap='viridis')
        fig.colorbar(scatter, ax=ax, label='First principal component intensity')

        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        plt.title('2D PCA Visualization')
        plt.show()


def pca_to_image(tensor, save_path, draw=True):
    with torch.no_grad():
        # Step 1: Reshape the tensor
        data = tensor.view(tensor.shape[1], -1).T  # Reshape from [1, 256, 256, 256] to [65536, 256]

        # Step 2: Implement PCA using PyTorch
        # Center the data by subtracting the mean
        mean = torch.mean(data, dim=0)
        data_centered = data - mean

        # Compute the covariance matrix
        cov_matrix = torch.mm(data_centered.T, data_centered) / (data_centered.size(0) - 1)

        # Eigenvalue decomposition using eigh, since the covariance matrix is symmetric
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues in descending order
        idxs = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, idxs]

        # Select the top 3 eigenvectors (principal components)
        projection_matrix = eigenvectors[:, :3]

        # Project the data onto the top 3 principal components
        data_reduced = torch.mm(data_centered, projection_matrix)

        # Normalize the projected data to be in the range [0, 1] for RGB image creation
        data_reduced = (data_reduced - data_reduced.min()) / (data_reduced.max() - data_reduced.min())

        # Step 3: Reshape back to 256x256x3 to visualize as an RGB image
        image_rgb = data_reduced.view(tensor.shape[2], tensor.shape[2], 3).detach().cpu().numpy()

        if draw:
            # Visualize as an RGB image
            plt.imshow(image_rgb)
            plt.axis('off')  # Turn off axis numbers and ticks
            plt.title('PCA Components as RGB Image')
            plt.savefig(save_path)
        return image_rgb

def normalize_image(tensor):
    # Assuming tensor is a PyTorch tensor of shape [channels, height, width]
    # Clone the tensor to avoid changing the original data
    tensor = tensor.clone()
    
    # Find the minimum and maximum values in the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    
    # Normalize the tensor to [0, 1]
    tensor_norm = (tensor - min_val) / (max_val - min_val)
    
    return tensor_norm

def pca_diff(tensor1, tensor2, feature1, feature2, save_path):
    with torch.no_grad():
        image_pca = pca_to_image(feature1 - feature2, save_path="", draw=False)
        fig, axes = plt.subplots(1, 4)
        axes[0].imshow(normalize_image(tensor1)[0].permute(1, 2, 0).detach().cpu().numpy())
        axes[1].imshow(normalize_image(tensor2)[0].permute(1, 2, 0).detach().cpu().numpy())
        axes[2].imshow(normalize_image(tensor1 - tensor2)[0].permute(1, 2, 0).detach().cpu().numpy())
        axes[3].imshow(image_pca)
        plt.axis('off')
        plt.savefig(save_path)