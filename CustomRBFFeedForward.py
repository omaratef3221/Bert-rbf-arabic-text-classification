import torch
import torch.nn as nn
from rbf_layer import RBFLayer  # Assuming RBFLayer is your custom RBF implementation

def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)

# Gaussian RBF
def rbf_gaussian(x):
    return (-x.pow(2)).exp()

def multiquadric_kernel(eps_r):
    return torch.sqrt(eps_r ** 2 + 1)

def linear_kernel(x):
    return x  # Linear kernel simply returns the input

def rbf_laplacian(x):
    return (-x.abs()).exp()

class CustomRBFFeedForward(nn.Module):
    def __init__(self, in_features, out_features, num_kernels, kernel_name='linear'):
        super(CustomRBFFeedForward, self).__init__()
        # RBFLayer from the given implementation
        self.rbf_layer = RBFLayer(
            in_features_dim=in_features,  # Input size (e.g., 5120)
            num_kernels=num_kernels,  # Number of kernels in the RBF layer (can be tuned)
            out_features_dim=out_features,  # Output size (e.g., 5120)
            radial_function=rbf_laplacian if kernel_name == 'laplacian' else  multiquadric_kernel,  # Use the Gaussian RBF
            norm_function=l_norm  # Use Euclidean norm
        )

    def forward(self, x):
        # Apply the RBF layer to the input x
        return self.rbf_layer(x)