import PIL
import PIL.Image
import numpy as np
import scipy.ndimage
import torch
import torchvision.transforms.functional as TF


class GaussianBlur(object):
    """

    Args:
        kernel_size ():
        sigma ():
    """

    def __init__(self, kernel_size=3, sigma=0.001):

        if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, list)):
            kernel_size = (kernel_size, kernel_size)
        ks = kernel_size[0]
        if ks <= 0 or ks % 2 == 0:
            raise ValueError("Kernel size value should be an odd and positive number.")
        if not (isinstance(sigma, tuple) or isinstance(sigma, list)):
            sigma = (sigma, sigma)
        s = sigma[0]
        if s < 0:
            raise ValueError("Sigma value should be a  positive number.")
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, frame):
        """

        Args:
            frame (PIL.Image or np.array or torch.Tensor):

        Returns:  frame (PIL.Image or np.array or torch.Tensor)

        """
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.gaussian_blur(frame, self.kernel_size, self.sigma)
        elif isinstance(frame, np.ndarray):
            return scipy.ndimage.gaussian_filter(frame, sigma=self.sigma, order=0)
