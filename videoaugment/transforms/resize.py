import random
import numpy as np
import PIL.Image
import PIL.Image
import torchvision.transforms.functional as TF
import skimage.transform


class Resize(object):

    def __init__(self, size):
        """
        Resize a frame to the desired size
        Args:
            size (tuple or int): output size of image
        """
        if not (isinstance(size, (tuple, list))):
            size = (size, size)
        self.size = size

    def __call__(self, frame):
        """

        Args:
            frame (PIL.Image or np.array or torch.Tensor):

        Returns: resized frame (PIL.Image or np.array or torch.Tensor)

        """
        if isinstance(frame, PIL.Image.Image):
            return frame.resize(self.size)
        elif isinstance(frame, np.ndarray):
            frame = PIL.Image.fromarray(frame)
            return frame.resize(self.size)
