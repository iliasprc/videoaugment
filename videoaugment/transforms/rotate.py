import random
import numpy as np
import PIL.Image
import PIL.Image
import torchvision.transforms.functional as TF
import skimage.transform

class Rotation:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, frame):
        """

        Args:
            frame (PIL.Image or np.array or torch.Tensor):

        Returns:  frame (PIL.Image or np.array or torch.Tensor)

        """
        if isinstance(frame, PIL.Image.Image):
            return TF.rotate(frame, self.angle)
        elif isinstance(frame,np.ndarray):
            skimage.transform.rotate(frame, self.angle)
        else:

            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(frame)))


class RandomRotation:
    """Rotate by random angle in specific range."""

    def __init__(self, angle_range=10):
        assert angle_range > 0, f'Given angle range{angle_range} not > 0'
        self.angle = random.randint(-angle_range, angle_range)

    def __call__(self, frame):
        """

        Args:
            frame (PIL.Image or np.array or torch.Tensor):

        Returns:  frame (PIL.Image or np.array or torch.Tensor)

        """
        if isinstance(frame, PIL.Image.Image):
            return TF.rotate(frame, self.angle)
        elif isinstance(frame,np.ndarray):
            skimage.transform.rotate(frame, self.angle)
        else:

            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(frame)))
