import random

import PIL
import PIL.Image
import torch
import torchvision.transforms.functional as TF



class Brightness:
    """Change brightness of frame."""

    def __init__(self, brightness):
        self.brightness = brightness

    def __call__(self, frame):
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_brightness(frame, brightness_factor=self.brightness)

        else:

            raise TypeError('Expected  PIL.Image or Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class RandomBrightness:
    """Change randomly brightness of frame."""

    def __init__(self, abs_brightness=0.01):
        self.brightness = 1 + random.uniform(-abs(abs_brightness), abs(abs_brightness))

    def __call__(self, frame):
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_brightness(frame, brightness_factor=self.brightness)

        else:

            raise TypeError('Expected  PIL.Image or Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class Hue:
    """
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`."""

    def __init__(self, hue):
        assert abs(hue) <= 0.5, f'hue value is {hue}, it should be <=0.5'
        self.hue = hue

    def __call__(self, frame):
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_hue(frame, hue_factor=self.hue)

        else:

            raise TypeError('Expected  PIL.Image or torch.Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class RandomHue:
    """
    Change randomly hue value
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`."""

    def __init__(self, hue_range=0.5):
        assert abs(hue) <= 0.5, f'hue value is {hue}, it should be <=0.5'
        self.hue = random.uniform(0, 1) * hue_range

    def __call__(self, frame):
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_hue(frame, hue_factor=self.hue)
        else:

            raise TypeError('Expected  PIL.Image or torch.Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class Contrast:
    """Change contrast value  of frame.
      contrast (float): 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    """

    def __init__(self, contrast=1):
        """

        Args:
            contrast ():
        """
        assert 0 <= contrast <= 2, f'contrast should be in the range of [0,2] , given value was {contrast}'
        self.contrast = contrast

    def __call__(self, frame):
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_contrast(frame, contrast_factor=self.contrast)

        else:

            raise TypeError('Expected  PIL.Image or Tensor' +
                            ' but got list of {0}'.format(type(frame)))


class RandomContrast:
    """Change randomly brightness of frame."""

    def __init__(self, contrast=1):
        # assert 0 <= contrast <= 2, f'contrast should be in the range of [0,2] , given value was {contrast}'
        self.contrast = random.uniform(0, 2)

    def __call__(self, frame):
        if isinstance(frame, PIL.Image.Image) or isinstance(frame, torch.Tensor):
            return TF.adjust_contrast(frame, contrast_factor=self.contrast)

        else:

            raise TypeError('Expected  PIL.Image or Tensor' +
                            ' but got list of {0}'.format(type(frame)))
