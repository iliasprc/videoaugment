import PIL.Image
import PIL.Image
import numpy as np
import torch
import torchvision.transforms.functional as TF


class HorizontalFlip(object):
    """
    Horizontally flip the video.
    """

    def __call__(self, frame):
        if isinstance(frame, np.ndarray):
            return np.fliplr(frame)
        elif isinstance(frame, PIL.Image.Image):

            return frame.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif isinstance(frame, torch.Tensor):
            return TF.hflip(frame)


        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(frame)))


class VerticalFlip(object):
    """
    Vertical flip the video.
    """

    def __call__(self, frame):
        if isinstance(frame, np.ndarray):
            return np.flipud(frame)
        elif isinstance(frame, PIL.Image.Image):

            return frame.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        elif isinstance(frame, torch.Tensor):
            return TF.vflip(frame)


        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            ' but got list of {0}'.format(type(frame)))
