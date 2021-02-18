
import random
import torch
import PIL
import PIL.Image
import numpy as np
import torchvision.transforms.functional as TF

class CenterCrop(object):
    """

    """

    def __init__(self, size):

        self.size = size

    def __call__(self, frame):

        crop_h, crop_w = self.size
        if isinstance(frame, np.ndarray):
            im_h, im_w, im_c = frame.shape
        elif isinstance(frame, PIL.Image.Image):
            im_w, im_h = frame.size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image')

        if crop_w > im_w or crop_h > im_h:
            error_msg = (f'Initial image size should be larger then' +
                         f'cropped size but got cropped sizes : ' +
                         f'({crop_w}, {crop_h}) while initial image is ({im_w}, ' +
                         f'{im_h})')
            raise ValueError(error_msg)

        w1 = int(round((im_w - crop_w) / 2.))
        h1 = int(round((im_h - crop_h) / 2.))

        if isinstance(frame, np.ndarray):
            return frame[h1:h1 + crop_h, w1:w1 + crop_w, :]
        elif isinstance(frame, PIL.Image.Image):
            return frame.crop((w1, h1, w1 + crop_w, h1 + crop_h))
        elif isinstance(frame, torch.Tensor):
            return TF.center_crop(frame)


class RandomCrop(object):
    """
    Extract random crop of the video.
    Args:
        crop_size (sequence or int): Desired output size for the crop in format (h, w).
        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, crop_size, img_size):
        if isinstance(crop_size, int):
            if crop_size < 0:
                raise ValueError('If size is a single number, it must be positive')
            crop_size = (crop_size, crop_size)
        else:
            if len(crop_size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.crop_size = crop_size
        ## select same crop for video
        self.w1 = random.randint(0, img_size - self.crop_size[0])
        self.h1 = random.randint(0, img_size - self.crop_size[1])

    def __call__(self, frame):
        crop_h, crop_w = self.crop_size
        if isinstance(frame, np.ndarray):
            im_h, im_w, im_c = frame.shape
        elif isinstance(frame, PIL.Image.Image):
            im_w, im_h = frame.size
        elif isinstance(frame, torch.Tensor):
             im_h, im_w, im_c = frame.shape
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(frame)))

        if crop_w > im_w or crop_h > im_h:
            error_msg = ('Initial image size should be larger then' +
                         'cropped size but got cropped sizes : ' +
                         '({w}, {h}) while initial image is ({im_w}, ' +
                         '{im_h})'.format(im_w=im_w, im_h=im_h, w=crop_w,
                                          h=crop_h))
            raise ValueError(error_msg)

        if isinstance(frame, np.ndarray):
            return frame[self.h1:self.h1 + crop_h, self.w1:self.w1 + crop_w, :]
        elif isinstance(frame, PIL.Image.Image):
            return frame.crop((self.w1, self.h1, self.w1 + crop_w, self.h1 + crop_h))
        elif isinstance(frame, torch.Tensor):
            return frame[self.h1:self.h1 + crop_h, self.w1:self.w1 + crop_w, :]
