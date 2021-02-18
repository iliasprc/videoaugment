import math
import random
import warnings

import PIL
import PIL.Image
import numpy as np
import torch
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


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.parameters = self.get_params(self.scale, self.ratio)

    @staticmethod
    def get_params(scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = 256 * 256

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            # print(aspect_ratio,target_area)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= 256 and h <= 256:
                i = random.randint(0, 256 - h)
                j = random.randint(0, 256 - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = 256 / 256
        if in_ratio < min(ratio):
            w = 256
            h = w / min(ratio)
        elif in_ratio > max(ratio):
            h = 256
            w = h * max(ratio)
        else:  # whole image
            w = 256
            h = 256
        i = (256 - h) // 2
        j = (256 - w) // 2

        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.parameters
        return TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
