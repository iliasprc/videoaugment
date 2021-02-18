import math
import random

import numpy as np


class TemporalDownsample(object):
    """
    Downsample video  by deleting uniformly some of its frames
    Args:
       sampling_factor (float):
    """

    def __init__(self, sampling_factor=1.0):
        self.sampling_factor = sampling_factor

    def __call__(self, video):
        return_ind = [int(i) for i in np.linspace(1, len(video), num=int(self.sampling_factor * len(video)))]
        return [video[i - 1] for i in return_ind]


class RandomTemporalDownsample(object):
    """
    Downsample video  by deleting randomly some of its frames
    Args:
       sampling_factor (float):
    """

    def __init__(self, sampling_factor=1.0):
        self.sampling_factor = sampling_factor

    def __call__(self, video):
        return_ind = list(range(len(video)))
        return_ind = sorted(random.sample(return_ind, k=int(self.sampling_factor * len(video))))

        return [video[i - 1] for i in return_ind]


class Upsample(object):
    """
    Temporally upsampling a video by duplicating uniformly some of its frames.
    Args:
        ratio (float): Upsampling ratio in [1.0 < ratio < infinity].
    """

    def __init__(self, ratio=1.0):
        if ratio < 1.0:
            raise TypeError('ratio should be 1.0 < ratio. ' +
                            'Please use downsampling for ratio <= 1.0')
        self.ratio = ratio

    def __call__(self, clip):
        nb_return_frame = int(self.ratio * len(clip))
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=nb_return_frame)]

        return [clip[i - 1] for i in return_ind]


class TemporalCenterCrop(object):
    """
    Temporally crop the given frame indices at a center.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        center_index = len(clip) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(clip))

        out = clip[begin_index:end_index]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class TemporalRandomCrop(object):
    """
    Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, clip):
        rand_end = max(0, len(clip) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(clip))

        out = clip[begin_index:end_index]

        for img in out:
            if len(out) >= self.size:
                break
            out.append(img)

        return out


class TemporalElasticTransformation(object):
    """
    Stretches or schrinks a video at the beginning, end or middle parts.
    In normal operation, augmenter stretches the beggining and end, schrinks
    the center.
    In inverse operation, augmenter shrinks the beggining and end, stretches
    the center.
    """

    def __call__(self, clip):
        nb_images = len(clip)
        new_indices = self._get_distorted_indices(nb_images)
        return [clip[i] for i in new_indices]

    def _get_distorted_indices(self, nb_images):
        inverse = random.randint(0, 1)

        if inverse:
            scale = random.random()
            scale *= 0.21
            scale += 0.6
        else:
            scale = random.random()
            scale *= 0.6
            scale += 0.8

        frames_per_clip = nb_images

        indices = np.linspace(-scale, scale, frames_per_clip).tolist()
        if inverse:
            values = [math.atanh(x) for x in indices]
        else:
            values = [math.tanh(x) for x in indices]

        values = [x / values[-1] for x in values]
        values = [int(round(((x + 1) / 2) * (frames_per_clip - 1), 0)) for x in values]
        return values


class TemporalScale(object):

    def __init__(self, num_of_frames):
        self.num_of_frames = num_of_frames
    def __call__(self, clip):
        return_ind = [int(i) for i in np.linspace(1, len(clip), num=self.num_of_frames)]
        return [clip[i - 1] for i in return_ind]