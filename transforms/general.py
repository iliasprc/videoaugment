# TODO
# video numpy array to tensor
# list of PIL images to tensor and numpy
# list of numpy to PIL
# list of
import PIL
import PIL.Image
import torch
import numpy as np
import torchvision.transforms.functional as TF

from einops import rearrange


class RearrangeTensor:


    def __call__(self, input_tensor):
       return rearrange(input_tensor, 'h w c -> c h w')

class PILToNumpy:
    def __call__(self, image):
        """

        Args:
            image ():

        Returns:

        """
        return np.asarray(image)


class PILToTensor:
    def __call__(self, image):
        """

        Args:
            image ():

        Returns:

        """
        return torch.from_numpy(np.asarray(image))


class NumpyToTensor(object):
    """Converts numpy array to tensor
    """

    def __call__(self, array):
        """

        Args:
            array ():

        Returns:

        """
        tensor = torch.from_numpy(array)
        return tensor


class Normalize(object):

    def __init__(self, mean, std):
        """

        Args:
            mean ():
            std ():
        """
        self.mean = mean
        self.std = std

    def __call__(self, frame):
        """

        Args:
            frame ():

        Returns:

        """
        if isinstance(frame, torch.Tensor):
            return TF.normalize(frame, self.mean, self.std, False)
        elif isinstance(frame, np.ndarray):
            return TF.normalize(torch.from_numpy(frame), self.mean, self.std, False)
        elif isinstance(frame, PIL.Image.Image):
            return TF.normalize(torch.from_numpy(np.asarray(frame)), self.mean, self.std, False)


class ComposeSpatialTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms
        if not isinstance(self.transforms, list):
            self.transforms = [self.transforms]

    def __call__(self, frame):
        # print(self.transforms)
        for t in self.transforms:
            if t != None:
                frame = t(frame)
        return frame


class ComposeTemporalTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms
        if not isinstance(self.transforms, list):
            self.transforms = [self.transforms]

    def __call__(self, video):
        for t in self.transforms:
            video = t(video)
        return video

class VideoToTensor:


    def __call__(self, video):

        vid = torch.stack(video,dim=1)
        print(vid.shape)
        return vid
class VideoTransform(object):
    def __init__(self, spatial_transforms, temporal_transforms):
        self.spatial_transforms = ComposeSpatialTransforms(spatial_transforms)
        self.temporal_transforms = ComposeTemporalTransforms(temporal_transforms)

    def __call__(self, video):
        transformed_video = []
        for frame in video:
            # if self.spatial_transforms != None:
            frame = self.spatial_transforms(frame)
            transformed_video.append(frame)
        if self.temporal_transforms != None:
            transformed_video = self.temporal_transforms(transformed_video)


        return transformed_video
