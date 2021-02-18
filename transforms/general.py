# TODO
# video numpy array to tensor
# list of PIL images to tensor and numpy
# list of numpy to PIL
# list of

import torch


class NpToTensor(object):
    """Converts numpy array to tensor
    """

    def __call__(self, array):
        tensor = torch.from_numpy(array)
        return tensor


class ComposeSpatialTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame):
        for t in self.transforms:
            frame = t(frame)
        return frame


class ComposeTemporalTransforms(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, video):
        for t in self.transforms:
            video = t(video)
        return video


class VideoTransform(object):
    def __init__(self, spatial_transforms, temporal_transforms):
        self.spatial_transforms = ComposeSpatialTransforms(spatial_transforms)
        self.temporal_transforms = temporal_transforms

    def __call__(self, video):
        transformed_video = []
        for frame in video:
            frame = self.spatial_transforms(frame)
            transformed_video.append(frame)

        transformed_video = self.temporal_transforms(transformed_video)

        return transformed_video
