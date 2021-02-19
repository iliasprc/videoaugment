import skvideo.io

import videoaugment.transforms as VA
from videoaugment.base.baseloader import BaseVideoDataset


class VideoDataset(BaseVideoDataset):

    def __init__(self, config, args, mode, classes):
        """

        Args:
            config ():
            args ():
            mode ():
            classes ():
        """
        super().__init__(config, args, mode, classes)

        self.do_augmentation = (mode == 'train')
        self.videos = list()
        self.labels = list()

    def video_loader(self, index):
        if self.do_augmentation:
            st = VA.ComposeSpatialTransforms(
                transforms=[VA.CenterCrop(512), VA.Resize(256),
                            VA.RandomCrop(crop_size=self.config.dim[0], img_size=256),
                            VA.RandomColorAugment(brightness=0.2, contrast=0.2, hue=0.2,
                                                  saturation=0.2),
                            VA.RandomRotation(10), VA.RandomHorizontalFlip(0.5), VA.Rescale(1. / 255.0),
                            VA.PILToTensor(),
                            VA.RearrangeTensor(),
                            VA.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            tt = VA.ComposeTemporalTransforms(
                transforms=[VA.RandomTemporalDownsample(0.5), VA.TemporalElasticTransformation(),
                            VA.TemporalScale(num_of_frames=self.config.num_of_frames), VA.VideoToTensor()])
            videogen = skvideo.io.vreader(self.videos[index])
            video = []
            for frame in videogen:
                augmented_frame = st(frame)
                video.append(augmented_frame)
            video_tensor = tt(video)
