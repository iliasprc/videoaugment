from transforms.crop import RandomCrop, RandomResizedCrop, CenterCrop
from transforms.flip import VerticalFlip, HorizontalFlip, RandomVerticalFlip, RandomHorizontalFlip
from transforms.rotate import Rotation, RandomRotation
from transforms.resize import Resize
from transforms.geometric import GaussianBlur
from transforms.general import ComposeSpatialTransforms, ComposeTemporalTransforms, VideoTransform, NumpyToTensor, \
    PILToTensor, Normalize,RearrangeTensor,VideoToTensor

from transforms.intensity import Hue, RandomHue, Brightness, RandomBrightness, Contrast, RandomContrast, Saturation, \
    RandomSaturation, RandomColorAugment, Rescale

from transforms.temporal_transform import TemporalElasticTransformation, TemporalDownsample, \
    Upsample, \
    RandomTemporalDownsample, TemporalRandomCrop, TemporalCenterCrop

from loader.utils import load_image, plot_video, plot_img, load_img_sequence
import numpy as np
from PIL import Image

import cv2

import glob

img_path = '/mnt/784C5F3A4C5EF1FC/PROJECTS/datasets/health1/health1_signer1_rep2_sentences/sentences0001/frame_0062.jpg'

frame = load_image(img_path)
width, height = frame.size

video = load_img_sequence(
    '/mnt/784C5F3A4C5EF1FC/PROJECTS/datasets/health1/health1_signer1_rep2_sentences/sentences0001')
print(len(video))

t = VideoTransform(
    spatial_transforms=[CenterCrop(400), RandomColorAugment(brightness=0.2, contrast=0.2, hue=0.2, saturation=0.2),
                        RandomRotation(10), RandomHorizontalFlip(0.5) ],
    temporal_transforms=[RandomTemporalDownsample(0.2), TemporalElasticTransformation()])
v = t(video)
print(len(v))
plot_video(v, window_name="TemporalElasformation")
print(f" TEST DONE")

video = np.array(video)
t = VideoTransform(
    spatial_transforms=[CenterCrop(450), Resize(256), RandomCrop(crop_size=224, img_size=256),
                        RandomColorAugment(brightness=0.2, contrast=0.2, hue=0.2, saturation=0.2),
                        RandomRotation(10), RandomHorizontalFlip(0.5)],
    temporal_transforms=[RandomTemporalDownsample(0.7), TemporalElasticTransformation()])
v = t(video)

VideoToTensor()(v)
print(len(v))
plot_video(v, window_name="TemporalElasticTransformation")


t = VideoTransform(
    spatial_transforms=[CenterCrop(450), Resize(256), RandomCrop(crop_size=224, img_size=256),
                        RandomColorAugment(brightness=0.2, contrast=0.2, hue=0.2, saturation=0.2),
                        RandomRotation(10), RandomHorizontalFlip(0.5), Rescale(1./255.0), PILToTensor(),RearrangeTensor(),
                        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))],
    temporal_transforms=[RandomTemporalDownsample(0.7), TemporalElasticTransformation(),VideoToTensor()])
v = t(video)
