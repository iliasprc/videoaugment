from transforms.crop import RandomCrop, RandomResizedCrop, CenterCrop
from transforms.flip import VerticalFlip, HorizontalFlip
from transforms.rotate import Rotation, RandomRotation
from transforms.general import ComposeSpatialTransforms, ComposeTemporalTransforms, VideoTransform, NpToTensor
from transforms.geometric import GaussianBlur
from transforms.intensity import Hue, RandomHue, Brightness, RandomBrightness, Contrast, RandomContrast, Saturation, \
    RandomSaturation, RandomColorAugment

from transforms.temporal_transform import TemporalElasticTransformation, TemporalDownsample, TemporalDownsample, \
    Upsample, \
    RandomTemporalDownsample, TemporalRandomCrop, TemporalCenterCrop

import numpy as np
from PIL import Image

import cv2

img_path = '/mnt/784C5F3A4C5EF1FC/PROJECTS/datasets/health1/health1_signer1_rep2_sentences/sentences0001/frame_0062.jpg'

frame = Image.open(img_path)
width, height = frame.size


def plot_img(img):
    # if isinstance(img,(Image)):
    img = np.array(img).astype(np.uint8)

    cv2.imshow('f', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1000)


plot_img(frame)

t = RandomCrop(400, img_size=(width, height))

plot_img(t(frame))

t = RandomResizedCrop(400, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.))

plot_img(t(frame))

t = CenterCrop(400)

plot_img(t(frame))

t = VerticalFlip()

plot_img(t(frame))

t = HorizontalFlip()

plot_img(t(frame))

t = Rotation(10)

plot_img(t(frame))


t = RandomRotation(90)

plot_img(t(frame))