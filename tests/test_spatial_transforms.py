from transforms.crop import RandomCrop, RandomResizedCrop, CenterCrop
from transforms.flip import VerticalFlip, HorizontalFlip
from transforms.rotate import Rotation, RandomRotation
from transforms.geometric import GaussianBlur
from transforms.general import ComposeSpatialTransforms, ComposeTemporalTransforms, VideoTransform, NpToTensor

from transforms.intensity import Hue, RandomHue, Brightness, RandomBrightness, Contrast, RandomContrast, Saturation, \
    RandomSaturation, RandomColorAugment

from transforms.temporal_transform import TemporalElasticTransformation, TemporalDownsample, TemporalDownsample, \
    Upsample, \
    RandomTemporalDownsample, TemporalRandomCrop, TemporalCenterCrop

import numpy as np
from PIL import Image

import cv2

import glob

def load_image(path):
    return Image.open(path)

def load_img_sequence(path):
    frames = sorted(glob.glob(f"{path}/*jpg"))
    video = []
    for f in frames:
        video.append(load_image(f))
    return video


img_path = '/mnt/784C5F3A4C5EF1FC/PROJECTS/datasets/health1/health1_signer1_rep2_sentences/sentences0001/frame_0062.jpg'

frame = load_image(img_path)
width, height = frame.size


def plot_img(img,window_name='Frame'):
    # if isinstance(img,(Image)):
    img = np.array(img).astype(np.uint8)

    cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1000)
    cv2.destroyWindow(window_name)

#
# plot_img(frame)
#
# t = RandomCrop(400, img_size=(width, height))
#
# plot_img(t(frame),t.__class__.__name__)
#
# t = RandomResizedCrop(400, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.))
#
# plot_img(t(frame),t.__class__.__name__)
#
# t = CenterCrop(400)
#
# plot_img(t(frame),t.__class__.__name__)
#
# t = VerticalFlip()
#
# plot_img(t(frame),t.__class__.__name__)
#
# t = HorizontalFlip()
#
# plot_img(t(frame),t.__class__.__name__)
#
# t = Rotation(10)
#
# plot_img(t(frame),t.__class__.__name__)
#
#
# t = RandomRotation(90)
#
# plot_img(t(frame),t.__class__.__name__)
#
# t = Hue(hue=0.1)
# plot_img(t(frame),t.__class__.__name__)


# t = Brightness(brightness=1.5)
# plot_img(t(frame),t.__class__.__name__)
#
#
# t = Contrast(contrast=1.5)
# plot_img(t(frame),t.__class__.__name__)


t = RandomColorAugment(brightness=0.2,contrast=0.2,hue=0.2,saturation=0.2)
plot_img(t(frame),t.__class__.__name__)