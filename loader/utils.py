import numpy as np
import cv2
import PIL.Image as Image
import glob
import skvideo.io
def plot_img(img, window_name='Frame'):
    # if isinstance(img,(Image)):
    img = np.array(img).astype(np.uint8)

    cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1000)
    cv2.destroyWindow(window_name)


def plot_video(video, window_name='Frame'):
    for img in video:
        img = np.array(img).astype(np.uint8)

        cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(100)
    cv2.destroyWindow(window_name)


def load_image(path):
    return Image.open(path)


def load_img_sequence(path,to_np=True):
    frames = sorted(glob.glob(f"{path}/*jpg"))
    video = []
    for f in frames:
        if to_np:
            video.append(np.asarray(load_image(f)))
        else:
            video.append(load_image(f))
    return video


def load_video(path):
    videogen = skvideo.io.vreader(path)
    video = []
    for frame in videogen:
        video.append(frame)
    return video




