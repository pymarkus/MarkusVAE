import numpy as np
# import cv2
import os
import matplotlib.pyplot as plt
from torchvision import transforms

def video_gen(meta):
    "meta is list of images"
    video_name = 'video.avi'

    frame = cv2.imread(meta[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in meta:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

def vizual_meta(imgs):
    n = len(imgs)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 20))

    for i, image in enumerate(imgs):
        axes[i].imshow(transforms.ToPILImage()(image[0]))