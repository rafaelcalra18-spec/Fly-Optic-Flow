import cv2
import numpy as np
import matplotlib.pyplot as plt
from step01_load import load_video

cap = cv2.VideoCapture('data/fan_normal.mov')
frames_start = 240
frames_end = 264
list_time = [] #list of times of each frame (when they appear in the video)

cap.set(cv2.CAP_PROP_POS_FRAMES, frames_start)

for i in range(frames_start, frames_end):
    cap.read()
    time = cap.get(cv2.CAP_PROP_POS_MSEC)
    list_time.append(time)


if __name__ == "__main__":
    frames, fps = load_video("data/fan_normal.mov")
     #24 consecutive frames from frames_start, each titled with its timestamp
    fig, axes = plt.subplots(4, 6, figsize=(15, 8))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(frames[frames_start + k], cmap="gray")
        ax.set_title(f"{list_time[k]:.2f} ms", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.show()