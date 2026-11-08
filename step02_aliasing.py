import cv2
import numpy as np
import matplotlib.pyplot as plt
from step01_load import load_video

if __name__ == "__main__":
    frames_1, fps_1 = load_video("data/fan_normal.mov")
    frames_2, fps_2 = load_video("data/fan_slomo.mov")

    print(f'frames_1.shape: {frames_1.shape}, Duration: {817/fps_1} seconds')
    print(f'frames_2.shape: {frames_2.shape}, Duration: {3784/fps_2} seconds') #3784 is the number of frames in the video, so this calculates the duration of the video in seconds
    print(f'fps_1: {fps_1}, fps_2: {fps_2}') #print the fps of the two videos
    