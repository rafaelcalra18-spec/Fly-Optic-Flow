import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('data/fan_slomo_2.mov')

L_times = [] #list of times of each frame (when they appear in the video)
L_framecount = [] # list of frame counts
number_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #frames in the video

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#this loop takes the time of each frame and appends it to a list, as well as the frame count
for i in range(number_frames):
    ok, frame = cap.read()
    if not ok:
        break
    time_of_frame = cap.get(cv2.CAP_PROP_POS_MSEC)
    L_times.append(time_of_frame)
    L_framecount.append(i)

dt = np.diff(L_times)
frames_lost = 478 #calculated by hand by looking at the histogram bellow

fps_1 = 240 # VERY IMPORTANT: this is different from the value we get from the video file which is 217.86 fps
# the reason is that it takes into account all the frames taken by the camera (which aren't all at 240fps)
# by adding back the missing frames, and deviding by the intended frame rate of 240fps, we get the actual duration of the video in seconds

print(f'Sum of time differences: {np.sum(dt)}')
print((frames_lost + number_frames)/fps_1)  # here is the 'actual' duration of the video
#however there is a mismatch of 30ms which is explained by the 68 frames taken at dt = 4.58ms and not 4.17ms


#this histogram shows the distribution of ratio of the time differences between frames
#e.g: 3 indicates 3 times more space bewteen frame and frame than with 240fps
values, counts = np.unique(np.round(dt / (1000/240), 2), return_counts=True)
plt.bar(values, counts, width=0.05)
plt.yscale("log")
plt.show()
