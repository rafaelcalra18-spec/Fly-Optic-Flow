import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('data/IMG_0130.mov')

L_thing = []
L_framecount = []

number_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in range(number_frames):
    ok, frame = cap.read()
    if not ok:
        break
    time_of_frame = cap.get(cv2.CAP_PROP_POS_MSEC)
    L_thing.append(time_of_frame)
    L_framecount.append(i)

d = np.diff(L_thing)
frames_lost = 478
frames_total = 4770
fps_1 = 240 # VERY IMPORTANT: this is different from the value we get from the video file which is 217.86 fps
# the reason is that it takes into account all the frames taken by the camera (which aren't all at 240fps)
# by adding back the missing frames, and deviding by the intended frame rate of 240fps, we get the actual duration of the video in seconds

print(f'Sum of time differences: {np.sum(d)}')
print((frames_lost + frames_total)/fps_1)  # here is the 'actual' duration of the video
#however there is a mismatch of 30ms which is explained by the 68 frames taken at dt = 4.58ms and not 4.17ms

values, counts = np.unique(np.round(d / (1000/240), 2), return_counts=True)
plt.bar(values, counts, width=0.05)
plt.yscale("log")
plt.show()
