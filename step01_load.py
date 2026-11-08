import cv2
import numpy as np
import matplotlib.pyplot as plt
ok = False
cap = cv2.VideoCapture('data/still.mov')
list_of_frames = []

while True:
    
    ok, frame = cap.read()
    if ok == False:
            break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_resized = cv2.resize(gray_frame, (160, 90), interpolation=cv2.INTER_AREA)
    list_of_frames.append(gray_frame_resized)
    
frames = np.stack(list_of_frames, axis = 0).astype(np.float32)    


print(frames.shape)

