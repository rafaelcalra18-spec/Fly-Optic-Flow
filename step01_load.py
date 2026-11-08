import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_video(path, width=160):
    cap = cv2.VideoCapture(path) #shortcut to the video file
    list_of_frames = [] #create an empty list to store the frames
    ok, frame = cap.read()
    original_height, original_width = frame.shape[:2]
    shrink_factor = width / original_width #calculate the shrink factor based on the desired width
    height = round(original_height * shrink_factor) #calculate the new height based on the shrink factor
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True: 
        #main loop, reads every frame grayscales it resizes it and appends it to the list of frames
        ok, frame = cap.read()


        if ok == False: #breaks the loop when there are no more frames to read
                break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_resized = cv2.resize(gray_frame, (width, height), interpolation=cv2.INTER_AREA)
        list_of_frames.append(gray_frame_resized) #appends the resized frame to the list of frames
        
    frames = np.stack(list_of_frames, axis = 0).astype(np.float32) 
    fps = cap.get(cv2.CAP_PROP_FPS) #get the fps of the video
    cap.release() #release the video capture object

    return frames, fps

if __name__ == "__main__":
    frames, fps = load_video("data/turn.mov")
    print(frames.shape, frames.dtype, fps, frames.shape[0] / fps)

