import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_video(path, width=160):
    cap = cv2.VideoCapture(path) #shortcut to the video file
    list_of_frames = [] #create an empty list to store the frames 

    #this whole block gets the original height and width of the video and calculates the new height based on the desired width""
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
        
    frames = np.stack(list_of_frames, axis = 0).astype(np.float32) #converts the list into an array and converts the data type to float32
    fps = cap.get(cv2.CAP_PROP_FPS) #get the fps of the video
    cap.release() #release the video capture object

    return frames, fps

if __name__ == "__main__":
    frames, fps = load_video("data/still.mov")
    plt.figure()
    plt.imshow(frames[0], cmap="gray") #show the first frame of the

    plt.figure()
    plt.imshow(frames[:, 48, :], cmap="gray", aspect="auto") #show the line at height 48 of the video, with the pixel column on the x-axis and the frame number on the y-axis
    plt.xlabel("pixel column")
    plt.ylabel("frame")

    plt.figure()
    plt.hist(frames.ravel(), bins=256)
    plt.title("raw brightness")

    plt.figure()
    plt.hist(np.log(frames + 1).ravel(), bins=256)
    plt.title("log brightness")
    plt.show()

