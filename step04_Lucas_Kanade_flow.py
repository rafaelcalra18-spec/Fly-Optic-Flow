from step01_load import load_video
import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    frames, fps = load_video('data/still.mov')

    f0 = frames[0] #first frame
    f1 = np.roll(f0, 1, axis=1) # this basically moves every pixel in the frame by one step to the right

    diff = f1 - f0

    m = np.abs(diff).max()
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(f0, cmap="gray")
    ax[0].set_title("f0")
    ax[1].imshow(f1, cmap="gray")
    ax[1].set_title("f1 (rolled by 1)")
    ax[2].imshow(diff, cmap="gray", vmin=-m, vmax=m)  # zero is mid-grey
    ax[2].set_title("f1 - f0")
    plt.show()
