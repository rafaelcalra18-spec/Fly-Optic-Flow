from step01_load import load_video
import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    frames, fps = load_video('data/still.mov')

    f0 = frames[0] #first frame
    f1 = np.roll(f0, 1, axis=1) # this basically moves every pixel in the frame by one step to the right

    diff = f1 - f0
    Iy, Ix = np.gradient(frames[0]) # unsteady vector field, retuns (Iy ; Ix)

    It = diff   # the time derivative is the frame difference

    # one shared scale for all three derivatives, so their brightness can be compared directly
    lim = max(np.abs(Ix).max(), np.abs(Iy).max(), np.abs(It).max())

    fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)   # 2 rows, 2 columns of panels
    ax[0, 0].imshow(f0, cmap="gray")
    ax[0, 0].set_title("f0")
    ax[0, 1].imshow(Ix, cmap="gray", vmin=-lim, vmax=lim)
    ax[0, 1].set_title("Ix (brightness slope to the right)")
    ax[1, 0].imshow(Iy, cmap="gray", vmin=-lim, vmax=lim)
    ax[1, 0].set_title("Iy (brightness slope downwards)")
    ax[1, 1].imshow(It, cmap="gray", vmin=-lim, vmax=lim)
    ax[1, 1].set_title("It = f1 - f0")
    plt.show()
