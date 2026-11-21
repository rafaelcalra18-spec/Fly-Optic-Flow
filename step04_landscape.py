# Extra for step 4: the frame as a landscape, and its gradient field.
# Left: brightness as height (bright = hills, dark = valleys, edges = cliffs).
# Right: the same patch seen from above, with an arrow at every 2nd pixel pointing uphill.
# The arrow length is how steep the brightness is there.
from step01_load import load_video
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    frames, fps = load_video('data/still.mov')
    f0 = frames[0]

    r0, c0, size = 66, 40, 30                          # top-left corner and size of the patch (textured bottom strip)
    patch = f0[r0:r0 + size, c0:c0 + size]

    Iy, Ix = np.gradient(patch)                       # np.gradient gives the row direction (down) first, then the column direction (right)

    rows, cols = np.mgrid[0:size, 0:size]             # pixel coordinates for every point of the patch

    fig = plt.figure(figsize=(12, 5.5))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(cols, rows, patch, cmap="gray", linewidth=0)
    ax3d.set_xlabel("column (x)")
    ax3d.set_ylabel("row (y)")
    ax3d.set_zlabel("brightness")
    ax3d.invert_yaxis()                               # row 0 at the back, like the top of the image
    ax3d.set_title("brightness as height (drag to rotate)")

    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.imshow(patch, cmap="gray")
    step = 2                                          # one arrow every 2 pixels, or it gets too crowded
    longest = np.hypot(Ix, Iy).max()
    ax2d.quiver(cols[::step, ::step], rows[::step, ::step],
                Ix[::step, ::step], Iy[::step, ::step],
                color="red", angles="xy", scale_units="xy", scale=longest / 2.5)  # angles="xy" so a positive Iy points down, like the rows
    ax2d.set_xlabel("column (x)")
    ax2d.set_ylabel("row (y)")
    ax2d.set_title("seen from above: gradient (Ix, Iy) points uphill")

    plt.tight_layout()
    plt.show()
