from step01_load import load_video
from step03_lowpass import lowpass
import cv2
import numpy as np

frames, fps = load_video('data/still.mov')
filtered = lowpass(frames, 0.1, fps)
diff = frames - filtered

for n in range(len(frames)):
    change = np.clip(128 + 10 * diff[n], 0, 255)   # grey = no change, light = brighter, dark = darker
    view = np.hstack([frames[n], filtered[n], change]).astype(np.uint8)
    view = cv2.resize(view, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)  # 4x bigger to see it
    cv2.imshow("original | lowpass | change", view)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):          # q to stop
        break
cv2.destroyAllWindows()