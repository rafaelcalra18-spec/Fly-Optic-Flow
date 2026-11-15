import cv2
import numpy as np
import matplotlib.pyplot as plt
from step01_load import load_video

def lowpass(signal, tau, fps):
    output_before_step = signal[0]
    List = [output_before_step]
    step = 1/fps
    gain = 1
    for i in range (1, len(signal)):
        output_before_step = List[i-1]
        output_after_step = output_before_step + step * (gain * signal[i] - output_before_step)/ tau
        List.append(output_after_step)
    filtered = np.array(List, dtype=float)
    return filtered

print(lowpass(np.array([0, 1, 1, 1, 1, 1], dtype=float), 0.05, 100))
