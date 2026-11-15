import cv2
import numpy as np
import matplotlib.pyplot as plt
from step01_load import load_video

# turns a signal into a transformed output
# the transformed output is what a neuron (pixel) stores
# after a given signal (a visual input of brightness for example)
#///////////////////////////////////////////////////////////////////
# The lowpass is a ODE, solved numerically here with Euler
def lowpass(signal, tau, fps):

    output_before_step = signal[0]
    List = [output_before_step] # we put the first value of output = input because of indexing issues
    step = 1/fps # this is usefull for when we will analyse videos
    gain = 1 # K, as we don't care for amplifying K = 1

    for i in range (1, len(signal)):
        output_before_step = List[i-1] # corrects the index
        # main euler equation bellow (look notes)
        output_after_step = output_before_step + step * (gain * signal[i] - output_before_step)/ tau 
        List.append(output_after_step)

    filtered = np.array(List, dtype=float) 
    return filtered

if __name__ == '__main__':
    #Plot 1: lowpass numerical vs lowpass analytical (with dirac as input)
    fps = 100
    tau = 0.05 #seconds
    x_1 = np.zeros(30)
    x_1[1:] = 1 # 0 at the first sample, then 1 from the second on (dirac's function), this is the input
    t_1 = np.arange(30) / fps # time of each sample in s

    def lowpass_analytical(t): # this is the exact solution to the ODE
        return (1 - np.exp(-t/tau))

    # we can plot the difference here (it's pretty small)
    plt.figure()
    plt.plot(t_1, lowpass_analytical(t_1))
    plt.plot(t_1, lowpass(x_1, tau, fps))

    #Plot 2: sin(t) through lowpass
    frequency = 1
    t_2 = np.arange(2000) / 1000 # 1000 is the 'fps'
    x_2 = np.sin(2*np.pi * frequency * t_2) # this is the input

    plt.figure()
    plt.plot(t_2, x_2)
    plt.plot(t_2, lowpass(x_2, 0.05, 1000))

    plt.show()
