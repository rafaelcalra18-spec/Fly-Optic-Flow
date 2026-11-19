import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Gives the gain and the lag of a sin-wave of a input => frequency, for a lowpass of input => tau
# and a sampling rate fs (basically fps)
def Gain_lag(frequency, tau = 0.05, fs = 1000): 
    w = 2 * np.pi * frequency
    time = np.arange(3 * fs) / fs # creates 3s worth of time (3*fs amount of data)
    unfiltered = np.sin( w * time) #sin of frequency f
    filtered = lowpass(unfiltered, tau, fs) #passes through lowpass here

    lower_bound = fs # is the sample at t = 1s
    upper_bound = 2*fs # is the sample at t = 2s

    Gain = np.max(filtered[lower_bound: upper_bound]) # amplitude of the filtered wave

    peak_unfiltered_local = np.argmax(unfiltered[lower_bound: upper_bound])

    # CAREFUL!!!, we want the x-coordinate of the filtered peak right next to unfiltered peak
    # therefore we only look at filtered from where unfiltered peaks, and one full cycle later (fs/frequency)
    peak_unfiltered_global = peak_unfiltered_local + lower_bound # sample value of the peak in the whole function
    lag = np.argmax(filtered[peak_unfiltered_global: peak_unfiltered_global + int(fs/frequency)]) / fs
    #lag is therefore the sample at which the peak of filtered is measured from, 
    #because sample 0 IS the unfiltered peak

    predicted_Gain = 1 / np.sqrt(1 + (w * tau) ** 2) # this is what the book says
    predicted_lag = np.arctan(w * tau) / w # same here

    return Gain, lag, predicted_Gain, predicted_lag

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

    #Plot 2: sin(t) through lowpass, gain and lag 
    frequency = 1
    t_2 = np.arange(2000) / 1000 # 1000 is the 'fps'
    x_2 = np.sin(2*np.pi * frequency * t_2) # this is the input

    plt.figure()
    plt.plot(t_2[1000:2000], x_2[1000: 2000])
    plt.plot(t_2[1000:2000], lowpass(x_2, 0.05, 1000)[1000: 2000])

    Gain = np.max(lowpass(x_2, 0.05, 1000)[1000: 2000])
    lag = np.argmax(x_2[1000: 2000]) - np.argmax(lowpass(x_2, 0.05, 1000)[1000: 2000])

    print (f'Gain is: {Gain}')
    print (f'Lag is: {lag} ms')
    plt.show()

    # Test 3: Gain_lag function test
    print(f"{'f (Hz)':>7} {'gain meas':>10} {'gain pred':>10} {'lag meas (ms)':>14} {'lag pred (ms)':>14}")

    for f in [1, 3, 10]:
        g_m, l_m, g_p, l_p = Gain_lag(f)
        print(f"{f:>7} {g_m:>10.3f} {g_p:>10.3f} {l_m * 1000:>14.1f} {l_p * 1000:>14.1f}")