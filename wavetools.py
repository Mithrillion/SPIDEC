from scipy import signal


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return b, a


def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = signal.butter(order, high, btype="low")
    return b, a


def lowpass_filtering(X, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order)
    filtered = signal.filtfilt(b, a, X, 0)
    return filtered


def bandpass_filtering(X, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered = signal.filtfilt(b, a, X, 0)
    return filtered