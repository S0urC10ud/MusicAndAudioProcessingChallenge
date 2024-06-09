import numpy as np
from matplotlib import pyplot as plt

MAX_BPM = 200
MIN_BPM = 60

def get_lag_range(fps):
    min_lag = int(60 / MAX_BPM * fps)
    max_lag = int(60 / MIN_BPM * fps)

    return min_lag, max_lag


def find_highest_peak(signal, threshold=0):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
            peaks.append(i)

    peaks = np.array(peaks)
    if len(peaks) == 0: # return an error code when no nice peaks have been detected
        return -1

    highest_peak = np.argmax(signal[peaks])
    highest_peak_lag_value = np.where(signal == signal[peaks[highest_peak]])[0][0]
    return highest_peak_lag_value

def detect_tempo(sample_rate, signal, fps, spect, magspect, melspect, odf_rate, odf, onsets, options):
    min_lag, max_lag = get_lag_range(fps)

    full_autocorr = np.correlate(odf, odf, mode='full')
    full_autocorr = (full_autocorr[len(full_autocorr) // 2:])

    # plt.plot(full_autocorr)
    # plt.show()

    # Compute an estimate what the highest bpm should be
    first_peak_lag = find_highest_peak(full_autocorr[min_lag:max_lag])
    if first_peak_lag == -1:
        # print('returned', (MAX_BPM + MIN_BPM) / 2)
        return [(MAX_BPM + MIN_BPM) / 2]    # we found no initial peaks -> so something with the signal is wrong

    first_estimated_bpm = 60 / (first_peak_lag + min_lag) * fps

    # we get an estimate where the second bpm estimation could be (by taking the first estimate * 0.5) ...
    second_bpm_region_center = int(60 / (first_estimated_bpm * 0.5) * fps)
    lag_search_window = 10

    # ... place a window over it a search there for the highest peak
    second_peak_lag = find_highest_peak(full_autocorr[second_bpm_region_center - lag_search_window:second_bpm_region_center + lag_search_window])
    if second_peak_lag == -1:
        # print('returned', first_estimated_bpm)
        return [first_estimated_bpm]    # -> no second peak -> just report the first one

    # report both peaks back as tempo estimates
    second_estimated_bpm = 60 / (second_peak_lag + second_bpm_region_center - lag_search_window) * fps
    # print('returning', second_estimated_bpm, first_estimated_bpm)
    return [second_estimated_bpm, first_estimated_bpm]
