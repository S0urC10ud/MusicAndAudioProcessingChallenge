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

    highest_peak = np.argmax(signal[peaks])
    highest_peak_lag_value = np.where(signal == signal[peaks[highest_peak]])[0][0]
    return highest_peak_lag_value


def compute_bpm_estimate(peaks, autocorr, fps, min_lag):
    # NOTE: there are cases where the odf has a hard time detecting onsets resulting in no peaks from the autocorr
    # Hot-Fix: just return a default BPM
    if len(peaks) == 0:
        return (MAX_BPM + MIN_BPM) / 2

    highest_peak = np.argmax(autocorr[peaks])
    highest_peak_lag = np.where(autocorr == autocorr[peaks[highest_peak]])[0][0]    # sadly not so pretty

    estimated_bpm = 60 / (highest_peak_lag + min_lag) * fps
    return estimated_bpm

def detect_tempo(sample_rate, signal, fps, spect, magspect, melspect, odf_rate, odf, onsets, options):
    min_lag, max_lag = get_lag_range(fps)

    full_autocorr = np.correlate(odf, odf, mode='full')
    full_autocorr = (full_autocorr[len(full_autocorr) // 2:])

    # Compute an estimate what the highest bpm should be
    first_peak_lag = find_highest_peak(full_autocorr[min_lag:max_lag])
    first_estimated_bpm = 60 / (first_peak_lag + min_lag) * fps

    # we get an estimate where the second bpm estimation could be (by taking the first estimate * 0.5) ...
    second_bpm_region_center = int(60 / (first_estimated_bpm * 0.5) * fps)
    lag_search_window = 10

    # ... place a window over it a search there for the highest peak
    second_peak_lag = find_highest_peak(full_autocorr[second_bpm_region_center - lag_search_window:second_bpm_region_center + lag_search_window])
    second_estimated_bpm = 60 / (second_peak_lag + second_bpm_region_center - lag_search_window) * fps

    return [second_estimated_bpm, first_estimated_bpm]
