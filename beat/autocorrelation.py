import numpy as np
from matplotlib import pyplot as plt

MAX_BPM = 200
MIN_BPM = 60


def get_lag_range(fps):
    min_lag = int(60 / MAX_BPM * fps)
    max_lag = int(60 / MIN_BPM * fps)

    return min_lag, max_lag


def compute_lag_of_bpm_lag(autocorr, fps):
    min_lag, max_lag = get_lag_range(fps)
    highest_peal_lag = find_highest_peak(autocorr[min_lag:max_lag])
    return highest_peal_lag + min_lag


def lag_to_time(lag, fps):
    return lag / fps


def find_highest_peak(signal, threshold=0):
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
            peaks.append(i)

    peaks = np.array(peaks)
    if len(peaks) == 0:
        return len(signal) // 2
    else:
        highest_peak_index = np.argmax(signal[peaks])
        highest_peak_lag_value = np.where(signal == signal[peaks[highest_peak_index]])[0][0]
        return highest_peak_lag_value


def compute_time_offset(onsets, first_beat_timestamp):
    max_min_onset = onsets[0]

    for i in range(1, len(onsets)):
        if onsets[i] < first_beat_timestamp:
            max_min_onset = onsets[i]
        else:
            break

    return first_beat_timestamp - max_min_onset

def find_max_smallest_onset(onsets, time_stamp):
    max_min_onset = onsets[0]
    for i in range(1, len(onsets)):
        if onsets[i] < time_stamp:
            max_min_onset = onsets[i]
        else:
            break
    return max_min_onset

def detect_beats(sample_rate, signal, fps, spect, magspect, melspec, odf_rate, odf, onsets, tempo, options):
    full_autcorr = np.correlate(odf, odf, mode='full')
    full_autcorr = full_autcorr[len(full_autcorr) // 2:]

    bpm_lag = compute_lag_of_bpm_lag(full_autcorr, fps)
    lag_window = 5

    beat_lags = []
    expected_num_beats = len(full_autcorr) // bpm_lag   # Note: the last beat does not get captured yet
    for i in range(1, expected_num_beats + 1):
        expected_lag_center = i * bpm_lag
        peak_lag = find_highest_peak(full_autcorr[expected_lag_center - lag_window:expected_lag_center + lag_window])

        beat_lag = peak_lag + expected_lag_center - lag_window
        beat_lags.append(beat_lag)

    beat_lags.append(beat_lags[-1] + bpm_lag)
    beat_lags = np.array(beat_lags)
    beat_timestamps = np.array([lag_to_time(lag, fps) for lag in beat_lags])

    # Shifting the extracted beats so that they align with onsets
    # Nice idea but does not work
    # time_correction = compute_time_offset(onsets, beat_timestamps[0])
    # corrected_beat_timestamps = beat_timestamps - time_correction
    # return corrected_beat_timestamps

    # Snap the extracted beat locations to the onsets
    snapped_beat_timestamps = np.array([find_max_smallest_onset(onsets, beat_timestamp) for beat_timestamp in beat_timestamps])
    return snapped_beat_timestamps
