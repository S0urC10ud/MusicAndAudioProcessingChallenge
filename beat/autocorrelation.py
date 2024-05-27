import numpy as np
from matplotlib import pyplot as plt

MAX_BPM = 200
MIN_BPM = 60


def get_lag_range(fps):
    min_lag = int(60 / MAX_BPM * fps)
    max_lag = int(60 / MIN_BPM * fps)

    return min_lag, max_lag


def find_peaks(auto_correlation, threshold=0):
    peaks = []
    for i in range(1, len(auto_correlation) - 1):
        if auto_correlation[i] > auto_correlation[i - 1] and auto_correlation[i] > auto_correlation[i + 1] and \
                auto_correlation[i] > threshold:
            peaks.append(i)
    return np.array(peaks)


def compute_autocorrelation(odf):
    full_autocorrelation = np.correlate(odf, odf, mode='full')
    return full_autocorrelation[len(full_autocorrelation) // 2:]


def compute_bpm_estimate(peaks, autocorr, fps, min_lag):
    # NOTE: there are cases where the odf has a hard time detecting onsets resulting in no peaks from the autocorr
    # Hot-Fix: just return a default BPM
    if len(peaks) == 0:
        return (MAX_BPM + MIN_BPM) // 2

    highest_peak = np.argmax(autocorr[peaks])
    highest_peak_lag = np.where(autocorr == autocorr[peaks[highest_peak]])[0][0]
    # print(highest_peak_lag + min_lag)

    estimated_bpm = 60 / (highest_peak_lag + min_lag) * fps
    return estimated_bpm


def detect_beats(sample_rate, signal, fps, spect, magspect, melspec, odf_rate, odf, onsets, tempo, options):
    min_lag, max_lag = get_lag_range(fps)
    autocorr = compute_autocorrelation(odf)[min_lag:max_lag]
    peaks = find_peaks(autocorr)

    estimated_bpm = compute_bpm_estimate(peaks, autocorr, fps, min_lag)
    beat_duration = 60 / estimated_bpm

    # get the largets onset value that is still lower than the beat duration
    smaller_onsets = onsets[onsets < beat_duration]
    if len(smaller_onsets) == 0:
        start_onset = beat_duration
    else:
        start_onset = np.max(smaller_onsets)

    beats = list([start_onset])
    next_predicted_beat = start_onset + beat_duration
    error_margin = 0.1

    while next_predicted_beat < onsets[-1]:
        # check if we can find an onset that roughly matches with our next beat prediction
        onsets_on_beat_prediction = [value for value in onsets if
                                     next_predicted_beat - error_margin <= value <= next_predicted_beat + error_margin
                                     and value not in beats]  # avoid endless loops by excluding values we have in the beats already

        if len(onsets_on_beat_prediction) == 0:
            beats.append(next_predicted_beat)
            next_predicted_beat = next_predicted_beat + beat_duration
        else:
            aligned_onset = onsets_on_beat_prediction[0]
            beats.append(aligned_onset)
            next_predicted_beat = next_predicted_beat + beat_duration

    return beats
