import numpy as np
from matplotlib import pyplot as plt


def compute_inter_distances(onsets):
    inter_distances = list()
    window_size = 15

    for i in range(len(onsets)):
        for j in range(i + 1, min(i + 1 + window_size, len(onsets))):
            distance = abs(onsets[i] - onsets[j])
            inter_distances.append(distance)

    return np.array(inter_distances)


def histogram_peak_picking(inter_distances):
    bpm_range = inter_distances[
        (inter_distances >= 0.3) & (inter_distances <= 1.0)]  # only get the values that lie within 200 bpm to 60 bpm
    hist, bins = np.histogram(bpm_range, bins=100)

    highest_peak_bin = np.argmax(hist)
    highest_peak_value = (bins[highest_peak_bin] + bins[highest_peak_bin + 1]) / 2

    return highest_peak_value


def detect_beats(sample_rate, signal, fps, spect, magspect, melspect, odf_rate, odf, onsets, tempo, options):
    inter_distances = compute_inter_distances(onsets)
    highest_peak_value = histogram_peak_picking(inter_distances)

    error_margin = 0.3
    next_predicted_beat = highest_peak_value
    beats = list()

    while next_predicted_beat < onsets[-1]:
        # check if we can find an onset that roughly matches with our next beat prediction
        onsets_on_beat_prediction = [value for value in onsets if
                                     next_predicted_beat - error_margin <= value <= next_predicted_beat + error_margin
                                     and value not in beats]  # avoid endless loop by excluding values we have in the beats already

        if len(onsets_on_beat_prediction) == 0:
            # if not we just take the prediction
            beats.append(next_predicted_beat)
            next_predicted_beat = next_predicted_beat + highest_peak_value
        else:
            # if we do, we take this onset as the beat location
            aligned_onset = onsets_on_beat_prediction[0]
            beats.append(aligned_onset)
            next_predicted_beat = aligned_onset + highest_peak_value

    return beats
