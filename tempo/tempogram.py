import numpy as np
import librosa
from matplotlib import pyplot as plt

# Based on the implementations found at: https://www.kaggle.com/code/enrcdamn/tempo-estimation-and-beat-tracking-pipeline#2.-Novelty-Functions

def rms(signal, frame_length, hop_length):
    rms_list = []

    for i in range(0, len(signal), hop_length):
        rms_current_frame = np.sqrt(np.sum(signal[i:i+frame_length]**2) / frame_length)
        rms_list.append(rms_current_frame)

    return np.array(rms_list)

def find_highest_value_in_range(lag_to_tempo, auto_correlation, min_tempo, max_tempo):
    tempo_range_window = (lag_to_tempo > min_tempo) & (lag_to_tempo < max_tempo)
    highest_peak_index = np.argmax(auto_correlation[1:][tempo_range_window])
    return lag_to_tempo[tempo_range_window][highest_peak_index]

def detect_tempo(sample_rate, signal, fps, spect, magspect, melspect, odf_rate, odf, onsets, options):
    # print('hello world', sample_rate // fps)

    hop_length = sample_rate // fps
    rmse = rms(signal, 2048, sample_rate // fps)

    rmse_diff = np.zeros_like(rmse)
    rmse_diff[1:] = np.diff(rmse)

    energy_novelty = np.max([np.zeros_like(rmse_diff), rmse_diff], axis=0)
    auto_correlation = np.correlate(energy_novelty, energy_novelty, mode='full')
    auto_correlation = auto_correlation[len(auto_correlation) // 2:]

    # lag to tempo conversion
    n1 = len(auto_correlation)
    frames = np.arange(1, n1)
    t = frames * (hop_length / sample_rate)
    lag_to_tempo = 60 / t

    base_bpm_estimation = find_highest_value_in_range(lag_to_tempo, auto_correlation, 60, 200)

    search_window = 10
    second_bpm_estimation = find_highest_value_in_range(lag_to_tempo, auto_correlation, base_bpm_estimation / 2 - search_window, base_bpm_estimation / 2 + search_window)

    return [second_bpm_estimation, base_bpm_estimation]