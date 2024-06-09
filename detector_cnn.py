#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detects onsets, beats and tempo in WAV files.

For usage information, call with --help.

Author: Jan Schlüter
"""

from pathlib import Path
from argparse import ArgumentParser
import json
from onset.cnn.inference import perform_inference
import numpy as np
from scipy.io import wavfile

import librosa
try:
    import tqdm
except ImportError:
    tqdm = None


from onset import superflux
from tempo import autocorrelation, ioi_history


def opts_parser():
    usage =\
"""Detects onsets, beats and tempo in WAV files.
"""
    parser = ArgumentParser(description=usage)
    parser.add_argument('indir',
            type=str,
            help='Directory of WAV files to process.')
    parser.add_argument('outfile',
            type=str,
            help='Output JSON file to write.')
    parser.add_argument('--threshold',
            type=float,
            help='Threshold for onset detection',
            default=0.73)
    parser.add_argument('--plot',
            action='store_true',
            help='If given, plot something for every file processed.')
    return parser


def detect_everything(filename, options):
    """
    Computes some shared features and calls the onset, tempo and beat detectors.
    """
    # read wave file (this is faster than librosa.load)
    sample_rate, signal = wavfile.read(filename)

    # convert from integer to float
    if signal.dtype.kind == 'i':
        signal = signal / np.iinfo(signal.dtype).max

    # convert from stereo to mono (just in case)
    if signal.ndim == 2:
        signal = signal.mean(axis=-1)

    # compute spectrogram with given number of frames per second
    fps = 70
    hop_length = sample_rate // fps
    spect = librosa.stft(
            signal, n_fft=2048, hop_length=hop_length, window='hann')

    # only keep the magnitude
    magspect = np.abs(spect)

    # compute a mel spectrogram
    melspect = librosa.feature.melspectrogram(
            S=magspect, sr=sample_rate, n_mels=80, fmin=27.5, fmax=8000)

    # compress magnitudes logarithmically
    melspect = np.log1p(100 * melspect) 

    # compute onset detection function
    odf, odf_rate = onset_detection_function(
            sample_rate, signal, fps, spect, magspect, melspect, options)

    # detect onsets from the onset detection function
    onsets = detect_onsets(odf_rate, odf, options)

    # detect tempo from everything we have
    tempo = detect_tempo(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, options)

    # detect beats from everything we have (including the tempo)
    beats = detect_beats(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, tempo, options)

    # plot some things for easier debugging, if asked for it
    if options.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, sharex=True)
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(filename)
        axes[0].set_title('melspect')
        axes[0].imshow(melspect, origin='lower', aspect='auto',
                       extent=(0, melspect.shape[1] / fps,
                               -0.5, melspect.shape[0] - 0.5))
        axes[1].set_title('onsets')
        axes[1].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in onsets:
            axes[1].axvline(position, color='tab:orange')
        axes[2].set_title('beats (tempo: %r)' % list(np.round(tempo, 2)))
        axes[2].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in beats:
            axes[2].axvline(position, color='tab:red')
        plt.show()

    return {'onsets': list(np.round(onsets, 3)),
            'beats': list(np.round(beats, 3)),
            'tempo': list(np.round(tempo, 2))}


def onset_detection_function(sample_rate, signal, fps, spect, magspect,
                             melspect, options):
    """
    Compute an onset detection function. Ideally, this would have peaks
    where the onsets are. Returns the function values and its sample/frame
    rate in values per second as a tuple: (values, values_per_second)
    """
    return perform_inference(signal, sample_rate), sample_rate / 220 # where 44 is the hop size

    # we only have a dumb dummy implementation here.
    # it returns every 1000th absolute sample value of the input signal.
    # this is not a useful solution at all, just a placeholder.
    
    values = np.abs(signal[::1000])
    values_per_second = sample_rate / 1000
    return values, values_per_second

def detect_onsets(odf_rate, odf, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """
    threshold = 0.92
    peaks=[]
    for ind in range(1,len(odf)-1):
        if (odf[ind]>threshold) and (odf[ind+1] < odf[ind] > odf[ind-1]):
            peaks.append(ind)
    return np.array(peaks) / odf_rate
    

def detect_onsets_old(odf_rate, odf, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """
    peaks = []
    potential_peak = None
    for i in range(len(odf)):
        if odf[i] > options.threshold: # threshold
            if potential_peak is None or odf[i] > odf[potential_peak]:
                potential_peak = i
        if potential_peak is not None and i >= potential_peak + 25: # getting out of the min distance window of 15
            peaks.append(potential_peak)
            potential_peak = None
    if potential_peak is not None:
        peaks.append(potential_peak)
    #width = 30
    #flattened_detection_fn = np.convolve(odf, np.ones(width), mode="same")
    #flattened_detection_fn = np.convolve(flattened_detection_fn, np.ones(width), mode="same")
    #flattened_detection_fn = np.convolve(flattened_detection_fn, np.ones(width), mode="same")
    #flattened_detection_fn /= np.max(flattened_detection_fn)
    #return scipy.signal.find_peaks(flattened_detection_fn, distance = 50, prominence=options.threshold, wlen=50)[0] / odf_rate
    return np.array(peaks) / odf_rate


def detect_tempo(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, options):
    """
    Detect tempo using any of the input representations.
    Returns one tempo or two tempo estimations.
    """    
    # we only have a dumb dummy implementation here.
    # it uses the time difference between the first two onsets to
    # define the tempo, and returns half of that as a second guess.
    # this is not a useful solution at all, just a placeholder.
    # if len(onsets) < 2:
    #     return [120.0, 60.0]
    # tempo = 60 / (onsets[1] - onsets[0])
    # return [tempo / 2, tempo]

    # use superflux onset-detection as it is best for tempo estimation via autocorrelation which works best so far
    odf, odf_rate = superflux.onset_detection_function(sample_rate, signal, fps, spect, magspect, melspect, options)
    return autocorrelation.detect_tempo(sample_rate, signal, fps, spect, magspect, melspect, odf_rate, odf, onsets, options)

    # return ioi_history.detect_tempo(sample_rate, signal, fps, spect, magspect, melspect, odf_rate, odf, onsets, options)

def detect_beats(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, tempo, options):
    """
    Detect beats using any of the input representations.
    Returns the positions of all beats in seconds.
    """
    if len(onsets) < 2:
        return np.array([1.0,2.0]) # TODO: different strategy for beat detection needed @Daniel
    return ioi_history.detect_beats(sample_rate, signal, fps, spect, magspect, melspect,
                                    odf_rate, odf, onsets, tempo, options)

    # we only have a dumb dummy implementation here.
    # it returns every 10th onset as a beat.
    # this is not a useful solution at all, just a placeholder.

    # return onsets[::10]


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()

    # iterate over input directory
    indir = Path(options.indir)
    infiles = list(indir.glob('*.wav'))
    if tqdm is not None:
        infiles = tqdm.tqdm(infiles, desc='File')
    results = {}
    for filename in infiles:
        results[filename.stem] = detect_everything(filename, options)

    # write output file
    with open(options.outfile, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

