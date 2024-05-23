import numpy as np
import numpy.typing as npt

from matplotlib import pyplot as plt

# python detector.py data/train/ data/out.json && python evaluate.py data/train/ data/out.json

def spectral_difference(spect: npt.NDArray, norm_index: int = 1) -> npt.NDArray:
    spect=spect.T
    diff = np.zeros_like(spect)
    diff[:,1:] = spect[:,1:]-spect[:,:-1]
    diff[:,0] = diff[:,1]
    
    positive_diff = np.abs(diff)
    return np.sum(positive_diff**norm_index, axis=1)**(1/norm_index)

def onset_detection_function(sample_rate: int, signal: npt.NDArray, fps: int, spect: npt.NDArray, magspect: npt.NDArray,
                             melspect: npt.NDArray, options):
    """
    Compute an onset detection function. Ideally, this would have peaks
    where the onsets are. Returns the function values and its sample/frame
    rate in values per second as a tuple: (values, values_per_second)
    """

    values = spectral_difference(spect)

    # normalize to roughly [0,1]
    n_outliers = len(values)//25 # //25 means 50 elements considered to be outliers in min/max calculation
    min_value = np.max(values[np.argpartition(values, n_outliers)[:n_outliers]])
    max_value = np.min(values[np.argpartition(values, -n_outliers)[-n_outliers:]])
    values = (values-min_value)/(max_value-min_value)

    return values, fps

def adaptive_thresholding(odf: npt.NDArray, adaptive_threshold_window_size:int, required_max_window_size:int, delta:float|int, l:float|int, high_window_reduction_factor: int=3):
    onsets = np.zeros_like(odf, dtype=bool)
    
    last_picked = -1000

    for i in range(len(odf)):

        # if there are multiple possible peaks near each other, peak the first one
        # this is done by looking at fewer values "in the future" for local max checking
        # and then skipping a few iterations i.e. not considering peaks after that

        if i > last_picked + required_max_window_size: # avoid multiple peaks
            
            # check for local max
            # require to be the maximum in a local window
            req_max_low = max(0, i-required_max_window_size)
            req_max_high = min(len(odf), i+int(required_max_window_size/high_window_reduction_factor))
            is_local_max = odf[i]>=np.max(odf[req_max_low:req_max_high])

            # adaptive thresholding
            low = max(0, i-adaptive_threshold_window_size)
            high = min(len(odf), i+adaptive_threshold_window_size)
            med = np.median(odf[low:high])
            threshold = delta + l*med
            
            is_onset = odf[i] > threshold and is_local_max
            if is_onset:
                last_picked = i
            onsets[i] = is_onset

    # no onset at very beginning 
    onsets[:2] = False
    return onsets

restore_plot_false = [False]


def find_onsets(odf: npt.NDArray, odf_rate: int, adaptive_threshold_window_size:int, required_max_window_size:int, delta:float|int, l:float|int, high_window_reduction_factor:int):
    onsets = adaptive_thresholding(odf,
        adaptive_threshold_window_size=adaptive_threshold_window_size, required_max_window_size=required_max_window_size,
        delta=delta, l=l, high_window_reduction_factor=high_window_reduction_factor)

    strongest_indices = np.where(onsets)[0]

    return strongest_indices / odf_rate

def detect_onsets(odf_rate: int, odf: npt.NDArray, options) -> npt.NDArray:
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """

    onsets = find_onsets(odf, odf_rate,
        adaptive_threshold_window_size=odf_rate//10, required_max_window_size=odf_rate//20,
        delta=0.01, l=1.1,high_window_reduction_factor=3)

    if restore_plot_false[0]:
        restore_plot_false[0] = False
        options.plot = False
    if len(onsets) <=5:
        print("WARN: few onsets detected")
        options.plot = True
        restore_plot_false[0] = True
    
    return onsets