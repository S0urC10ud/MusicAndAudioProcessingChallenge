import numpy as np
import numpy.typing as npt

from matplotlib import pyplot as plt

# python detector.py data/train/ data/out.json && python evaluate.py data/train/ data/out.json

def spectral_difference(spect: npt.NDArray, norm_index: int = 1) -> npt.NDArray:
    spect=spect.T
    diff = np.zeros_like(spect)
    diff[:,1:] = spect[:,1:]-spect[:,:-1]
    diff[:,0] = diff[:,1]
    
    positive_diff = np.abs(np.maximum(diff, 0))
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

def adaptive_thresholding(odf: npt.NDArray, adaptive_threshold_window_size:int, required_max_window_size:int, delta:float|int, l:float|int):
    onsets = np.zeros_like(odf, dtype=bool)
    
    last_picked = -1000

    for i in range(len(odf)):

        # if there are multiple possible peaks near each other, peak the first one
        # this is done by looking at fewer values "in the future" for local max checking
        # and then skipping a few iterations i.e. not considering peaks after that

        if i > last_picked + required_max_window_size: # avoid multiple peaks
            low = max(0, i-adaptive_threshold_window_size)
            high = min(len(odf), i+adaptive_threshold_window_size)

            # require to be the maximum in a local window
            req_max_low = max(0, i-required_max_window_size)
            req_max_high = min(len(odf), i+required_max_window_size//3)
            is_local_max = odf[i]>=np.max(odf[req_max_low:req_max_high])

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

def detect_onsets(odf_rate: int, odf: npt.NDArray, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """

    onsets = adaptive_thresholding(odf,
        adaptive_threshold_window_size=odf_rate//10, required_max_window_size=odf_rate//15,
        delta=0.005, l=1.1)

    strongest_indices = np.where(onsets)[0]

    # Debugging: show plot in case of few detected onsets
    if restore_plot_false[0]:
        restore_plot_false[0] = False
        options.plot = False
    if len(strongest_indices) <=10:
        options.plot = True
        restore_plot_false[0] = True
        
        # can add artificial onsets in order to display plot
        #strongest_indices = np.concatenate([[0,1,2,3], strongest_indices])

    return strongest_indices / odf_rate