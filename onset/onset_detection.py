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
    #plt.plot(values)
    #plt.title("spectral difference")
    #plt.show()
    #values_per_second = sample_rate * spect.shape[1]/len(signal)
    return values, fps

def adaptive_thresholding(odf: npt.NDArray, adaptive_threshold_window_size:int, required_max_window_size:int, delta:float|int, l:float|int):
    onsets = np.zeros_like(odf, dtype=bool)
    
    for i in range(len(odf)):
        low = max(0, i-adaptive_threshold_window_size)
        high = min(len(odf), i+adaptive_threshold_window_size)

        req_max_low = max(0, i-required_max_window_size)
        req_max_high = min(len(odf), i+required_max_window_size)
        is_local_max = odf[i]>=np.max(odf[req_max_low:req_max_high])

        med = np.median(odf[low:high])
        threshold = delta + l*med
        
        onsets[i] = odf[i] > threshold and is_local_max

    # no onset at very beginning 
    onsets[:2] = False
    return onsets

def detect_onsets(odf_rate: int, odf: npt.NDArray, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """
    # we only have a dumb dummy implementation here.
    # it returns the timestamps of the 100 strongest values.
    # this is not a useful solution at all, just a placeholder.
    
    onsets = adaptive_thresholding(odf, odf_rate, 5, 0, 1)

    strongest_indices = np.where(onsets)[0]#fixed threshold
    return strongest_indices / odf_rate