import numpy as np
import numpy.typing as npt
import scipy

# https://phenicx.upf.edu/system/files/publications/Boeck_DAFx-13.pdf

from onset.onset_detection import find_onsets

def onset_detection_function(sample_rate: int, signal: npt.NDArray, fps: int, spect: npt.NDArray, magspect: npt.NDArray,
                             melspect: npt.NDArray, options) -> tuple[npt.NDArray, int]:
    """
    Compute an onset detection function. Ideally, this would have peaks
    where the onsets are. Returns the function values and its sample/frame
    rate in values per second as a tuple: (values, values_per_second)
    """

    N = 10
    ratio = 0.5

    hop_size = sample_rate/fps


    #first_significant_sample = np.where(magspect>0)[0]

    #frame_offset = max(1, (np.floor(N/2 - first_significant_sample)/hop_size + 0.5))
    frame_offset = 2

    
    magspect=magspect.T
    diff = magspect[frame_offset:,:]-magspect[0:-frame_offset]
    rectified = (diff+np.floor(diff))/2
    pos_rect = np.maximum(rectified, 0)
    
    # TODO filter before sum   
    #filtered = scipy.ndimage.maximum_filter(pos_rect, size=(0,3))
    filtered = pos_rect
    logdiff = np.log10(filtered+1)
    

    spectral_flux = np.zeros(len(magspect))
    spectral_flux[frame_offset:]=np.sum(filtered, axis=1)

    assert spectral_flux.shape==(len(magspect),)

    values = spectral_flux
    # normalize to roughly [0,1]
    n_outliers = len(values)//25 # //25 means 1/50 elements considered to be outliers in min/max calculation
    min_value = np.max(values[np.argpartition(values, n_outliers)[:n_outliers]])
    max_value = np.min(values[np.argpartition(values, -n_outliers)[-n_outliers:]])
    values = (values-min_value)/(max_value-min_value)

    return values, fps

def detect_onsets(odf_rate: int, odf: npt.NDArray, options) -> npt.NDArray:
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """

    #onsets = find_onsets(odf, odf_rate,
    #    adaptive_threshold_window_size=odf_rate//10, required_max_window_size=odf_rate//20,
    #    delta=0.02, l=1.4,high_window_reduction_factor=3)


    pre_max = int(odf_rate*30/1000)
    post_max = int(odf_rate*30/1000)
    pre_avg = odf_rate//10
    post_avg = int(odf_rate*70/1000)
    combination_width = int(odf_rate*30/1000)
    delta = 0.1 # TODO

    last_onset = 0
    onset_indices = []
    for i,onset in enumerate(odf):
        if i>=pre_max and i<len(odf)-post_max and i>=pre_avg and i<len(odf)-post_avg:
            max_cond = odf[i] == np.max(odf[i-pre_max:i+post_max])
            mean_cond =  odf[i] >= np.mean(odf[i-pre_avg:i+post_avg])+delta
            combin_cond = i > last_onset+combination_width
            if max_cond and mean_cond and combin_cond:
                onset_indices.append(i)
                last_onset = i

    

    return np.array(onset_indices) / odf_rate


    return onsets