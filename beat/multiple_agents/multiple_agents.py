import numpy as np
import numpy.typing as npt

import socket
import struct
import traceback


#from sys import path
#from os.path import dirname as dir
#print(dir("../.."))
#path.append(dir("../.."))

from onset.onset_detection import find_onsets
#from ...onset_detection import find_onsets



def smooth_data(data, window_length=15):
    """Smooth the data using a moving average."""
    #somewhat gaussian window...
    window = np.array([0.01, 0.02,0.02,0.05,0.05,0.15, 0.15 ,0.15, 0.15, 0.15, 0.05, 0.05, 0.02, 0.02, 0.01])
    #window = np.ones(window_length)
    return np.convolve(data, window, mode='same')



# I am deliberately not closing this socket. that is done automatically when the program terminates
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("localhost",1337))

reset_plot = [False]
cnt = [0]

def detect_tempo(sample_rate: int, signal: npt.NDArray, fps: int, spect: npt.NDArray, magspect: npt.NDArray,
                             melspect: npt.NDArray, odf_rate: int, odf: npt.NDArray, onsets: npt.NDArray, options) -> list[float]:
    sock.send(b'\x01') # beat tracking identification

    sock.send(struct.pack("!i", len(onsets)))
    sock.send(struct.pack(f'>{len(onsets)}d', *onsets))

    sock.send(b'\x00')#sanity check byte


    results =  read_double_array()
    assert sock.recv(1) == b'\x00', "wrong sanity byte"

    assert len(results)==2, f"expected 2 results, got: {len(results)}"

    return results

def detect_beats(sample_rate: int, signal: npt.NDArray, fps: int, spect: npt.NDArray, magspect: npt.NDArray,
                             melspect: npt.NDArray, odf_rate: int, odf: npt.NDArray, onsets: npt.NDArray, tempo: npt.NDArray, options) -> npt.NDArray:

    if reset_plot[0]:
        reset_plot[0]=False
        options.plot=False

    try:
        #onsets = find_onsets(smooth_data(odf), odf_rate,
        #    adaptive_threshold_window_size=odf_rate//1, required_max_window_size=odf_rate//5,
        #    delta=0.03, l=0.9, high_window_reduction_factor=1)


        sock.send(b'\x00') # beat tracking identification

        sock.send(struct.pack("!i", len(onsets)))

        odf_indices = np.array(onsets*odf_rate, int)
        odf_values = odf[odf_indices]

        #odf_values = np.log(odf_values+1)

        sock.send(struct.pack(f'>{len(onsets)}d', *onsets))
        sock.send(struct.pack(f'>{len(onsets)}d', *odf_values))

        sock.send(b'\x00')#sanity check byte
        
        received_beats = np.array(read_double_array())
        
        #received_beats = struct.unpack(f'>{n_beats}d', received_data)

        assert sock.recv(1) == b'\x00', "wrong sanity byte"
        #return onsets
        return np.array(received_beats)
    except struct.error as e:
        print("ERROR ")
        print(traceback.format_exc())
        options.plot=True
        reset_plot[0]=True
        return onsets # for debugging

def read_double_array() -> list[float]:
    n_beats_tuple = struct.unpack("!i",sock.recv(4))
    assert len(n_beats_tuple)==1
    n_beats = n_beats_tuple[0]
    assert n_beats>=0, "expect positive number of beats received"

    if n_beats==0:
        print("ERROR: no beats received")
        options.plot=True
        return onsets # for debugging

    target_bytes = n_beats*8
    received_data: list[float] = []
    while target_bytes > 0:
        received = sock.recv(target_bytes)
        received_data += (struct.unpack(f'>{len(received)//8}d', received))
        target_bytes -= len(received)
    
    assert len(received_data)==n_beats, f"invalid number of beats received, expected {n_beats}, got {len(received_beats)}"

    return received_data