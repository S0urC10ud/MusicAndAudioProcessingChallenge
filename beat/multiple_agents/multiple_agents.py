import numpy as np
import numpy.typing as npt

import socket
import struct
import traceback

cnt = [0]

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("localhost",1337))

def detect_beats(sample_rate: int, signal: npt.NDArray, fps: int, spect: npt.NDArray, magspect: npt.NDArray,
                             melspect: npt.NDArray, odf_rate: int, odf: npt.NDArray, onsets: npt.NDArray, tempo: npt.NDArray, options):
    
    options.plot=False

    #cnt[0] += 1
    #if cnt[0]==43:
    #    options.plot = True
    try:
        sock.send(struct.pack("!i", len(onsets)))

        #onsets.tofile(onset_file)
        #onset_file.write(struct.pack(f'>{len(onsets)}f', *onsets))
        
        

        odf_indices = np.array(onsets*odf_rate, int)
        odf_values = odf[odf_indices]

        sock.send(struct.pack(f'>{len(onsets)}d', *onsets))
        sock.send(struct.pack(f'>{len(onsets)}d', *odf_values))

        sock.send(b'\x00')#sanity check byte
        
        n_beats_tuple = struct.unpack("!i",sock.recv(4))
        assert len(n_beats_tuple)==1
        n_beats = n_beats_tuple[0]

        if n_beats==0:
            print("ERROR: no beats received")
            options.plot=True
            return onsets[::10]

        target_bytes = n_beats*8
        received_data = []
        while target_bytes > 0:
            received = sock.recv(target_bytes)
            received_data += (struct.unpack(f'>{len(received)//8}d', received))
            target_bytes -= len(received)
        
        received_beats = np.array(received_data)
        
        assert len(received_beats)==n_beats, f"invalid number of beats received, expected {n_beats}, got {len(received_beats)}"
        #received_beats = struct.unpack(f'>{n_beats}d', received_data)

        sanity = sock.recv(1)
        if sanity != b'\x00':
            raise ValueError("wrong sanity byte")
        return np.array(received_beats)
    except struct.error as e:
        print("ERROR ")
        print(traceback.format_exc())
        options.plot=True
        return onsets[::10]

    