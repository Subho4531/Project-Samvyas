import time

import numpy as np

from mimi_decoder import MimiDecoderStream

dec = MimiDecoderStream("mimi")

for block in (1, 2, 4):
    codes = np.random.randint(0, 2048, size=(32, block), dtype=np.int64)
    dec.reset()
    for _ in range(5):
        dec.decode_codes(codes)
    t = time.perf_counter()
    for _ in range(40):
        dec.decode_codes(codes)
    ms = (time.perf_counter() - t) * 1000 / 40
    audio_ms = block * 80
    print(f"block {block:2d} frames ({audio_ms:3d} ms of audio): {ms:6.2f} ms  ->  "
          f"{ms / audio_ms:.3f}x realtime, {audio_ms / ms:.1f}x headroom")
