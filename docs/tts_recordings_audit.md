# New Bengali recording audit

Files under `configs/tts/recordings/` are the new source takes. Keep these M4A
originals unchanged. They are 48 kHz stereo AAC; temporary analysis converted
them to mono 16 kHz PCM and did not publish any training clips.

| Recording | Length | Peak | RMS |
| --- | ---: | ---: | ---: |
| `continuous_sentences_bn_pt1.m4a` | 2:02 | 0.501 | 0.078 |
| `continuous_sentences_bn_pt2.m4a` | 1:42 | 0.467 | 0.081 |
| `continuous_story_bn.m4a` | 8:52 | 0.492 | 0.079 |

The levels look usable: none of the takes is near silence or close to digital
clipping. This says nothing about transcript accuracy or voice similarity.

The sentence script has 30 lines. The existing pause detector finds 16 clips in
part one and 15 in part two with a 0.8-second quiet gap; a 1.2-second gap finds
16 and 14. That is 31 clips at the looser setting. The parts therefore cannot be
paired blindly with 15 lines each. The sentence script was modified shortly
after the recordings. Two likely intended words were corrected based on the
prior prompt text (`কলম` and `বাসে`); confirm them by listening to the clips.

The story matches the current 1,037-word contemporary script and runs 8:52. Its
long pauses do not provide sentence alignment: a 0.8-second cut yields 43 pieces
from 1.8 to 40.4 seconds; a 1.2-second cut yields 15 pieces, several far beyond
the 15-second clip limit in the current training-data preparer. The story needs
transcription alignment and sentence-level crops before it can safely join the
training set. Do not treat automatic pause cuts as transcript alignment.

Current training uses short, exact-text utterances. The two corrected sentence
recordings now yield exactly 30 sequential clips at the existing 1.2-second
pause setting (16 from part one, 14 from part two), and were prepared alongside
the previous 50 accepted clips as an isolated 80-clip dataset. The story remains
outside this run until utterance alignment is ready. Existing source dataset and
v1 checkpoint remain untouched. Kaggle versions 4 and 5 stopped during dependency
setup: the Python 3.13 runtime lacked PyTorch 2.5.1 wheels, then IndicF5’s legacy
NumPy cap conflicted with Kaggle NumPy 2.1.3. Version 6 reused Kaggle torch 2.11
and completed. The report records 72 train clips / 453.28 sec, 8 validation
clips / 52.95 sec, 40 epochs / 720 updates, and validation loss improvement from
0.65291 to 0.52525. Base and adapted previews for three texts at 24 kHz and 16 kHz
were downloaded to `outputs/tts/jeet_adaptation_v2/jeet_adaptation_v2/listening/`;
WAV headers were verified mono, 24-bit, with the requested sample rates. The 1,286 MiB `best.safetensors` checkpoint was subsequently retrieved locally.
The 1,037-word story was synthesized locally on the RTX 2050 in 87 chunks at 32
steps (CUDA; 12.4 min), stitched with pauses at paragraph boundaries, and saved
as 506.39-second mono PCM-24 WAVs at 24 kHz and 16 kHz in
`outputs/tts/jeet_story_local_20261011/`. File headers were verified. Voice
intelligibility and similarity still need human listening review. See the report
and history JSON in `outputs/tts/jeet_adaptation_v2/jeet_adaptation_v2/checkpoints/`.
