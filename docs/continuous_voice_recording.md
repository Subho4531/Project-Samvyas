# Record once, review the clips afterward

Run `.venv-recorder\Scripts\python.exe scripts\voice_recorder_web.py`, open the
local page and select your existing user name. Restart an already running server
to load the new modes.

Choose **All 30 sentences**. Press Record once, read one line at a time and leave
**two seconds of silence between lines**, then press Stop after the last line.
Do not read numbers or headings. The entire script appears on the page and can
also be opened in [continuous_sentences_bn.txt](../configs/tts/continuous_sentences_bn.txt).
Keep ordinary pauses inside a sentence shorter than a second. Use your normal
voice, varying expression naturally when reading questions or excited sentences.

Press **Save master & review cuts**. Listen to each proposed crop, edit its text
to exactly what you said, tick the confirmation and press **Accept clip**. Only
accepted clips enter the existing training manifest. If the detected clip count
differs from the line count, transcripts remain blank rather than being assigned
to the wrong audio. You may transcribe those clips manually or record again with
clearer gaps. Silence detection does not recognize speech: even matching counts
still need a listening check. A long pause inside a sentence can cause an extra
cut; noise or breathing between lines can prevent a cut.

Choose **Continuous story voiceover** for the separate
[continuous_story_bn.txt](../configs/tts/continuous_story_bn.txt). Read it naturally
as one performance. The story uses contemporary everyday Bengali conversation,
with dialogue and common modern settings such as remote work, group chats, and
smartphones; read the dialogue with your natural expression. It is saved whole for later transcription alignment, without
automatically adding a long recording to the short-clip training dataset.

Both modes allow up to **10 minutes**. Edit the script field before saving if you
changed the words. Each take gets a new folder under
`assets/audio/voice_users/<user>/sessions/<session-id>/`, containing the exact
browser WAV (`original.wav`), a mono 16 kHz PCM master, `script.txt`, `review.json`
and sentence crops. Recording uses the existing lossless browser capture without
denoising or automatic gain. Full originals are preserved even after accepting
crops. Unsaved browser takes are lost on refresh; saved files remain on disk.
The review cards currently remain in the current browser session, so finish
accepting desired clips before switching modes/users or refreshing.
