"""Microphone and terminal behavior with no physical recording."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from samvyas.audio.recording import MicrophoneRecorder, countdown, guided_session, terminal_write
from samvyas.data.voice_dataset import VoiceDataset, VoicePrompt


class FakeRecorder:
    def __init__(self):
        self.calls = 0
        self.played = 0

    def record(self):
        self.calls += 1
        return np.full(96000, self.calls * 0.05, dtype=np.float32)

    def play(self, audio):
        self.played += 1


def session(tmp_path, answers, recorder=None, prompts=None):
    dataset = VoiceDataset(tmp_path)
    recorder = recorder or FakeRecorder()
    iterator = iter(answers)
    messages = []
    guided_session(
        dataset,
        prompts or [VoicePrompt("one", "আমি ভালো আছি।", "normal")],
        recorder,
        read=lambda _: next(iterator),
        write=messages.append,
        prepare=lambda: None,
    )
    return dataset, recorder, messages


def test_retry_play_edit_confirm_and_resume(tmp_path):
    dataset, recorder, _ = session(tmp_path, ["r", "p", "n", "r", "e", "আমি ঠিক আছি।", "y"])
    assert recorder.calls == 2 and recorder.played == 1
    assert len(dataset.records) == 1
    assert dataset.records[0]["transcript"] == "আমি ঠিক আছি।"
    assert dataset.records[0]["quality"]["peak"] == pytest.approx(0.1)
    assert len(list(dataset.wav_dir.glob("*.wav"))) == 1
    resumed, recorder, _ = session(tmp_path, [])
    assert recorder.calls == 0 and len(resumed.records) == 1


@pytest.mark.parametrize("answers", [["r", "q"], ["s"], ["q"]])
def test_quit_skip_do_not_save(tmp_path, answers):
    dataset, _, _ = session(tmp_path, answers)
    assert dataset.records == []
    assert not list(dataset.wav_dir.iterdir())


def test_interrupted_capture_keeps_previous_take(tmp_path):
    class Interrupted(FakeRecorder):
        def record(self):
            raise KeyboardInterrupt

    dataset = VoiceDataset(tmp_path)
    dataset.accept(VoicePrompt("old", "old", "normal"), "old", np.full(16000, 0.1))
    dataset, _, messages = session(tmp_path, ["r"], recorder=Interrupted())
    assert dataset.accepted_ids == {"old"}
    assert any("discarded" in msg for msg in messages)


def test_microphone_error_returns_to_controls(tmp_path):
    class Failed(FakeRecorder):
        def record(self):
            raise RuntimeError("input overflow")

    dataset, _, messages = session(tmp_path, ["r", "q"], recorder=Failed())
    assert not dataset.records
    assert any("input overflow" in msg for msg in messages)


def test_countdown():
    messages, waits = [], []
    countdown(messages.append, waits.append)
    assert messages == ["3", "2", "1"] and waits == [1, 1, 1]


def test_eof_while_reviewing_discards_take(tmp_path):
    dataset = VoiceDataset(tmp_path)
    count = 0

    def read(_):
        nonlocal count
        count += 1
        if count == 1:
            return "r"
        raise EOFError

    guided_session(
        dataset,
        [VoicePrompt("one", "test", "normal")],
        FakeRecorder(),
        read=read,
        write=lambda _: None,
        prepare=lambda: None,
    )
    assert not dataset.records and not list(dataset.wav_dir.iterdir())


def test_microphone_check_does_not_save(monkeypatch):
    import samvyas.audio.recording as module

    recorder = FakeRecorder()
    recorder.record = lambda *args, **kwargs: np.full(48000, 0.1, dtype=np.float32)
    monkeypatch.setattr(module, "countdown", lambda _: None)
    replies = iter(["r", "y"])
    assert module.microphone_check(recorder, read=lambda _: next(replies), write=lambda _: None)
    assert recorder.played == 1


@pytest.mark.parametrize("key", ["\r", "\x03", None])
def test_enter_stop_ctrl_c_and_deadline(monkeypatch, key):
    import samvyas.audio.recording as module

    ticks = [0.0]
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(module.time, "monotonic", lambda: ticks[0])
    monkeypatch.setattr(
        module.time, "sleep", lambda seconds: ticks.__setitem__(0, ticks[0] + seconds)
    )
    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(kbhit=lambda: key is not None, getwch=lambda: key)
    )
    if key == "\x03":
        with pytest.raises(KeyboardInterrupt):
            module.wait_for_enter(0.1)
    else:
        module.wait_for_enter(0.1)
        assert ticks[0] == 0 if key else ticks[0] >= 0.1


def test_terminal_write_falls_back_to_python_stdout(capsys, monkeypatch):
    import samvyas.audio.recording as module

    monkeypatch.setattr(module.os, "name", "posix")
    terminal_write("আমি বাংলায় কথা বলছি")
    assert capsys.readouterr().out == "আমি বাংলায় কথা বলছি\n"


def test_capture_caps_frames_and_closes_on_interrupt(monkeypatch):
    class Stop(Exception):
        pass

    state = {"closed": 0}

    class Stream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]
            assert kwargs["samplerate"] == 16000 and kwargs["channels"] == 1

        def __enter__(self):
            try:
                self.callback(np.full((17000, 1), 0.2, dtype=np.float32), 17000, None, None)
            except Stop:
                pass
            return self

        def __exit__(self, *args):
            state["closed"] += 1

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(
            check_input_settings=lambda **_: None,
            InputStream=Stream,
            CallbackStop=Stop,
            CallbackAbort=Stop,
            PortAudioError=RuntimeError,
        ),
    )
    recorder = MicrophoneRecorder()
    audio = recorder.record(1, wait=lambda _: None)
    assert audio.shape == (16000,) and audio.dtype == np.float32

    def interrupt(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        recorder.record(1, wait=interrupt)
    assert state["closed"] == 2
