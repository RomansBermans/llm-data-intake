from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.speech import OpenAISpeech, record_microphone


class SpeechTests(unittest.TestCase):
    def test_transcription_sends_audio_and_returns_text(self):
        client = Mock()
        client.audio.transcriptions.create.return_value = SimpleNamespace(
            text="hello there"
        )
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "candidate.wav"
            audio.write_bytes(b"audio-bytes")

            transcript = OpenAISpeech(api_key="test", client=client).transcribe(audio)

        self.assertEqual(transcript, "hello there")
        call = client.audio.transcriptions.create.call_args
        self.assertEqual(call.kwargs["model"], "gpt-4o-mini-transcribe")
        self.assertEqual(call.kwargs["file"].name, str(audio))

    @patch("src.speech._play_wav")
    def test_speech_generation_plays_generated_audio(self, play_wav):
        client = Mock()
        client.audio.speech.create.return_value.read.return_value = b"wav-bytes"
        OpenAISpeech(api_key="test", client=client).speak("Welcome")

        client.audio.speech.create.assert_called_once_with(
            input="Welcome",
            model="gpt-4o-mini-tts",
            response_format="wav",
            voice="marin",
        )
        play_wav.assert_called_once_with(b"wav-bytes")

    def test_microphone_recording_writes_mono_wav(self):
        class InputStream:
            def __init__(self, callback, **_kwargs):
                self.callback = callback

            def __enter__(self):
                self.callback(b"\x00\x01" * 10, 10, None, None)
                return self

            def __exit__(self, *_args):
                return None

        sounddevice = Mock(RawInputStream=InputStream)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "recording.wav"
            with patch.dict(sys.modules, {"sounddevice": sounddevice}), patch(
                "builtins.input", return_value=""
            ):
                record_microphone(target)
            with wave.open(str(target), "rb") as audio:
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getframerate(), 16_000)
                self.assertEqual(audio.getnframes(), 10)

if __name__ == "__main__":
    unittest.main()
