from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI, OpenAIError


class SpeechProvider(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...

    def speak(self, text: str) -> None: ...


class SpeechError(RuntimeError):
    """Recording, transcription, or speech playback failed."""


class OpenAISpeech:
    def __init__(
        self,
        api_key: str | None = None,
        transcription_model: str | None = None,
        tts_model: str | None = None,
        voice: str | None = None,
        client: OpenAI | None = None,
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when --mode voice is used")
        self.transcription_model = transcription_model or os.environ.get(
            "OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"
        )
        self.tts_model = tts_model or os.environ.get(
            "OPENAI_TTS_MODEL", "gpt-4o-mini-tts"
        )
        self.voice = voice or os.environ.get("OPENAI_TTS_VOICE", "marin")
        self.client = client or OpenAI(api_key=api_key, timeout=60)

    def transcribe(self, audio_path: Path) -> str:
        try:
            with audio_path.open("rb") as audio:
                response = self.client.audio.transcriptions.create(
                    model=self.transcription_model,
                    file=audio,
                )
        except (OSError, OpenAIError) as error:
            raise SpeechError(f"Transcription failed: {error}") from error
        transcript = response if isinstance(response, str) else response.text
        transcript = transcript.strip()
        if not transcript:
            raise SpeechError("No speech was detected")
        return transcript

    def speak(self, text: str) -> None:
        try:
            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice=self.voice,
                input=text,
                response_format="wav",
            )
        except OpenAIError as error:
            raise SpeechError(f"Speech generation failed: {error}") from error
        _play_wav(response.read())


def record_microphone(output_path: Path, sample_rate: int = 16_000) -> None:
    sounddevice = _sounddevice()

    frames: list[bytes] = []
    recording_errors: list[str] = []

    def capture(data: Any, _frames: int, _time: Any, status: Any) -> None:
        if status:
            recording_errors.append(str(status))
        frames.append(bytes(data))

    try:
        with sounddevice.RawInputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            callback=capture,
        ):
            input("——— Press ENTER to STOP recording. ")
    except SpeechError:
        raise
    except Exception as error:
        raise SpeechError(f"Microphone recording failed: {error}") from error
    if recording_errors:
        raise SpeechError(f"Microphone recording failed: {recording_errors[0]}")
    if not frames:
        raise SpeechError("No audio was recorded")
    with wave.open(str(output_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"".join(frames))


def _play_wav(content: bytes) -> None:
    sounddevice = _sounddevice()
    try:
        with wave.open(io.BytesIO(content), "rb") as audio:
            width = audio.getsampwidth()
            dtype = {1: "uint8", 2: "int16", 4: "int32"}.get(width)
            if not dtype:
                raise SpeechError(f"Unsupported WAV sample width: {width}")
            with sounddevice.RawOutputStream(
                samplerate=audio.getframerate(),
                channels=audio.getnchannels(),
                dtype=dtype,
            ) as output:
                while frames := audio.readframes(4096):
                    output.write(frames)
    except SpeechError:
        raise
    except Exception as error:
        raise SpeechError(f"Voice playback failed: {error}") from error


def _sounddevice() -> Any:
    try:
        import sounddevice
    except ImportError as error:
        raise SpeechError("Voice mode requires the sounddevice package") from error
    return sounddevice
