from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol

from .speech import OpenAISpeech, SpeechError, SpeechProvider, record_microphone


class ConversationError(RuntimeError):
    """The active conversation interface could not read or deliver a message."""


class ConversationIO(Protocol):
    def read_candidate(self) -> str: ...

    def send_agent(self, message: str) -> None: ...


class TextIO:
    def read_candidate(self) -> str:
        return input("Candidate: ")

    def send_agent(self, message: str) -> None:
        print(f"Agent: {message}")


class VoiceIO:
    def __init__(self, speech: SpeechProvider):
        self.speech = speech
        self._disclosed = False

    def read_candidate(self) -> str:
        try:
            input("——— Press ENTER to START recording. ")
            with tempfile.TemporaryDirectory() as directory:
                audio_path = Path(directory) / "candidate.wav"
                record_microphone(audio_path)
                transcript = self.speech.transcribe(audio_path)
            correction = input(
                f"Candidate: {transcript}\n——— Press ENTER to ACCEPT, or type a correction: "
            )
            correction = correction.strip()
            if correction:
                print(f"Candidate: {correction}")
                return correction
            return transcript
        except SpeechError as error:
            raise ConversationError(
                "Voice input could not be processed. Please try again."
            ) from error

    def send_agent(self, message: str) -> None:
        if not self._disclosed:
            print("Voice mode uses an AI-generated voice.")
            self._disclosed = True
        print(f"Agent: {message}")
        try:
            self.speech.speak(message)
        except SpeechError as error:
            raise ConversationError(
                "Voice playback failed. The response is still available as text."
            ) from error


def create_conversation(mode: str) -> ConversationIO:
    if mode == "text":
        return TextIO()
    if mode == "voice":
        return VoiceIO(OpenAISpeech())
    raise ValueError("mode must be 'text' or 'voice'")
