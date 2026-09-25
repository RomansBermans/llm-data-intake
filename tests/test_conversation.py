from __future__ import annotations

import unittest
from unittest.mock import Mock, call, patch

from src.conversation import ConversationError, TextIO, VoiceIO, create_conversation
from src.speech import SpeechError


class ConversationIOTests(unittest.TestCase):
    def test_factory_creates_text_io(self):
        self.assertIsInstance(create_conversation("text"), TextIO)

    @patch("src.conversation.OpenAISpeech")
    def test_factory_creates_voice_io_with_speech_adapter(self, speech_class):
        conversation = create_conversation("voice")

        self.assertIsInstance(conversation, VoiceIO)
        self.assertIs(conversation.speech, speech_class.return_value)

    def test_factory_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "mode must be"):
            create_conversation("unknown")

    @patch("builtins.print")
    @patch("builtins.input", return_value="typed answer")
    def test_text_io_reads_and_writes_text(self, _input, print_message):
        conversation = TextIO()

        message = conversation.read_candidate()
        conversation.send_agent("question")

        self.assertEqual(message, "typed answer")
        print_message.assert_called_once_with("Agent: question")

    @patch("src.conversation.record_microphone")
    @patch("builtins.print")
    @patch("builtins.input", side_effect=["", "corrected answer"])
    def test_voice_io_records_transcribes_and_echoes_correction(
        self, _input, print_message, record
    ):
        speech = Mock()
        speech.transcribe.return_value = "spoken answer"
        conversation = VoiceIO(speech)

        message = conversation.read_candidate()

        self.assertEqual(message, "corrected answer")
        record.assert_called_once()
        speech.transcribe.assert_called_once_with(record.call_args.args[0])
        print_message.assert_called_once_with("Candidate: corrected answer")

    @patch(
        "src.conversation.record_microphone",
        side_effect=SpeechError("transcription failed"),
    )
    @patch("builtins.input", return_value="")
    def test_voice_io_translates_input_errors(self, _input, record):
        with self.assertRaisesRegex(ConversationError, "Voice input could not be processed") as raised:
            VoiceIO(Mock()).read_candidate()
        self.assertIs(raised.exception.__cause__, record.side_effect)

    @patch("builtins.print")
    def test_voice_io_discloses_once_and_speaks_every_message(self, print_message):
        speech = Mock()
        conversation = VoiceIO(speech)

        conversation.send_agent("first")
        conversation.send_agent("second")

        self.assertEqual(
            print_message.call_args_list,
            [
                call("Voice mode uses an AI-generated voice."),
                call("Agent: first"),
                call("Agent: second"),
            ],
        )
        self.assertEqual(speech.speak.call_args_list, [call("first"), call("second")])

    @patch("builtins.print")
    def test_voice_io_translates_output_errors(self, _print_message):
        speech = Mock()
        speech.speak.side_effect = SpeechError("playback failed")

        with self.assertRaisesRegex(ConversationError, "response is still available") as raised:
            VoiceIO(speech).send_agent("question")
        self.assertIs(raised.exception.__cause__, speech.speak.side_effect)


if __name__ == "__main__":
    unittest.main()
