"""Step 2: Model wrappers.

Both models are exposed through one common interface:
    wrapper.predict(wav_path) -> transcription string

This is the piece you adapt to the AIP SDK once you see its docs — the
platform will define what shape a "system under test" must have; keep
these classes and add whatever method names / registration calls it
expects around them.
"""

import torch


class WhisperWrapper:
    """OpenAI Whisper (encoder-decoder, multilingual)."""

    name = "whisper-base"

    def __init__(self, size: str = "base"):
        import whisper                       # pip install openai-whisper
        self.model = whisper.load_model(size)

    def predict(self, wav_path: str) -> str:
        # fp16=False -> works on CPU; language pinned to English so the
        # comparison is about acoustics, not language detection.
        result = self.model.transcribe(wav_path, language="en", fp16=False)
        return result["text"].strip()


class Wav2Vec2Wrapper:
    """Facebook wav2vec2-base-960h (CTC, English-only, outputs UPPERCASE
    without punctuation — that's why analyze.py normalizes text before WER)."""

    name = "wav2vec2-base-960h"

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        import soundfile as sf
        self._sf = sf
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()

    def predict(self, wav_path: str) -> str:
        audio, sr = self._sf.read(wav_path)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(ids)[0].strip()
