"""Step 2: Model wrappers.
The idea: we have two different speech-to-text models, and each one has
its own way of being loaded and used. To keep the rest of the project
simple, we wrap each model in a small class that works the same way:

    wrapper = WhisperWrapper()
    text = wrapper.predict("some_clip.wav")   # -> transcription string

So every wrapper has:
    - a .name attribute  (used in the results table)
    - a .predict(wav_path) method that returns the transcribed text
"""

import torch


class WhisperWrapper:
    """OpenAI Whisper (an encoder-decoder model, multilingual)."""

    # Class attribute: same for every WhisperWrapper we create.
    name = "whisper-base"

    def __init__(self, size="base"):
        # Import inside __init__ so the script doesn't crash at import
        # time if whisper isn't installed but you only want wav2vec2.
        import whisper                       # pip install openai-whisper

        # Download (first time) and load the model into memory.
        self.model = whisper.load_model(size)

    def predict(self, wav_path):
        # Read the wav ourselves with soundfile and hand Whisper the raw
        # samples. This avoids Whisper's default path-loading, which
        # calls the external program ffmpeg (not installed on Windows
        # by default -> WinError 2). Our wavs are already 16 kHz mono
        # float32, exactly the format Whisper expects.
        import soundfile as sf
        audio, sr = sf.read(wav_path, dtype="float32")

        result = self.model.transcribe(audio, language="en", fp16=False)
        return result["text"].strip()


class Wav2Vec2Wrapper:
    """Facebook wav2vec2-base-960h (a CTC model, English-only).

    Heads up: this model outputs UPPERCASE TEXT WITHOUT PUNCTUATION.
    That's why analyze.py normalizes all text before computing WER —
    otherwise Whisper would look better just because of capitalization.
    """

    name = "wav2vec2-base-960h"

    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        import soundfile as sf

        # Keep a reference to the soundfile module so predict() can use it.
        self._sf = sf

        # The processor converts raw audio into the numbers the model
        # expects, and later converts the model's output back into text.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        # The actual neural network.
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # Put the model in evaluation mode (turns off training-only
        # behavior like dropout). Important for consistent results.
        self.model.eval()

    def predict(self, wav_path):
        # Step 1: read the wav file from disk.
        # 'audio' is a numpy array of samples, 'sr' is the sample rate.
        audio, sr = self._sf.read(wav_path)

        # Step 2: prepare the audio for the model.
        # return_tensors="pt" means "give me PyTorch tensors".
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")

        # Step 3: run the model.
        # torch.no_grad() tells PyTorch we're not training, so it can
        # skip bookkeeping and run faster with less memory.
        with torch.no_grad():
            output = self.model(inputs.input_values)
            logits = output.logits

        # Step 4: turn the model's raw scores into text.
        # 'logits' contains, for each moment in time, a score for every
        # possible character. argmax picks the highest-scoring character
        # at each moment.
        ids = torch.argmax(logits, dim=-1)

        # batch_decode turns those character ids into an actual string
        # (and handles CTC details like collapsing repeated characters).
        # It returns a list (one entry per audio clip); we sent one clip,
        # so we take the first entry with [0].
        text = self.processor.batch_decode(ids)[0]
        return text.strip()