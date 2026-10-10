import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

TARGET_SR = 16000


@dataclass
class STTResult:
    text: str
    language: str
    language_probability: float
    duration_sec: float
    wall_sec: float = 0.0
    segments: list = field(default_factory=list)

class SpeechToText:
    def __init__(
        self,
        model_size: str = "small",
        device: str | None = None,
        compute_type: str | None = None,
        beam_size: int = 3,             # подняли с 1 до 3 для лучшего выравнивания
        vad_filter: bool = True,
        # НОВОЕ: подсказка модели
        initial_prompt: str = "Джарвис. Джарвис, проснись. Ассистент. Агент. ",
    ):
        from faster_whisper import WhisperModel

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        self.device = device
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.initial_prompt = initial_prompt

    @staticmethod
    def _load_wav(audio_path: str | Path) -> np.ndarray:
        
        wav, sr = torchaudio.load(str(audio_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        return wav.squeeze(0).numpy().astype(np.float32)

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> STTResult:
        t0 = time.perf_counter()
        audio_np = self._load_wav(audio_path)
        duration_sec = len(audio_np) / TARGET_SR

        # temperature — список: сначала 0, если не получилось — 0.2, 0.4...
        # это стандартная практика Whisper против галлюцинаций
        segments, info = self.model.transcribe(
            audio_np,
            language=language,
            beam_size=self.beam_size,
            best_of=1,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=self.initial_prompt,   # <-- НОВОЕ
        )

        raw_texts = []
        for seg in segments:
            t = seg.text.strip()
            if t:
                raw_texts.append(t)

        raw = " ".join(raw_texts).strip()
        text = self._cleanup_hallucinations(raw)

        wall = time.perf_counter() - t0
        return STTResult(
            text=text,
            language=info.language,
            language_probability=round(info.language_probability, 3),
            duration_sec=round(duration_sec, 2),
            wall_sec=round(wall, 3),
            segments=[text],
        )

    @staticmethod
    def _cleanup_hallucinations(text: str) -> str:
        """
        Убирает типичные галлюцинации Whisper в начале русской речи.
        Это известный список артефактов модели на OOV-словах и тишине.
        """
        import re

        hallucinations = [
            r"^здравствуйте[,\.!\s]*",
            r"^добрый день[,\.!\s]*",
            r"^доброе утро[,\.!\s]*",
            r"^добрый вечер[,\.!\s]*",
            r"^спасибо за просмотр[,\.!\s]*",
            r"^подписывайтесь на канал[,\.!\s]*",
            r"^ставьте лайк[,\.!\s]*",
            r"^до свидания[,\.!\s]*",
            r"^you[,\.!\s]*",
            r"^thank you[,\.!\s]*",
        ]

        cleaned = text
        for pattern in hallucinations:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        return cleaned.strip()
    
    
# if __name__ == "__main__":
#     import argparse

#     ap = argparse.ArgumentParser()
#     ap.add_argument("--file", default="/home/fedosdan2/Study/Jarvis/test_sound/test3.wav")
#     ap.add_argument("--model", default="small")
#     ap.add_argument("--language", default="ru", help="ru / en / None=авто")
#     ap.add_argument("--compute", default="float16", help="float16 / int8 / None=авто")
#     args = ap.parse_args()

#     logging.basicConfig(level=logging.INFO)
#     stt = SpeechToText(model_size=args.model, compute_type=args.compute)
#     res = stt.transcribe(args.file, language=args.language)

#     rtf = res.wall_sec / res.duration_sec if res.duration_sec else 0
#     print(f"\nText:     {res.text}")
#     print(f"Language: {res.language} (p={res.language_probability})")
#     print(f"Audio:    {res.duration_sec}s | Decode: {res.wall_sec}s | RTF={rtf:.2f}")
#     print("(RTF < 1.0 означает быстрее реального времени)")