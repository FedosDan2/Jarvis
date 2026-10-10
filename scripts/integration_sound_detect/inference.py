#!/usr/bin/env python3
"""
Jarvis Inference Pipeline.

Полный пайплайн обработки входящего аудио:
  1. AudioPreprocessor: любой формат -> WAV 16k mono (возвращает тензор)
  2. AdaptiveIdentityService: проверка "я / не я" на первых 5 секундах
  3. SpeechToText: расшифровка всего аудио (запускается независимо от результата верификации)
"""
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from utils.audio_preprocessor import AudioPreprocessor
from utils.verify_profile import AdaptiveIdentityService
from utils.speech_to_text import SpeechToText


VERIFICATION_WINDOW_SEC = 5.0
TARGET_SR = 16000


@dataclass
class JarvisResult:
    """Результат обработки одного аудио-запроса."""
    is_owner: bool
    confidence: float
    profile_size: int
    text: str
    language: str
    language_probability: float
    audio_duration_sec: float
    input_file: str
    processed_wav_path: str


class JarvisInference:
    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        profile_path: str = "models/identity_service/owner_profile.npy",
        stt_model: str = "small",
        device: str | None = None,
        enable_adaptation: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.preprocessor = AudioPreprocessor(device=device)

        self.verifier = AdaptiveIdentityService(
            model_source=model_source,
            profile_path=profile_path,
            high_confidence_threshold=0.5,
            verify_threshold=0.5,
        )
        self.enable_adaptation = enable_adaptation

        self.stt = SpeechToText(
            model_size=stt_model,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
            beam_size=3,
            vad_filter=True,
        )

    def _tensor_to_wav_file(self, tensor: torch.Tensor, sr: int = TARGET_SR) -> str:
        """
        Сохраняет тензор во временный WAV-файл.
        Возвращает путь к файлу.
        """
        # Переносим тензор на CPU и конвертируем в numpy
        audio_np = tensor.cpu().numpy()
        
        # Если shape (1, N), убираем первое измерение
        if audio_np.ndim == 2 and audio_np.shape[0] == 1:
            audio_np = audio_np.squeeze(0)
        
        # Создаём временный файл
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="jarvis_proc_")
        
        # Сохраняем как WAV 16k mono PCM16
        sf.write(tmp_path, audio_np, sr, subtype="PCM_16")
        
        return tmp_path

    def _extract_verification_window(self, audio_np: np.ndarray, sr: int = TARGET_SR) -> str:
        """
        Извлекает первые VERIFICATION_WINDOW_SEC секунд из numpy массива
        и сохраняет во временный файл для верификации.
        """
        max_samples = int(VERIFICATION_WINDOW_SEC * sr)
        window = audio_np[:max_samples]

        fd, tmp_path = tempfile.mkstemp(suffix=".verify_window.wav", prefix="jarvis_")
        sf.write(tmp_path, window, sr, subtype="PCM_16")

        return tmp_path

    def process(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> JarvisResult:
        """Основной метод инференса."""
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        # ==================================================================
        # ШАГ 1: AudioPreprocessor -> тензор
        # ==================================================================
        processed_tensor = self.preprocessor.preprocess(str(audio_path))
        
        # Сохраняем тензор во временный WAV-файл
        processed_wav_path = self._tensor_to_wav_file(processed_tensor, TARGET_SR)
        
        # Узнаём длительность
        audio_duration = len(processed_tensor.squeeze()) / TARGET_SR

        # ==================================================================
        # ШАГ 2: Верификация на первых 5 секундах
        # ==================================================================
        # Конвертируем тензор в numpy для извлечения окна
        audio_np = processed_tensor.cpu().numpy()
        if audio_np.ndim == 2 and audio_np.shape[0] == 1:
            audio_np = audio_np.squeeze(0)

        verify_window_path = self._extract_verification_window(audio_np, TARGET_SR)
        
        is_owner, confidence = self.verifier.verify_and_adapt(
            verify_window_path,
            enable_adaptation=self.enable_adaptation,
        )

        # Удаляем временный файл окна верификации
        Path(verify_window_path).unlink(missing_ok=True)

        # ==================================================================
        # ШАГ 3: STT на всем аудио
        # ==================================================================
        stt_result = self.stt.transcribe(processed_wav_path, language=language)
        # ==================================================================
        # Собираем результат
        # ==================================================================
        result = JarvisResult(
            is_owner=is_owner,
            confidence=confidence,
            profile_size=len(self.verifier.profile_vectors),
            text=stt_result.text,
            language=stt_result.language,
            language_probability=stt_result.language_probability,
            audio_duration_sec=round(audio_duration, 2),
            input_file=str(audio_path),
            processed_wav_path=processed_wav_path,
        )

        return result

    def cleanup(self, result: JarvisResult):
        """Удаляет временный WAV-файл."""
        try:
            Path(result.processed_wav_path).unlink(missing_ok=True)
        except OSError:
            pass


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Jarvis full inference pipeline")
    ap.add_argument("--file", default="/home/fedosdan2/Study/Jarvis/test_sound/eng.ogg")
    ap.add_argument("--language", default=None, help="ru / en / None=auto")
    ap.add_argument("--stt-model", default="small")
    ap.add_argument("--profile", default="models/identity_service/owner_profile.npy")
    ap.add_argument("--no-adapt", action="store_true")
    args = ap.parse_args()

    jarvis = JarvisInference(
        profile_path=args.profile,
        stt_model=args.stt_model,
        enable_adaptation=not args.no_adapt,
    )

    result = jarvis.process(args.file, language=args.language)

    print("\n" + "=" * 70)
    print("JARVIS INFERENCE RESULT")
    print("=" * 70)

    print(f"\n📂 Input:        {Path(result.input_file).name}")
    print(f"🎵 Duration:     {result.audio_duration_sec:.2f}s")
    print(f"👤 Profile size: {result.profile_size} vectors")

    print(f"\n🔐 VERIFICATION")
    print(f"   Status:     {'✅ OWNER' if result.is_owner else '❌ NOT OWNER'}")
    print(f"   Confidence: {result.confidence:.4f}")

    print(f"\n📝 TRANSCRIPTION")
    print(f"   Text:     \"{result.text}\"")
    print(f"   Language: {result.language} (p={result.language_probability:.3f})")

    print(f"\n💡 RECOMMENDATION")
    if result.is_owner:
        print(f"   -> Execute command: \"{result.text}\"")
    else:
        print(f"   -> DENY: voice does not match owner profile")
        print(f"   -> Recognized text: \"{result.text}\"")

    print("=" * 70)

    jarvis.cleanup(result)


if __name__ == "__main__":
    main()