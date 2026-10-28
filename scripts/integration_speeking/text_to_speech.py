#!/usr/bin/env python3
import gc
import io
import re
import hashlib
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch


class TTSservice:
    """
    Озвучка ответов агента.

    Бэкенд по умолчанию: Silero ru_v3.
    Модель грузится лениво: только когда реально нужен generate().
    Это позволяет не держать TTS в VRAM постоянно.
    """

    # Простая замена одиночных цифр на слова.
    # Для полноценной нормализации чисел лучше позже подключить num2words или собственный number-to-words.
    _NUM_WORDS = {
        "0": "ноль",
        "1": "один",
        "2": "два",
        "3": "три",
        "4": "четыре",
        "5": "пять",
        "6": "шесть",
        "7": "семь",
        "8": "восемь",
        "9": "девять",
    }

    def __init__(
        self,
        backend: str = "silero",
        sample_rate: int = 48000,
        cache_size: int = 100,
        cache_dir: Optional[str] = "models/tts_cache",
        speaker: str = "eugene",
        max_chars_per_chunk: int = 250,      # <-- НОВОЕ: первый срез по длине
        inter_chunk_pause_sec: float = 0.18, # <-- НОВОЕ: пауза между чанками
    ):
        self.max_chars_per_chunk = int(max_chars_per_chunk)
        self.inter_chunk_pause_sec = float(inter_chunk_pause_sec)
        self.backend = (backend or "silero").strip().lower()
        self.sample_rate = int(sample_rate)
        self.speaker = speaker
        self.cache_size = int(cache_size)
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None

        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._load_lock = threading.Lock()
        self._cache_lock = threading.Lock()

        self.model = None
        self._load_error: Optional[Exception] = None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._warm_disk_cache()

    # ------------------------------------------------------------------
    # Загрузка / выгрузка
    # ------------------------------------------------------------------
    def _load_model(self):
        with self._load_lock:
            if self.model is not None:
                return

            if self.backend != "silero":
                raise NotImplementedError(
                    f"Backend '{self.backend}' пока не подключён. "
                    f"Доступен: silero."
                )

            device = "cuda" if torch.cuda.is_available() else "cpu"

            try:
                self.model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-models",
                    model="silero_tts",
                    language="ru",
                    speaker="ru_v3",
                    trust_repo=True,
                )

                if hasattr(self.model, "to"):
                    self.model.to(torch.device(device))

                self._load_error = None

            except Exception as e:
                self._load_error = e
                self.model = None
                raise RuntimeError(f"Не удалось загрузить TTS ({self.backend}): {e}") from e

    def _ensure_loaded(self):
        if self.model is None:
            if self._load_error is not None:
                raise RuntimeError(f"TTS не загружен: {self._load_error}")
            self._load_model()

    def warmup(self):
        """
        Принудительно загрузить модель, если хочешь прогреть перед первым запросом.
        """
        self._ensure_loaded()

    def unload(self):
        """
        Выгрузить модель из памяти.
        Полезно, если TTS нужен только эпизодически.
        """
        with self._load_lock:
            self.model = None

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Кэш
    # ------------------------------------------------------------------
    def _cache_key(self, text: str, speaker: str) -> str:
        raw = f"{self.backend}|{speaker}|{self.sample_rate}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _warm_disk_cache(self):
        if not self.cache_dir:
            return

        files = list(self.cache_dir.glob("*.wav"))

        # Грузим не всё подряд, а только последние cache_size файлов.
        files = files[-self.cache_size:]

        for p in files:
            try:
                data, sr = sf.read(str(p), dtype="float32")
                buf = io.BytesIO()
                sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
                buf.seek(0)
                self._cache[p.stem] = buf.getvalue()
            except Exception:
                continue

    def _cache_get(self, key: str) -> Optional[bytes]:
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _cache_put(self, key: str, value: bytes):
        with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)

            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        if self.cache_dir:
            try:
                path = self.cache_dir / f"{key}.wav"
                with open(path, "wb") as f:
                    f.write(value)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Нормализация текста
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        """
        Грубая, но полезная нормализация для Silero.
        Убирает markdown, эмодзи, лишние символы, частично приводит числа.
        """
        if not text:
            return ""

        text = str(text).strip()

        # Убираем markdown
        text = re.sub(r"[*_`#~]+", " ", text)

        # Замена частых символов на слова
        text = text.replace("%", " процентов ")
        text = text.replace("₽", " рублей ")
        text = text.replace("$", " долларов ")
        text = text.replace("€", " евро ")
        text = text.replace("№", " номер ")
        text = text.replace("&", " и ")
        text = text.replace("+", " плюс ")

        # Оставляем буквы, цифры, пробелы и базовую пунктуацию
        text = re.sub(r"[^\w\s.,!?;:()-]", " ", text, flags=re.UNICODE)

        # Схлопываем пробелы
        text = re.sub(r"\s+", " ", text).strip()

        # Одиночные цифры -> слова
        # Важно: многозначные числа типа 2026 не трогаем, иначе будет "два ноль два шесть".
        text = re.sub(
            r"(?<!\d)([0-9])(?!\d)",
            lambda m: self._NUM_WORDS.get(m.group(1), m.group(1)),
            text,
        )

        return text


        # ---------------- нарезка текста на чанки ----------------
    def _chunk_text(self, text: str, max_chars: int) -> list[str]:
        """
        Режет нормализованный текст на чанки <= max_chars по границам предложений.
        Знаки препинания сохраняются (Silero по ним ставит интонацию/паузы).
        """
        # режем ПОСЛЕ . ! ? … , оставляя знак в предыдущем куске
        sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
        sentences = [s for s in sentences if s]

        chunks: list[str] = []
        cur = ""
        for sent in sentences:
            if len(sent) > max_chars:
                if cur:
                    chunks.append(cur)
                    cur = ""
                chunks.extend(self._hard_split_words(sent, max_chars))
            elif len(cur) + len(sent) + 1 <= max_chars:
                cur = (cur + " " + sent).strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = sent
        if cur:
            chunks.append(cur)

        return chunks or [text]

    @staticmethod
    def _hard_split_words(sentence: str, max_chars: int) -> list[str]:
        """Жёсткая резка одного очень длинного предложения по словам."""
        words = sentence.split()
        out: list[str] = []
        cur = ""
        for w in words:
            if len(cur) + len(w) + 1 <= max_chars:
                cur = (cur + " " + w).strip()
            else:
                if cur:
                    out.append(cur)
                cur = w
        if cur:
            out.append(cur)
        return out or [sentence]

    @staticmethod
    def _split_point(chunk: str) -> int:
        """Точка разреза чанка пополам — по ближайшему пробелу к середине."""
        mid = len(chunk) // 2
        space = chunk.rfind(" ", 0, mid + 1)
        if space == -1:
            space = chunk.find(" ", mid)
        return space if space != -1 else mid

    # ---------------- синтез одного чанка -> numpy ----------------
    def _synthesize_chunk_to_np(self, chunk: str, speaker: str) -> np.ndarray:
        """
        Озвучивает один чанк. Если Silero падает по длине — режет пополам
        и склеивает половинки с паузой (robust fallback).
        """
        with torch.no_grad():
            try:
                audio = self.model.apply_tts(
                    text=chunk,
                    speaker=speaker,
                    sample_rate=self.sample_rate,
                    put_accent=True,
                    put_yo=True,
                )
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                too_long = (
                    "too long" in msg
                    or "size of tensor" in msg
                    or "must match" in msg
                )
                if too_long and len(chunk) > 20:
                    mid = self._split_point(chunk)
                    left, right = chunk[:mid].strip(), chunk[mid:].strip()
                    if not left or not right:
                        raise
                    a1 = self._synthesize_chunk_to_np(left, speaker)
                    a2 = self._synthesize_chunk_to_np(right, speaker)
                    pause = np.zeros(
                        int(self.sample_rate * self.inter_chunk_pause_sec),
                        dtype=np.float32,
                    )
                    return np.concatenate([a1, pause, a2])
                raise

        arr = audio.detach().cpu().numpy().astype(np.float32)
        if arr.ndim > 1:
            arr = arr.squeeze()
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > 1.0:
            arr = arr / peak
        return arr

    @staticmethod
    def _np_to_wav_bytes(arr: np.ndarray, sr: int) -> bytes:
        buf = io.BytesIO()
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > 1.0:
            arr = arr / peak
        sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.getvalue()
    
    # ------------------------------------------------------------------
    # Генерация
    # ------------------------------------------------------------------
    def generate(self, text: str, speaker: Optional[str] = None) -> bytes:
        """
        Возвращает WAV bytes. Длинный текст режется на чанки и склеивается.
        Кэш — по полному нормализованному тексту (как раньше).
        """
        speaker = speaker or self.speaker
        norm = self._normalize(text)
        if not norm:
            return b""

        key = self._cache_key(norm, speaker)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        self._ensure_loaded()

        if self.backend != "silero":
            raise NotImplementedError(f"Backend {self.backend} не подключён.")

        chunks = self._chunk_text(norm, self.max_chars_per_chunk)

        arrays: list[np.ndarray] = []
        for i, ch in enumerate(chunks):
            arrays.append(self._synthesize_chunk_to_np(ch, speaker))
            if i < len(chunks) - 1:
                arrays.append(
                    np.zeros(
                        int(self.sample_rate * self.inter_chunk_pause_sec),
                        dtype=np.float32,
                    )
                )

        full = (
            np.concatenate(arrays)
            if arrays
            else np.zeros(1, dtype=np.float32)
        )

        wav_bytes = self._np_to_wav_bytes(full, self.sample_rate)
        self._cache_put(key, wav_bytes)
        return wav_bytes

    def _tensor_to_wav_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        buf = io.BytesIO()

        arr = audio_tensor.detach().cpu().numpy().astype(np.float32)

        if arr.ndim > 1:
            arr = arr.squeeze()

        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > 1.0:
            arr = arr / peak

        sf.write(buf, arr, self.sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)

        return buf.getvalue()

    # # ------------------------------------------------------------------
    # # Постобработка под "Джарвиса"
    # # ------------------------------------------------------------------
    # def jarvis_fx(self, wav_bytes: bytes) -> bytes:
    #     """
    #     Лёгкая обработка:
    #     - high-pass, чтобы убрать гул;
    #     - подъём верхних частот для металлического блеска;
    #     - мягкая компрессия;
    #     - лимитер.

    #     Если scipy не установлен, вернёт исходный WAV.
    #     """
    #     if not wav_bytes:
    #         return wav_bytes

    #     try:
    #         import scipy.signal as sig
    #     except ImportError:
    #         return wav_bytes

    #     arr, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

    #     if arr.ndim > 1:
    #         arr = arr.mean(axis=1)

    #     if arr.size == 0:
    #         return wav_bytes

    #     # High-pass ~90 Hz
    #     sos_hp = sig.butter(2, 90.0 / (sr / 2.0), btype="highpass", output="sos")
    #     arr = sig.sosfilt(sos_hp, arr)

    #     # Brightness: добавляем немного 2.5 kHz+
    #     sos_bright = sig.butter(1, 2500.0 / (sr / 2.0), btype="highpass", output="sos")
    #     bright = sig.sosfilt(sos_bright, arr) * 0.28
    #     arr = arr + bright

    #     # Мягкая компрессия
    #     thresh = 0.22
    #     abs_arr = np.abs(arr)
    #     gain = np.where(abs_arr > thresh, thresh / (abs_arr + 1e-9), 1.0)
    #     arr = arr * gain

    #     # Лимитер
    #     peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    #     if peak > 0.98:
    #         arr = arr / peak * 0.98

    #     buf = io.BytesIO()
    #     sf.write(buf, arr, sr, format="WAV", subtype="PCM_16")
    #     buf.seek(0)

    #     return buf.getvalue()

    def speak(self, text: str, speaker: Optional[str] = None, apply_fx: bool = True) -> bytes:
        """
        Удобный метод: text -> WAV bytes с эффектом Джарвиса.
        """
        raw = self.generate(text, speaker=speaker)
        return raw


# Совместимость со старым вызовом
def tts_ru(tts_service: TTSservice, text: str, speaker: str = "eugene") -> bytes:
    return tts_service.generate(text, speaker=speaker)