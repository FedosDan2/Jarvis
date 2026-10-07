import subprocess
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torchaudio

class AudioPreprocessor:
    """
    Универсальный препроцессор аудио для Jarvis Identity Service.
    
    Принимает путь к файлу любого формата (.ogg, .mp3, .m4a, .wav, .flac и т.д.)
    и возвращает нормализованный torch.Tensor (float32, mono, 16kHz).
    
    Если входной файл не является стандартным WAV 16k mono, 
    скрипт использует ffmpeg для конвертации во временный файл,
    загружает его, а затем удаляет временный файл.
    """

    TARGET_SR = 16000
    
    # Расширения, которые torchaudio/soundfile обычно читает нативно без проблем
    NATIVE_EXTENSIONS = {".wav"} 
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Проверяем наличие ffmpeg один раз при инициализации
        if not shutil.which("ffmpeg"):
            raise EnvironmentError(
                "FFmpeg is required for audio preprocessing but not found in PATH. "
                "Install it via 'sudo apt install ffmpeg' or equivalent."
            )

    def preprocess(self, file_path: str | Path) -> torch.Tensor:
        """
        Главная точка входа. Возвращает tensor shape (1, num_samples) на нужном устройстве.
        
        Args:
            file_path: Путь к исходному аудиофайлу.
            
        Returns:
            torch.Tensor: Аудио-тензор float32, mono, 16kHz.
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path_obj}")

        ext = path_obj.suffix.lower()
        
        # Если это нативный формат, пробуем читать напрямую
        # Но даже WAV может быть стерео или 44.1k, поэтому логика ниже все равно нужна
        # Однако для скорости: если это .wav, можно попробовать торчаудио сразу
        
        temp_wav_path = None
        
        try:
            # 1. Проверка необходимости конвертации
            # Мы всегда приводим к 16k mono. 
            # Если файл уже wav, но параметры другие — тоже лучше пересоздать через ffmpeg,
            # чтобы гарантировать совместимость с backend soundfile/torchaudio
            
            needs_conversion = True
            
            # Быстрая эвристика: если это .wav, попробуем прочитать инфо
            if ext == ".wav":
                try:
                    info = torchaudio.info(str(path_obj))
                    if info.sample_rate == self.TARGET_SR and info.num_channels == 1:
                        needs_conversion = False
                except Exception:
                    pass # Если ошибка чтения инфо, значит нужен ffmpeg
            
            if needs_conversion:
                temp_wav_path = self._convert_with_ffmpeg(path_obj)
                load_path = temp_wav_path
            else:
                load_path = str(path_obj)

            # 2. Загрузка аудио
            # Используем backend 'soundfile' так как он стабильнее всего работает с PCM16 WAV
            try:
                torchaudio.set_audio_backend("soundfile")
            except Exception:
                pass # Игнорируем, если бэкенд уже установлен или недоступен
                
            waveform, sr = torchaudio.load(load_path)
            
            # 3. Финальная нормализация (на всякий случай)
            # Каналы
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Частота
            if sr != self.TARGET_SR:
                waveform = torchaudio.functional.resample(waveform, sr, self.TARGET_SR)
                
            # Тип данных
            waveform = waveform.float()
            
            # Перенос на устройство
            return waveform.to(self.device)

        finally:
            # 4. Очистка временных файлов
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except OSError:
                    pass # Не критично, если не удалось удалить

    def _convert_with_ffmpeg(self, input_path: Path) -> str:
        """
        Конвертирует любой аудиофайл в WAV 16kHz Mono PCM16 во временную директорию.
        Возвращает путь к созданному временному файлу.
        """
        # Создаем временный файл с расширением .wav
        fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="jarvis_proc_")
        os.close(fd) # Закрываем дескриптор, ffmpeg откроет сам по пути
        
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y", # Переписать если существует (хотя temp уникальный)
            "-i", str(input_path),
            "-vn", # Отключить видео
            "-acodec", "pcm_s16le", # Кодек аудио: PCM 16-bit Little Endian
            "-ar", str(self.TARGET_SR), # Sample rate: 16000 Hz
            "-ac", "1", # Channels: 1 (Mono)
            temp_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return temp_path
        except subprocess.CalledProcessError as e:
            # Если ffmpeg упал, удаляем пустой/битый temp файл и кидаем ошибку
            if os.path.exists(temp_path):
                os.remove(temp_path)
            error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "Unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg conversion failed for {input_path}: {error_msg}")


# Импорт shutil нужен для проверки наличия ffmpeg
import shutil