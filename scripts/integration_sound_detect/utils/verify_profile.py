#!/usr/bin/env python3
"""
Адаптивная система верификации голоса.
Поддерживает режимы: только проверка, проверка с авто-обучением, сухой прогон.
"""
import json
import logging
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

from utils.audio_preprocessor import AudioPreprocessor

# Настройки логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AdaptiveIdentityService:
    def __init__(
        self,
        model_source="speechbrain/spkrec-ecapa-voxceleb",
        profile_path="models/identity_service/owner_profile.npy",
        config_path="models/identity_service/config.json",
        high_confidence_threshold=0.7,  
        verify_threshold=0.5855          
    ):
        self.profile_path = Path(profile_path)
        self.config_path = Path(config_path)
        
        # ... загрузка модели ...
        try:
            self.model = EncoderClassifier.from_hparams(
                source=model_source, 
                savedir="pretrained_models/ecapa",
                run_opts={"device": DEVICE}
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        # *** ДОБАВЛЕНО: Инициализация препроцессора ***
        self.preprocessor = AudioPreprocessor(device=DEVICE)
        
        # ... остальная инициализация профиля ...
        if not self.profile_path.exists():
            raise FileNotFoundError(f"Profile file not found: {self.profile_path}")
        
        self.profile_vectors = np.load(self.profile_path)
        
        self.high_confidence_threshold = high_confidence_threshold
        self.verify_threshold = verify_threshold
        self.stats = {"accepted": 0, "rejected": 0, "updated_profile": 0}

    def _get_embedding(self, audio_path: str) -> np.ndarray | None:
        """
        Извлекает эмбеддинг, автоматически обрабатывая любой формат аудио.
        """
        try:
            # 1. Получаем чистый тензор 16k mono через препроцессор
            # Это заменяет весь старый код с torchaudio.load, resample и mean
            wav_tensor = self.preprocessor.preprocess(audio_path)
            
            # 2. Прогоняем через модель
            with torch.no_grad():
                emb = self.model.encode_batch(wav_tensor).squeeze()
                
            # 3. Нормализация для cosine similarity
            emb = F.normalize(emb, dim=0)
            
            # 4. Возврат на CPU для NumPy операций сравнения
            return emb.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Failed to process audio {audio_path}: {e}")
            return None

    def _calculate_score(self, test_emb: np.ndarray) -> float:
        """Считает максимальное сходство с любым вектором в профиле."""
        if len(self.profile_vectors) == 0:
            return 0.0
            
        eps=1e-2
        # Dot product между нормализованными векторами = Cosine Similarity
        sims = np.dot(self.profile_vectors, test_emb)
        res = float(np.max(sims))
        if (res == 1.0):
            print(' ---  Запись уже имеется в базе, не добавляем')
        return res, (1.0 - res <= eps)

    def _update_profile(self, new_vector: np.ndarray):
        """Добавляет вектор в профиль и сохраняет на диск."""
        # Добавляем новый вектор
        updated_profile = np.vstack([self.profile_vectors, new_vector])
        self.profile_vectors = updated_profile
        
        # Сохраняем обратно
        np.save(self.profile_path, self.profile_vectors)
        logger.info(f"✅ Profile UPDATED. New size: {len(self.profile_vectors)} vectors.")
        
        # Обновляем конфиг
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                config['num_reference_vectors'] = int(len(self.profile_vectors))
                config['last_updated'] = datetime.datetime.now().isoformat()
                
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Could not update config.json: {e}")

    def verify_and_adapt(
        self, 
        audio_path: str, 
        enable_adaptation: bool = True,
        dry_run: bool = False
    ) -> tuple[bool, float]:
        """
        Основная функция.
        
        Args:
            audio_path: Путь к аудио файлу.
            enable_adaptation: Если True, разрешено обновление профиля.
            dry_run: Если True, изменения НЕ сохраняются на диск (только в память).
            
        Returns:
            (is_owner, score)
        """
        test_emb = self._get_embedding(audio_path)
        if test_emb is None:
            return False, 0.0
            
        score, add_flag = self._calculate_score(test_emb)
        is_owner = (score >= self.verify_threshold)
        
        log_prefix = "[DRY-RUN] " if dry_run else ""

        if is_owner:
            self.stats["accepted"] += 1
            logger.info(f"{log_prefix}ACCESS GRANTED. Score: {score:.4f}")
            
            # Логика адаптации
            should_update = (
                enable_adaptation and 
                not dry_run and 
                score >= self.high_confidence_threshold
            )
            
            if should_update and add_flag != 1:
                logger.info("High confidence detected. Adding sample to profile...")
                self._update_profile(test_emb)
                self.stats["updated_profile"] += 1
            elif should_update and add_flag == 1:
                logger.info("Very high confidence detected - voice massege already added to base.")
            elif enable_adaptation and score >= self.high_confidence_threshold:
                # Это случай dry_run или если мы хотим просто показать что бы произошло
                logger.info(f"{log_prefix}[SIMULATION] Would add vector to profile (Score {score:.4f} >= {self.high_confidence_threshold})")
            else:
                logger.debug("Confidence moderate or adaptation disabled. Not updating profile.")
                
        else:
            self.stats["rejected"] += 1
            
        return is_owner, score


# def main():
#     parser = argparse.ArgumentParser(description="Verify voice identity against owner profile.")
    
#     parser.add_argument(
#         "--file", 
#         type=str, 
#         required=True,
#         help="Path to the audio file to verify."
#     )
    
#     parser.add_argument(
#         "--no-adapt",
#         action="store_true",
#         help="Disable automatic profile updates (verification only)."
#     )
    
#     parser.add_argument(
#         "--dry-run",
#         action="store_true",
#         help="Simulate verification without saving changes to disk."
#     )
    
#     args = parser.parse_args()

#     # Инициализация сервиса
#     service = AdaptiveIdentityService()
    
#     test_file = Path(args.file)
#     if not test_file.exists():
#         print(f"File not found: {test_file}")
#         exit(1)

#     print("\n=== STARTING VERIFICATION ===")
    
#     # Вызов метода с учетом флагов
#     is_owner, score = service.verify_and_adapt(
#         audio_path=str(test_file),
#         enable_adaptation=not args.no_adapt,
#         dry_run=args.dry_run
#     )
    
#     status = "✅ ACCESS GRANTED" if is_owner else "❌ ACCESS DENIED"
#     mode = "DRY RUN" if args.dry_run else ("ADAPTIVE" if not args.no_adapt else "STRICT")
    
#     print(f"\nResult: {status}")
#     print(f"Mode:   {mode}")
#     print(f"Score:  {score:.4f}")
#     print(f"Stats:  Accepted={service.stats['accepted']}, Rejected={service.stats['rejected']}, Updated={service.stats['updated_profile']}")


# if __name__ == "__main__":
#     main()