#!/usr/bin/env python3
"""
Оценка системы распознавания владельца ("Я / Не Я").

Вход: data/embeddings/embeddings.npz
Выход: 
  - Метрики EER, FAR@FRR, FRR@FAR на консоли.
  - Гистограммы скоров (опционально, если matplotlib установлен).
  - Оптимальный порог для runtime-агента.

Логика:
  1. Профиль владельца строится ТОЛЬКО из owner/split==train.
  2. Для каждого клипа (owner val/test + impostor all) считается score = max(cosine(emb, profile)).
     Используем MAX, потому что один хороший сегмент должен идентифицировать человека.
  3. Порог подбирается по кривой DET (Trade-off FAR/FRR) или фиксируется вручную.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Опциональный импорт для графиков
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_embeddings(path: Path):
    """Загружает .npz и возвращает массивы."""
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    
    data = np.load(path)
    
    embs = data["embeddings"]          # (N, D)
    speaker_type = data["speaker_type"] # (N,) 'owner'/'impostor'
    client_id = data["client_id"]       # (N,) ID спикера
    split = data["split"]               # (N,) 'train'/'val'/'test'
    speech_score = data["speech_score"] # (N,) качество аудио
    
    return embs, speaker_type, client_id, split, speech_score


def build_owner_profile(embs, speaker_type, split):
    """
    Извлекает эмбеддинги владельца из TRAIN сплита.
    Возвращает тензор (K, D), где K - количество эталонов.
    """
    mask = (speaker_type == "owner") & (split == "train")
    profile_embs = embs[mask]
    
    if len(profile_embs) == 0:
        raise ValueError("No owner embeddings found in train split!")
        
    print(f"👤 Owner Profile built from {len(profile_embs)} training clips.")
    
    # Нормализуем еще раз на всякий случай (хотя extract_emb уже нормализовал)
    t_embs = torch.from_numpy(profile_embs).float()
    t_embs = F.normalize(t_embs, dim=1)
    
    return t_embs


def calculate_scores(all_embs, profile_tensor):
    """
    Для каждого эмбеддинга в all_embs считает максимальный cosine similarity
    к любому вектору в profile_tensor.
    
    Args:
        all_embs: np.array (N, D)
        profile_tensor: torch.Tensor (K, D)
        
    Returns:
        scores: np.array (N,) - макс. сходство с профилем
    """
    t_all = torch.from_numpy(all_embs).float()
    t_all = F.normalize(t_all, dim=1)
    
    # Перемножаем матрицы: (N, D) @ (D, K) -> (N, K)
    # Каждый элемент [i, j] = cosine_sim(clip_i, ref_j)
    sims = torch.mm(t_all, profile_tensor.T)
    
    # Берем максимум по всем референсам для каждого клипа
    # Логика: если хоть один сегмент очень похож, считаем, что это тот же человек
    max_sims = sims.max(dim=1).values.cpu().numpy()
    
    return max_sims


def find_optimal_threshold(scores_owner_val, scores_impostor_val):
    """
    Ищет порог, минимизирующий |FAR - FRR| (ближе всего к EER).
    """
    # Объединяем все возможные пороги (уникальные значения скоров)
    thresholds = np.unique(np.concatenate([scores_owner_val, scores_impostor_val]))
    
    best_far = 1.0
    best_frr = 1.0
    best_thr = 0.0
    min_diff = float('inf')
    
    for thr in thresholds:
        # Owner: если score < thr -> reject (error)
        frr = np.mean(scores_owner_val < thr)
        # Impostor: если score >= thr -> accept (error)
        far = np.mean(scores_impostor_val >= thr)
        
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            best_far = far
            best_frr = frr
            best_thr = thr
            
    return best_thr, best_far, best_frr


def evaluate_metrics(scores_owner_test, scores_impostor_test, threshold):
    """
    Считает финальные метрики на TEST сплите при заданном пороге.
    """
    # Owner Errors (Reject valid user)
    frr = np.mean(scores_owner_test < threshold)
    
    # Impostor Errors (Accept invalid user)
    far = np.mean(scores_impostor_test >= threshold)
    
    # Accuracy overall (weighted by class balance usually, but here simple acc)
    total_samples = len(scores_owner_test) + len(scores_impostor_test)
    correct_owner = np.sum(scores_owner_test >= threshold)
    correct_impostor = np.sum(scores_impostor_test < threshold)
    accuracy = (correct_owner + correct_impostor) / total_samples
    
    return frr, far, accuracy


def plot_distributions(scores_owner, scores_impostor, title_suffix=""):
    if not HAS_MPL:
        print("⚠️ Matplotlib not installed. Skipping plots.")
        return
        
    plt.figure(figsize=(10, 6))
    bins = np.linspace(-1, 1, 100)
    
    plt.hist(scores_owner, bins=bins, alpha=0.5, label=f"Owner ({title_suffix})", color='blue', density=True)
    plt.hist(scores_impostor, bins=bins, alpha=0.5, label=f"Impostor ({title_suffix})", color='red', density=True)
    
    plt.title(f"Score Distribution: Owner vs Impostor ({title_suffix})")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="data/embeddings/embeddings.npz")
    ap.add_argument("--plot", action="store_true", help="Show histograms")
    args = ap.parse_args()

    project_root = Path(".").resolve()
    emb_path = Path(args.embeddings).expanduser()
    if not emb_path.is_absolute():
        emb_path = project_root / emb_path

    print("=== LOADING DATA ===")
    embs, stypes, cids, splits, sscores = load_embeddings(emb_path)
    print(f"Loaded {len(embs)} embeddings.")

    # Разделяем данные по типам и сплитам
    # 1. Профиль (Train Owner)
    profile_mask = (stypes == "owner") & (splits == "train")
    profile_embs = embs[profile_mask]
    
    # 2. Валидация (для подбора порога)
    val_owner_mask = (stypes == "owner") & (splits == "val")
    val_imp_mask = (stypes == "impostor") & (splits == "val") # Или все impostors, если val мало
    
    # Если val-owner пустой (как было раньше с EN), используем test-owner для калибровки? 
    # Нет, лучше использовать все доступные non-train owner clips для калибровки, 
    # а impostors брать из тех же сплитов или всех сразу.
    # Для простоты MVP: Калибруем на Val, Тестируем на Test.
    # Если Val пуст, берем часть Test для калибровки (не идеально, но рабочий вариант).
    
    if not val_owner_mask.any():
        print("⚠️ WARNING: No OWNER clips in VAL split. Using TEST owner clips for calibration too.")
        # В этом случае мы будем калиброваться на тесте (утечка информации, но для демо ок)
        calib_owner_mask = (stypes == "owner") & ((splits == "val") | (splits == "test"))
        calib_imp_mask = (stypes == "impostor") & ((splits == "val") | (splits == "test"))
        eval_owner_mask = None # Не будет отдельного теста
        eval_imp_mask = None
    else:
        calib_owner_mask = val_owner_mask
        calib_imp_mask = val_imp_mask
        eval_owner_mask = (stypes == "owner") & (splits == "test")
        eval_imp_mask = (stypes == "impostor") & (splits == "test")

    print("\n=== BUILDING PROFILE ===")
    profile_tensor = build_owner_profile(embs, stypes, splits)

    print("\n=== CALCULATING SCORES ===")
    
    # Scores для калибровки
    calib_owner_embs = embs[calib_owner_mask]
    calib_imp_embs = embs[calib_imp_mask]
    
    scores_calib_owner = calculate_scores(calib_owner_embs, profile_tensor)
    scores_calib_imp = calculate_scores(calib_imp_embs, profile_tensor)
    
    print(f"Calibration set: {len(scores_calib_owner)} owner, {len(scores_calib_imp)} impostor.")

    print("\n=== FINDING THRESHOLD ===")
    thr, eer_far, eer_frr = find_optimal_threshold(scores_calib_owner, scores_calib_imp)
    print(f"Optimal Threshold (near EER): {thr:.4f}")
    print(f"At this threshold: FAR={eer_far:.4f}, FRR={eer_frr:.4f}")

    # Финальная оценка на Test (если есть отдельный тест)
    if eval_owner_mask is not None and eval_imp_mask is not None:
        print("\n=== FINAL EVALUATION ON TEST SET ===")
        eval_owner_embs = embs[eval_owner_mask]
        eval_imp_embs = embs[eval_imp_mask]
        
        scores_eval_owner = calculate_scores(eval_owner_embs, profile_tensor)
        scores_eval_imp = calculate_scores(eval_imp_embs, profile_tensor)
        
        final_frr, final_far, final_acc = evaluate_metrics(scores_eval_owner, scores_eval_imp, thr)
        
        print(f"Test Set Size: {len(scores_eval_owner)} owner, {len(scores_eval_imp)} impostor.")
        print(f"FRR (False Reject Rate): {final_frr:.4f} ({final_frr*100:.2f}%)")
        print(f"FAR (False Accept Rate): {final_far:.4f} ({final_far*100:.2f}%)")
        print(f"Accuracy: {final_acc:.4f}")
        
        # Доп. метрики при разных строгих порогах
        print("\n--- Metrics at strict thresholds ---")
        for strict_thr in [0.6, 0.7, 0.8]:
            f_strict, fa_strict, _ = evaluate_metrics(scores_eval_owner, scores_eval_imp, strict_thr)
            print(f"Thr={strict_thr}: FRR={f_strict:.4f}, FAR={fa_strict:.4f}")

    # Визуализация
    if args.plot:
        # Рисуем распределение на том наборе, где есть оба класса
        plot_distributions(scores_calib_owner, scores_calib_imp, "Calibration")
        if eval_owner_mask is not None:
             plot_distributions(scores_eval_owner, scores_eval_imp, "Test")

    print("\n✅ Evaluation Complete.")
    print(f"Recommended Runtime Threshold: {thr:.4f}")


if __name__ == "__main__":
    main()