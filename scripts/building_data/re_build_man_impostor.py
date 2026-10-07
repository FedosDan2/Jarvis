#!/usr/bin/env python3
"""
Копирует metadata из data/impostor/ru/metadata.csv в data/impostor/metadata.csv
с добавлением speaker_type=IMPOSTOR и нормализацией client_id.
"""
import csv
from pathlib import Path

INPUT_META = Path("data/impostor/ru/metadata.csv")
OUTPUT_META = Path("data/impostor/metadata.csv")

FIELDS_OUT = ["file", "speaker_type", "client_id", "language", "duration", "sentence"]

def main():
    if not INPUT_META.exists():
        raise SystemExit(f"Не найден входной файл: {INPUT_META}")
        
    OUTPUT_META.parent.mkdir(parents=True, exist_ok=True)
    
    unique_ids = set()
    mapping = {}
    
    # Первый проход: собираем уникальные ID и делаем маппинг IMPOSTOR_XXX
    with INPUT_META.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("client_id", "").strip()
            if cid and cid not in mapping:
                unique_ids.add(cid)
                
    # Сортируем для детерминированности
    sorted_ids = sorted(list(unique_ids))
    for i, original_id in enumerate(sorted_ids, start=1):
        new_id = f"IMPOSTOR_{i:03d}"
        mapping[original_id] = new_id
        
    print(f"Найдено уникальных спикеров: {len(mapping)}")
    
    # Второй проход: пишем выходной файл
    count = 0
    with INPUT_META.open("r", encoding="utf-8", newline="") as fin, \
         OUTPUT_META.open("w", encoding="utf-8", newline="") as fout:
         
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=FIELDS_OUT)
        writer.writeheader()
        
        for row in reader:
            orig_cid = row.get("client_id", "").strip()
            new_cid = mapping.get(orig_cid, "UNKNOWN")
            
            writer.writerow({
                "file": row.get("file", ""),
                "speaker_type": "impostor",
                "client_id": new_cid,
                "language": "ru",  # Жестко фиксируем RU
                "duration": row.get("duration", "0"),
                "sentence": row.get("sentence", ""),
            })
            count += 1
            
    print(f"Сохранено строк: {count} в {OUTPUT_META}")

if __name__ == "__main__":
    main()