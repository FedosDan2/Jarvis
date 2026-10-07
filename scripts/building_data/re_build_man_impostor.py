#!/usr/bin/env python3
"""
Объединяет impostor-манифесты в один общий CSV.

Вход:
    data/impostor/ru/metadata.csv
    data/impostor/en/metadata.csv

Ожидаемые входные колонки:
    file, client_id, language, duration, sentence

Выход:
    data/impostor/metadata.csv

Схема выхода:
    file,speaker_type,client_id,language,duration,sentence

Правила:
    speaker_type = IMPOSTOR
    client_id replaces original long client_id with IMPOSTOR_001, IMPOSTOR_002, ...

Дополнительно:
    data/impostor/client_id_mapping.csv
    mapped_client_id,original_client_id,languages,num_clips,first_file
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


OUTPUT_FIELDS = [
    "file",
    "speaker_type",
    "client_id",
    "language",
    "duration",
    "sentence",
]

MAPPING_FIELDS = [
    "mapped_client_id",
    "original_client_id",
    "languages",
    "num_clips",
    "first_file",
]

INPUT_REQUIRED = {
    "file",
    "client_id",
    "language",
    "duration",
}


def _clean(value) -> str:
    return (value or "").strip()


def _format_duration(value) -> str:
    """
    Приводим duration к нормальному числовому виду.
    Если не получается — оставляем как есть.
    """
    raw = _clean(value)
    try:
        return f"{float(raw):.3f}"
    except Exception:
        return raw


def read_one_metadata(path: Path, fallback_language: str | None = None):
    """
    Читает один metadata.csv и возвращает нормализованные записи.
    """
    if not path.exists():
        return [], [f"НЕ НАЙДЕН файл: {path}"]

    records = []
    warnings = []
    seen_keys = set()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])

        missing = INPUT_REQUIRED - cols
        if missing:
            return [], [
                f"{path}: отсутствуют обязательные колонки {sorted(missing)}. "
                f"Найдены: {sorted(cols)}"
            ]

        for line_no, row in enumerate(reader, start=2):
            file_name = _clean(row.get("file"))
            original_client_id = _clean(row.get("client_id"))
            language = _clean(row.get("language")) or (fallback_language or "")
            duration = row.get("duration")
            sentence = _clean(row.get("sentence"))

            if not file_name:
                warnings.append(f"{path}:{line_no}: пустой file, строка пропущена")
                continue

            if not original_client_id:
                warnings.append(
                    f"{path}:{line_no}: пустой client_id, строка пропущена"
                )
                continue

            if not language:
                warnings.append(
                    f"{path}:{line_no}: пустой language, строка пропущена"
                )
                continue

            language = language.lower()

            # Если в file случайно лежит путь, оставляем только имя файла.
            if "/" in file_name or "\\" in file_name:
                warnings.append(
                    f"{path}:{line_no}: file содержит путь, беру basename: {file_name}"
                )
                file_name = Path(file_name).name

            key = (language, file_name)
            if key in seen_keys:
                warnings.append(
                    f"{path}:{line_no}: дубликат (language={language}, file={file_name}), пропущен"
                )
                continue

            seen_keys.add(key)

            records.append({
                "file": file_name,
                "original_client_id": original_client_id,
                "language": language,
                "duration": _format_duration(duration),
                "sentence": sentence,
                "source_metadata": str(path),
            })

    return records, warnings


def build_client_id_mapping(records, prefix: str, min_width: int):
    """
    Строит глобальный маппинг original_client_id -> IMPOSTOR_001...

    Важно:
        - один и тот же original_client_id из ru и en получит один и тот же IMPOSTOR_xxx;
        - нумерация детерминированная: сортировка по original_client_id.
    """
    unique_original_ids = sorted({r["original_client_id"] for r in records})

    width = max(min_width, len(str(len(unique_original_ids))))

    mapping = {
        original_id: f"{prefix}{idx + 1:0{width}d}"
        for idx, original_id in enumerate(unique_original_ids)
    }

    return mapping, unique_original_ids, width


def build_mapping_stats(records, mapping):
    """
    Собирает статистику по оригинальным client_id для mapping-файла.
    """
    stats = defaultdict(lambda: {
        "languages": set(),
        "num_clips": 0,
        "first_file": None,
    })

    for r in records:
        original_id = r["original_client_id"]
        item = stats[original_id]

        item["languages"].add(r["language"])
        item["num_clips"] += 1

        if item["first_file"] is None:
            item["first_file"] = r["file"]

    rows = []

    for original_id, item in stats.items():
        rows.append({
            "mapped_client_id": mapping[original_id],
            "original_client_id": original_id,
            "languages": ",".join(sorted(item["languages"])),
            "num_clips": item["num_clips"],
            "first_file": item["first_file"],
        })

    rows.sort(key=lambda x: x["mapped_client_id"])
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Merge impostor ru/en metadata.csv into one common manifest."
    )

    ap.add_argument(
        "--impostor-dir",
        type=str,
        default="data/impostor",
        help="Папка data/impostor",
    )

    ap.add_argument(
        "--languages",
        nargs="+",
        default=["ru", "en"],
        help="Языковые подпапки, из которых читаем metadata.csv",
    )

    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="Куда сохранить объединённый metadata.csv. По умолчанию: <impostor-dir>/metadata.csv",
    )

    ap.add_argument(
        "--mapping-output",
        type=str,
        default="",
        help="Куда сохранить mapping original_client_id -> IMPOSTOR_xxx. "
             "По умолчанию: <impostor-dir>/client_id_mapping.csv",
    )

    ap.add_argument(
        "--speaker-type",
        type=str,
        default="IMPOSTOR",
        help="Значение колонки speaker_type для всех impostor-строк",
    )

    ap.add_argument(
        "--id-prefix",
        type=str,
        default="IMPOSTOR_",
        help="Префикс для новых client_id",
    )

    ap.add_argument(
        "--id-width",
        type=int,
        default=3,
        help="Минимальная ширина номера: IMPOSTOR_001, IMPOSTOR_002...",
    )

    args = ap.parse_args()

    impostor_dir = Path(args.impostor_dir).expanduser().resolve()

    output_path = Path(args.output).expanduser().resolve() if args.output else impostor_dir / "metadata.csv"
    mapping_output_path = (
        Path(args.mapping_output).expanduser().resolve()
        if args.mapping_output
        else impostor_dir / "client_id_mapping.csv"
    )

    print("=== MERGE IMPOSTOR MANIFESTS ===")
    print(f"impostor_dir: {impostor_dir}")
    print(f"languages:    {args.languages}")
    print(f"output:       {output_path}")
    print(f"mapping:      {mapping_output_path}")
    print()

    all_records = []
    all_warnings = []

    for language in args.languages:
        metadata_path = impostor_dir / language / "metadata.csv"

        print(f"Читаю: {metadata_path}")

        records, warnings = read_one_metadata(metadata_path, fallback_language=language)

        all_records.extend(records)
        all_warnings.extend(warnings)

        unique_in_file = len({r["original_client_id"] for r in records})

        print(f"  записей: {len(records)}")
        print(f"  уникальных original client_id: {unique_in_file}")
        print()

    if not all_records:
        raise SystemExit("Не удалось прочитать ни одной корректной записи.")

    if all_warnings:
        print("=== WARNINGS ===")
        for warning in all_warnings[:100]:
            print(f"  ⚠️ {warning}")

        if len(all_warnings) > 100:
            print(f"  ... и ещё {len(all_warnings) - 100}")
        print()

    mapping, unique_original_ids, width = build_client_id_mapping(
        records=all_records,
        prefix=args.id_prefix,
        min_width=args.id_width,
    )

    print("=== MAPPING SUMMARY ===")
    print(f"unique original client_id: {len(unique_original_ids)}")
    print(f"id prefix: {args.id_prefix}")
    print(f"id width: {width}")
    print(f"example ids: {list(mapping.values())[:5]}")
    print()

    output_rows = []

    for r in all_records:
        output_rows.append({
            "file": r["file"],
            "speaker_type": args.speaker_type,
            "client_id": mapping[r["original_client_id"]],
            "language": r["language"],
            "duration": r["duration"],
            "sentence": r["sentence"],
        })

    # Удобный порядок: сначала по новому client_id, потом язык, потом файл.
    output_rows.sort(key=lambda x: (x["client_id"], x["language"], x["file"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    mapping_rows = build_mapping_stats(all_records, mapping)

    mapping_output_path.parent.mkdir(parents=True, exist_ok=True)

    with mapping_output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MAPPING_FIELDS)
        writer.writeheader()
        writer.writerows(mapping_rows)

    print("=== DONE ===")
    print(f"Сохранено записей: {len(output_rows)}")
    print(f"Уникальных impostor-профилей: {len(mapping)}")
    print(f"Объединённый манифест: {output_path}")
    print(f"Mapping original ids: {mapping_output_path}")
    print()

    print("=== COUNTS BY LANGUAGE ===")
    counts = defaultdict(int)
    speakers = defaultdict(set)

    for row in output_rows:
        counts[row["language"]] += 1
        speakers[row["language"]].add(row["client_id"])

    for language in sorted(counts):
        print(f"  {language}: files={counts[language]}, speakers={len(speakers[language])}")


if __name__ == "__main__":
    main()