#!/usr/bin/env python3
"""
Валидация data/manifests/manifest.csv.

Ожидаемые колонки:
    audio_path, speaker_type, client_id, language, duration, sentence, session, split

Проверяет:
  - существование файлов;
  - корректность speaker_type / language / split;
  - положительность duration;
  - отсутствие утечек:
      impostor: один client_id не может быть в нескольких сплитах;
      owner:    одна session не может быть в нескольких сплитах.

Дополнительно:
  --deep:
      читает WAV через soundfile и проверяет sample_rate / channels.

  --trim-long:
      если фактическая длительность WAV больше MAX_DUR,
      обрезает файл до MAX_DUR и обновляет duration в manifest.csv.

  --dry-run:
      не менять файлы и manifest, только показать, что будет обрезано.

  --backup:
      перед заменой файла создавать .bak.

Внимание:
  --trim-long реально изменяет WAV-файлы и manifest.csv.
  Сначала запускай с --dry-run.
"""

import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None


REQUIRED = {
    "audio_path",
    "speaker_type",
    "client_id",
    "language",
    "duration",
    "session",
    "split",
}

VALID_SPLITS = {"train", "val", "test"}
VALID_TYPES = {"owner", "impostor"}
VALID_LANGS = {"ru", "en"}

EXPECTED_SR = 16000
EXPECTED_CH = 1

MIN_DUR = 1.5
MAX_DUR = 10.0

# Небольшой допуск для сравнения float-длительностей.
EPS = 1e-3


def _require_sf(operation: str):
    if sf is None:
        raise SystemExit(
            f"Для операции '{operation}' нужен пакет soundfile.\n"
            f"Установи его, например:\n"
            f"  uv pip install soundfile"
        )


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _resolve(p: str, project: Path) -> Path:
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        pp = project / pp
    return pp.resolve(strict=False)


def _is_wav_16k_mono(path: Path, info) -> bool:
    return (
        path.suffix.lower() == ".wav"
        and info.samplerate == EXPECTED_SR
        and info.channels == EXPECTED_CH
    )

def _trim_file_to_max(path: Path, max_dur: float, backup: bool) -> float:
    """
    Обрезает WAV-файл до max_dur секунд и возвращает новую фактическую длительность.
    Поддерживает только WAV 16 kHz mono.
    """
    _require_sf("trim-long")

    info = sf.info(str(path))

    if not _is_wav_16k_mono(path, info):
        raise ValueError(
            f"trim supports only WAV 16 kHz mono. "
            f"Got: suffix={path.suffix}, samplerate={info.samplerate}, channels={info.channels}"
        )

    frames = int(round(max_dur * info.samplerate))
    if frames <= 0:
        raise ValueError(f"Invalid target frames={frames} for max_dur={max_dur}")

    # Читаем только нужное количество сэмплов.
    data, sr = sf.read(str(path), dtype="float32", frames=frames)

    # Пытаемся сохранить исходный subtype, иначе PCM_16.
    subtype = info.subtype or "PCM_16"

    # ВАЖНО: временный файл должен иметь расширение .wav,
    # иначе soundfile не поймёт формат.
    tmp = path.with_name(path.stem + ".trim.wav")

    try:
        if backup:
            shutil.copy2(path, path.with_name(path.name + ".bak"))

        # Явно указываем format="WAV", чтобы не зависеть от детекции по расширению.
        sf.write(
            str(tmp),
            data,
            sr,
            format="WAV",
            subtype=subtype,
        )

        os.replace(str(tmp), str(path))

    finally:
        # Если временный файл остался после ошибки — удаляем.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    new_info = sf.info(str(path))
    return float(new_info.duration)

def main():
    ap = argparse.ArgumentParser(
        description="Validate manifest.csv and optionally trim long WAV clips to MAX_DUR."
    )

    ap.add_argument(
        "--manifest",
        default="data/manifests/manifest.csv",
        help="Путь к manifest.csv",
    )

    ap.add_argument(
        "--project",
        default=".",
        help="Корень проекта Jarvis",
    )

    ap.add_argument(
        "--deep",
        action="store_true",
        help="Читать WAV через soundfile и проверять sample_rate / channels",
    )

    ap.add_argument(
        "--trim-long",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Если фактическая длительность больше MAX_DUR, "
            "обрезать WAV до MAX_DUR и обновить duration в manifest.csv. "
            "Можно отключить через --no-trim-long."
        ),
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Не изменять файлы и manifest, только показать, что было бы обрезано",
    )

    ap.add_argument(
        "--backup",
        action="store_true",
        help="Перед заменой WAV создавать .bak",
    )

    args = ap.parse_args()

    project = Path(args.project).expanduser().resolve()
    manifest = Path(args.manifest).expanduser()
    if not manifest.is_absolute():
        manifest = project / manifest

    if not manifest.exists():
        raise SystemExit(f"Манифест не найден: {manifest}")

    if args.trim_long:
        _require_sf("trim-long")

    if args.deep:
        _require_sf("deep")

    # ------------------------------------------------------------------
    # Читаем весь manifest в память, чтобы потом можно было переписать.
    # ------------------------------------------------------------------
    with manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])

        missing = REQUIRED - set(fieldnames)
        if missing:
            raise SystemExit(
                f"В манифесте нет обязательных колонок: {sorted(missing)}. "
                f"Найдены: {fieldnames}. "
                f"Пересобери build_manifest.py."
            )

        rows = [dict(row) for row in reader]

    if not rows:
        print(f"Manifest пуст: {manifest}")
        return

    errors = []
    warnings = []

    counts = defaultdict(int)

    # client_id -> множество сплитов, для проверки утечек impostor
    imp_splits = defaultdict(set)

    # session -> множество сплитов, для проверки утечек owner
    own_splits = defaultdict(set)

    # split -> множество client_id / session, для отчёта
    imp_by_split = defaultdict(set)
    own_by_split = defaultdict(set)

    owner_client_ids = set()

    trimmed_count = 0
    would_trim_count = 0
    manifest_changed = False
    trim_messages = []

    total = len(rows)

    # ------------------------------------------------------------------
    # Основной проход по строкам.
    # ------------------------------------------------------------------
    for row in rows:
        audio_path_raw = (row.get("audio_path") or "").strip()
        speaker_type = (row.get("speaker_type") or "").strip().lower()
        client_id = (row.get("client_id") or "").strip()
        language = (row.get("language") or "").strip().lower()
        session = (row.get("session") or "").strip()
        split = (row.get("split") or "").strip()
        duration_manifest = _to_float(row.get("duration"))

        path = _resolve(audio_path_raw, project)

        counts[(speaker_type, language, split)] += 1

        # ------------------------------------------------------------------
        # Базовые проверки значений.
        # ------------------------------------------------------------------
        if speaker_type not in VALID_TYPES:
            errors.append(f"bad speaker_type '{speaker_type}': {path}")

        if language not in VALID_LANGS:
            errors.append(f"bad language '{language}': {path}")

        if split not in VALID_SPLITS:
            errors.append(f"bad split '{split}': {path}")

        if not client_id:
            errors.append(f"empty client_id: {path}")

        if duration_manifest <= 0:
            errors.append(f"non-positive manifest duration {duration_manifest}: {path}")

        # ------------------------------------------------------------------
        # Учёт групп для проверки утечек.
        # Делаем это даже если файл отсутствует, чтобы увидеть проблемы split.
        # ------------------------------------------------------------------
        if speaker_type == "impostor" and client_id:
            imp_splits[client_id].add(split)
            imp_by_split[split].add(client_id)

        elif speaker_type == "owner":
            if client_id:
                owner_client_ids.add(client_id)

            if session:
                own_splits[session].add(split)
                own_by_split[split].add(session)
            else:
                warnings.append(f"owner row without session: {path}")

        # ------------------------------------------------------------------
        # Проверка файла.
        # ------------------------------------------------------------------
        if not audio_path_raw:
            errors.append(f"empty audio_path: {row}")
            continue

        if not path.exists():
            errors.append(f"missing file: {path}")
            continue

        actual_duration = duration_manifest

        # ------------------------------------------------------------------
        # Если нужен deep-check или trim-long, читаем реальные параметры WAV.
        # ------------------------------------------------------------------
        if args.trim_long or args.deep:
            try:
                info = sf.info(str(path))
            except Exception as e:
                errors.append(f"cannot read audio info {path}: {e}")
                continue

            actual_duration = float(info.duration)

            # Проверка sample_rate / channels.
            # Для trim-long это обязательно, потому что режем только WAV 16k mono.
            if args.deep or args.trim_long:
                if info.samplerate != EXPECTED_SR:
                    errors.append(
                        f"sample_rate mismatch {path}: "
                        f"expected={EXPECTED_SR}, file={info.samplerate}"
                    )

                if info.channels != EXPECTED_CH:
                    errors.append(
                        f"channels mismatch {path}: "
                        f"expected={EXPECTED_CH}, file={info.channels}"
                    )

            # Сверяем duration из manifest и реальный duration файла.
            if duration_manifest > 0 and abs(duration_manifest - actual_duration) > 0.3:
                warnings.append(
                    f"duration mismatch {path}: "
                    f"manifest={duration_manifest:.3f}, file={actual_duration:.3f}"
                )

            # ------------------------------------------------------------------
            # Обрезка длинных записей.
            # ------------------------------------------------------------------
            if actual_duration > MAX_DUR + EPS:
                if args.trim_long:
                    if not _is_wav_16k_mono(path, info):
                        errors.append(
                            f"skip trim, file must be WAV 16 kHz mono: {path}"
                        )
                    else:
                        if args.dry_run:
                            would_trim_count += 1
                            trim_messages.append(
                                f"dry-run: would trim {path}: "
                                f"{actual_duration:.3f} -> {MAX_DUR:.3f}"
                            )
                        else:
                            try:
                                new_duration = _trim_file_to_max(
                                    path=path,
                                    max_dur=MAX_DUR,
                                    backup=args.backup,
                                )

                                row["duration"] = f"{new_duration:.3f}"
                                manifest_changed = True
                                trimmed_count += 1

                                trim_messages.append(
                                    f"trimmed {path}: "
                                    f"{actual_duration:.3f} -> {new_duration:.3f}"
                                )

                                actual_duration = new_duration

                            except Exception as e:
                                errors.append(f"trim failed {path}: {e}")
                else:
                    warnings.append(
                        f"duration {actual_duration:.2f}s > {MAX_DUR}: {path}"
                    )

        # ------------------------------------------------------------------
        # Финальная проверка диапазона длительности.
        # В dry-run режиме не ругаемся на длинные файлы, которые только "были бы" обрезаны.
        # ------------------------------------------------------------------
        if actual_duration <= 0:
            errors.append(f"non-positive actual duration {actual_duration}: {path}")
        elif not (MIN_DUR <= actual_duration <= MAX_DUR + EPS):
            suppress_because_dry_trim = (
                args.trim_long
                and args.dry_run
                and actual_duration > MAX_DUR + EPS
            )

            if not suppress_because_dry_trim:
                warnings.append(
                    f"duration {actual_duration:.2f}s "
                    f"outside [{MIN_DUR}, {MAX_DUR}]: {path}"
                )

    # ------------------------------------------------------------------
    # Проверка утечек.
    # ------------------------------------------------------------------
    imp_leaks = {k: v for k, v in imp_splits.items() if len(v) > 1}
    if imp_leaks:
        errors.append(
            f"LEAK impostor/client_id: {len(imp_leaks)} clients in multiple splits. "
            f"Example: {list(imp_leaks.items())[:3]}"
        )

    own_leaks = {k: v for k, v in own_splits.items() if len(v) > 1}
    if own_leaks:
        errors.append(
            f"LEAK owner/session: {len(own_leaks)} sessions in multiple splits. "
            f"Example: {list(own_leaks.items())[:3]}"
        )

    if len(owner_client_ids) > 1:
        warnings.append(
            f"owner has multiple client_id values: {sorted(owner_client_ids)}"
        )

    # ------------------------------------------------------------------
    # Перезапись manifest, если были изменения и это не dry-run.
    # ------------------------------------------------------------------
    if manifest_changed and not args.dry_run:
        with manifest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Manifest updated: {manifest}")
    elif args.dry_run and would_trim_count:
        print("Dry run: files and manifest were NOT changed.")

    # ------------------------------------------------------------------
    # Отчёт.
    # ------------------------------------------------------------------
    print(f"Manifest: {manifest}")
    print(f"Project:  {project}")
    print(f"Rows:     {total}")
    print(f"Mode:     {'DEEP' if args.deep else 'quick'}")
    print(f"Trim long: {'ON' if args.trim_long else 'OFF'}")
    print(f"Dry run:   {'ON' if args.dry_run else 'OFF'}")
    print(f"Backup:    {'ON' if args.backup else 'OFF'}")
    print()

    print("Counts by speaker_type / language / split:")
    for k in sorted(counts):
        speaker_type, language, split = k
        print(f"  {speaker_type:9s} {language:3s} {split:6s}: {counts[k]}")

    print("\nUnique impostor client_id by split:")
    for split in ["train", "val", "test"]:
        print(f"  {split:6s}: {len(imp_by_split.get(split, set()))}")

    print("\nOwner sessions by split:")
    for split in ["train", "val", "test"]:
        print(f"  {split:6s}: {sorted(own_by_split.get(split, set()))}")

    print()

    if trimmed_count:
        print(f"Trimmed files: {trimmed_count}")

    if would_trim_count:
        print(f"Would trim files: {would_trim_count}")

    if trim_messages:
        print("\nTrim log:")
        for msg in trim_messages[:50]:
            print(f"  {msg}")

        if len(trim_messages) > 50:
            print(f"  ... and {len(trim_messages) - 50} more")

    print()

    if warnings:
        print(f"WARNINGS: {len(warnings)}")
        for w in warnings[:50]:
            print(f"  ⚠️ {w}")

        if len(warnings) > 50:
            print(f"  ... and {len(warnings) - 50} more")

    if errors:
        print(f"\nERRORS: {len(errors)}")
        for e in errors[:100]:
            print(f"  ❌ {e}")

        if len(errors) > 100:
            print(f"  ... and {len(errors) - 100} more")

        raise SystemExit(1)

    print("✅ Validation OK.")


if __name__ == "__main__":
    main()