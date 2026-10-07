#!/usr/bin/env python3
"""
Собирает единый data/manifests/manifest.csv под архитектуру «профили + similarity».

ВХОД:
  data/impostor/metadata.csv   (объединённый, от merge_impostor_manifests.py)
      колонки: file, speaker_type, client_id, language, duration, sentence
      файлы лежат в data/impostor/{language}/{file}

  data/own/metadata.csv        (от re_slice_own.py)
      колонки: file, speaker_type, client_id, language, duration, sentence
      файлы лежат в data/own/{language}/{file}
      session извлекается из имени файла: {lang}_{session}_{idx:03d}.wav

ВЫХОД (data/manifests/manifest.csv):
  audio_path, speaker_type, client_id, language, duration, sentence, session, split

ПРАВИЛА SPLIT (честное разделение enrollment и evaluation):
  impostor -> по client_id  (один спикер = один сплит)
  owner    -> по session    (одна сессия = один сплит)

РОЛИ СПЛИТОВ:
  owner/train = ENROLLMENT, из этих эмбеддингов строится профиль владельца
  owner/val   = подбор порога / калибровка / измерение FRR
  owner/test  = финальная честная оценка FRR на новых сессиях
  impostor/*  = измерение FAR на чужих людях
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

# Колонки ИТОГОВОГО манифеста. session и split ОБЯЗАТЕЛЬНЫ,
# иначе весь смысл честного разделения теряется.
FIELDS = [
    "audio_path",
    "speaker_type",
    "client_id",
    "language",
    "duration",
    "sentence",
    "session",
    "split",
]

VALID_SPLITS = {"train", "val", "test"}
VALID_LANGS = {"ru", "en"}
VALID_TYPES = {"owner", "impostor"}


# ----------------------------------------------------------------------
# мелкие хелперы
# ----------------------------------------------------------------------
def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _norm_type(v):
    """OWNER/Own/owner -> owner ; IMPOSTOR/impostor -> impostor."""
    v = (v or "").strip().lower()
    if v in {"owner", "own", "me", "self"}:
        return "owner"
    if v in {"impostor", "other", "not_me", "not-me"}:
        return "impostor"
    return v  # пусть sanity-проверка поймает мусор


def _rel_path(path: Path, project: Path) -> str:
    """Сохраняем путь относительно корня проекта -> манифест переносим."""
    try:
        return str(path.expanduser().resolve(strict=False).relative_to(project))
    except Exception:
        return str(path.expanduser().resolve(strict=False))


def _owner_session_from_filename(filename: str, language: str) -> str:
    """
    Имя файла owner: {lang}_{session}_{idx:03d}.wav
    Примеры:
        ru_session_2026-10-07_000.wav  -> session_2026-10-07
        en_session_en_day2_001.wav     -> session_en_day2

    Надёжнее regex: режем по '_', первый элемент = lang, последний = idx (3 цифры),
    середина join'ом = session. Это не ломается, если session сам содержит '_'.
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    # минимум: lang + session + idx  -> 3 части
    if len(parts) < 3:
        return ""
    if parts[0] != language:
        return ""
    idx = parts[-1]
    if not (idx.isdigit() and len(idx) == 3):
        return ""

    return "_".join(parts[1:-1]).strip()


# ----------------------------------------------------------------------
# impostor: один объединённый metadata.csv
# ----------------------------------------------------------------------
def load_impostor(impostor_root: Path, project: Path):
    meta = impostor_root / "metadata.csv"
    if not meta.exists():
        raise SystemExit(
            f"Не найден объединённый impostor-манифест: {meta}\n"
            f"Сначала запусти: python scripts/merge_impostor_manifests.py "
            f"--impostor-dir {impostor_root}"
        )

    rows = []
    with meta.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        required = {"file", "speaker_type", "client_id", "language", "duration"}
        if not required.issubset(cols):
            raise SystemExit(
                f"{meta}: ожидаются колонки {sorted(required)}, найдены {sorted(cols)}."
            )

        for r in reader:
            fname = (r.get("file") or "").strip()
            cid = (r.get("client_id") or "").strip()
            lang = (r.get("language") or "").strip().lower()
            stype = _norm_type(r.get("speaker_type"))
            dur = _to_float(r.get("duration"))
            sent = (r.get("sentence") or "").strip()

            if not fname or not cid or lang not in VALID_LANGS:
                continue
            if stype != "impostor":
                continue  # в merged-файле не должно быть owner, но на всякий случай

            rows.append({
                "audio_path": _rel_path(impostor_root / lang / fname, project),
                "speaker_type": "impostor",
                "client_id": cid,
                "language": lang,
                "duration": f"{dur:.3f}",
                "sentence": sent,
                "session": "",      # у impostor сессий нет
                "split": "",
            })
    return rows


# ----------------------------------------------------------------------
# owner: metadata.csv + session из имени файла
# ----------------------------------------------------------------------
def load_owner(own_root: Path, project: Path):
    meta = own_root / "metadata.csv"
    if not meta.exists():
        raise SystemExit(
            f"Не найден {meta}\n"
            f"Сначала запусти: python scripts/re_slice_own.py "
            f"--raw-dir /home/fedosdan2/Music --out-dir {own_root}"
        )

    rows = []
    with meta.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        required = {"file", "speaker_type", "client_id", "language", "duration"}
        if not required.issubset(cols):
            raise SystemExit(
                f"{meta}: ожидаются колонки {sorted(required)}, найдены {sorted(cols)}."
            )

        has_session_col = "session" in cols  # если позже добавишь колонку — используем её

        for r in reader:
            fname = (r.get("file") or "").strip()
            cid = (r.get("client_id") or "OWNER_001").strip()
            lang = (r.get("language") or "").strip().lower()
            stype = _norm_type(r.get("speaker_type"))
            dur = _to_float(r.get("duration"))
            sent = (r.get("sentence") or "").strip()

            if not fname or lang not in VALID_LANGS:
                continue
            if stype != "owner":
                continue

            session = ""
            if has_session_col:
                session = (r.get("session") or "").strip()
            if not session:
                session = _owner_session_from_filename(fname, lang)
            if not session:
                session = "UNKNOWN_SESSION"

            rows.append({
                "audio_path": _rel_path(own_root / lang / fname, project),
                "speaker_type": "owner",
                "client_id": cid,
                "language": lang,
                "duration": f"{dur:.3f}",
                "sentence": sent,
                "session": session,
                "split": "",
            })
    return rows


# ----------------------------------------------------------------------
# split по группам без пересечений
# ----------------------------------------------------------------------
def assign_split_by_group(rows, group_key, train_ratio, val_ratio, seed, label):
    """
    Раскладываем ГРУППЫ (спикер/сессия) по сплитам, а не отдельные клипы.
    Гарантируем: train>=1; при n>=2 test>=1; val>=1 если val_ratio>0 и n>=3.
    Остаток докидываем в test.
    """
    groups = defaultdict(list)
    for r in rows:
        key = (r.get(group_key) or "").strip() or "UNKNOWN_GROUP"
        groups[key].append(r)

    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)
    n = len(keys)

    if n == 0:
        print(f"  [{label}] групп нет — split не присвоен")
        return groups

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    # гарантии непустоты
    if n >= 1 and n_train < 1:
        n_train = 1
    if n >= 2 and n_test < 1:
        n_test = 1
    if val_ratio > 0 and n >= 3 and n_val < 1:
        n_val = 1

    # если перебор — урезаем val, потом test, train не трогаем (он >=1)
    total = n_train + n_val + n_test
    if total > n:
        over = total - n
        cut_val = min(over, max(0, n_val - (1 if (val_ratio > 0 and n >= 3) else 0)))
        n_val -= cut_val
        over -= cut_val
        if over > 0:
            cut_test = min(over, max(0, n_test - (1 if n >= 2 else 0)))
            n_test -= cut_test
            over -= cut_test
        if over > 0:
            n_train = max(1, n_train - over)
    elif total < n:
        n_test += (n - total)  # остаток в test

    for i, k in enumerate(keys):
        if i < n_train:
            sp = "train"
        elif i < n_train + n_val:
            sp = "val"
        else:
            sp = "test"
        for r in groups[k]:
            r["split"] = sp

    print(f"  [{label}] групп={n}: train={n_train}, val={n_val}, test={n_test}")
    return groups


# ----------------------------------------------------------------------
# проверка утечек: одна группа не может быть в двух сплитах
# ----------------------------------------------------------------------
def validate_no_leak(rows, group_key, label):
    seen = defaultdict(set)
    for r in rows:
        if r["split"] and r[group_key]:
            seen[r[group_key]].add(r["split"])
    leaks = {k: v for k, v in seen.items() if len(v) > 1}
    if leaks:
        print(f"  ❌ УТЕЧКА [{label}]: {len(leaks)} групп в нескольких сплитах. "
              f"Пример: {list(leaks.items())[:3]}")
        return False
    print(f"  ✅ [{label}] утечек нет (каждая группа в одном сплите)")
    return True


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=str, default=".")
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    project = Path(args.project).expanduser().resolve()
    own_root = project / "data" / "own"
    impostor_root = project / "data" / "impostor"
    out_manifest = project / "data" / "manifests" / "manifest.csv"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    ok = True

    # ---------------- IMPOSTOR ----------------
    print("=== IMPOSTOR ===")
    imp_rows = load_impostor(impostor_root, project)
    n_clients = len(set(r["client_id"] for r in imp_rows))
    n_ru = sum(1 for r in imp_rows if r["language"] == "ru")
    n_en = sum(1 for r in imp_rows if r["language"] == "en")
    print(f"  merged: файлов={len(imp_rows)}, client_id={n_clients}, ru={n_ru}, en={n_en}")

    if imp_rows:
        assign_split_by_group(imp_rows, "client_id",
                              args.train_ratio, args.val_ratio, args.seed,
                              "impostor/client_id")
        ok &= validate_no_leak(imp_rows, "client_id", "impostor/client_id")
    all_rows.extend(imp_rows)

    # ---------------- OWNER ----------------
    print("\n=== OWNER ===")
    own_rows = load_owner(own_root, project)
    n_sess = len(set(r["session"] for r in own_rows if r["session"]))
    n_clients_own = len(set(r["client_id"] for r in own_rows))
    n_unknown = sum(1 for r in own_rows if r["session"] == "UNKNOWN_SESSION")
    print(f"  файлов={len(own_rows)} "
          f"(ru={sum(1 for r in own_rows if r['language']=='ru')}, "
          f"en={sum(1 for r in own_rows if r['language']=='en')}), "
          f"сессий={n_sess}, client_id={n_clients_own}, unknown_session={n_unknown}")

    if n_unknown > 0:
        print("  ⚠️ Часть owner-файлов не получила session из имени. "
              "Проверь формат имён {lang}_{session}_{idx:03d}.wav "
              "или добавь колонку session в re_slice_own.py.")
    if n_sess < 2:
        print("  ⚠️ Мало сессий (<2): честный split по сессиям невозможен. "
              "Доспиши записей в другой день/на другой микрофон и перезапусти re_slice_own.py.")
    if n_clients_own > 1:
        print(f"  ⚠️ У owner несколько client_id ({n_clients_own}). "
              "Для «я/не я» обычно один владелец. Мульти-юзер — игнорируй.")

    assign_split_by_group(own_rows, "session",
                          args.train_ratio, args.val_ratio, args.seed,
                          "owner/session")
    ok &= validate_no_leak(own_rows, "session", "owner/session")

    if not [r for r in own_rows if r["split"] == "test"]:
        print("  ⚠️ У owner пустой test: FRR на новых сессиях не измерить. "
              "Доспиши сессий или временно считай FRR по val.")
    all_rows.extend(own_rows)

    # ---------------- SANITY ----------------
    print("\n=== SANITY ===")
    for r in all_rows:
        if r["speaker_type"] not in VALID_TYPES:
            print(f"  ❌ bad speaker_type '{r['speaker_type']}': {r['audio_path']}"); ok = False
        if r["language"] not in VALID_LANGS:
            print(f"  ❌ bad language '{r['language']}': {r['audio_path']}"); ok = False
        if r["split"] not in VALID_SPLITS:
            print(f"  ❌ bad split '{r['split']}': {r['audio_path']}"); ok = False
        if _to_float(r["duration"]) <= 0:
            print(f"  ❌ non-positive duration: {r['audio_path']}"); ok = False

    if not ok:
        raise SystemExit("Сборка прервана: см. ошибки выше.")

    # ---------------- ЗАПИСЬ ----------------
    all_rows.sort(key=lambda r: (
        r["speaker_type"], r["language"], r["split"], r["session"], r["client_id"], r["audio_path"]
    ))
    with out_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    # ---------------- СВОДКА ----------------
    print("\n=== СВОДКА ===")
    counts = defaultdict(int)
    for r in all_rows:
        counts[(r["speaker_type"], r["language"], r["split"])] += 1
    for k in sorted(counts):
        print(f"  {k[0]:9s} {k[1]:3s} {k[2]:6s}: {counts[k]}")

    imp_by_split = defaultdict(set)
    own_by_split = defaultdict(set)
    for r in all_rows:
        if r["speaker_type"] == "impostor":
            imp_by_split[r["split"]].add(r["client_id"])
        else:
            own_by_split[r["split"]].add(r["session"])
    print("\nUnique impostor client_id by split:")
    for sp in ["train", "val", "test"]:
        print(f"  {sp:6s}: {len(imp_by_split.get(sp, set()))}")
    print("\nOwner sessions by split:")
    for sp in ["train", "val", "test"]:
        print(f"  {sp:6s}: {sorted(own_by_split.get(sp, set()))}")

    print(f"\nМанифест: {out_manifest}")
    print(f"Всего строк: {len(all_rows)}")


if __name__ == "__main__":
    main()