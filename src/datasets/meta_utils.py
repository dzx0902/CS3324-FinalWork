"""Flexible meta file parsing for JSON/CSV with various key conventions."""
from __future__ import annotations
import os
import json
import csv
from typing import List, Tuple


def _coerce_float(x) -> float:
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Cannot convert value to float: {x}")


def load_meta(meta_path: str, preferred_mos_key: str = "MOS_zscore") -> List[Tuple[str, float]]:
    ext = os.path.splitext(meta_path)[1].lower()
    items: List[Tuple[str, float]] = []
    if ext == ".json":
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for it in data:
            img = it.get("path") or it.get("image") or it.get("image_name")
            if img is None:
                raise KeyError(f"Missing image path key in JSON item: {it}")
            # priority of MOS keys
            if "score" in it:
                mos = _coerce_float(it["score"])
            elif "mos" in it:
                mos = _coerce_float(it["mos"])
            elif preferred_mos_key in it:
                mos = _coerce_float(it[preferred_mos_key])
            elif "MOS" in it:
                mos = _coerce_float(it["MOS"])
            else:
                raise KeyError(f"Missing MOS/score key in JSON item: {it}")
            items.append((img, mos))
    else:
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row.get("path") or row.get("image") or row.get("image_name")
                if img is None:
                    raise KeyError(f"Missing image path column in CSV row: {row.keys()}")
                # CSV MOS priority
                if preferred_mos_key in row:
                    mos = _coerce_float(row[preferred_mos_key])
                elif "MOS" in row:
                    mos = _coerce_float(row["MOS"])
                elif "mos" in row:
                    mos = _coerce_float(row["mos"])
                elif "score" in row:
                    mos = _coerce_float(row["score"])
                else:
                    raise KeyError(f"Missing MOS column in CSV row: {row.keys()}")
                items.append((img, mos))
    return items

