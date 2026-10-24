"""Reads ./data (organiser files): triage dataset (JSONL), caregivers (CSV), README-derived taxonomy.

Organiser files seen at build time (Sanocare synthetic dataset v1):
  triage_dataset.jsonl   nested {id, detected_language, request{...}, response{...}}   (also a flat CSV copy)
  caregivers.csv         caregiver_id, name, role, qualification, gender, languages;..., skills;..., base_area, base_lat, base_lng,
                         max_radius_km, tier, rating, experience_years, free_from, free_until, active
  multi_turn_examples.jsonl, README.md
Everything format-specific lives here (field aliases), so a changed file format means editing only this module.
Label *names* come from the dataset; definitions, ordering and matching rules from config/taxonomy.yaml.
"""
from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, time
from functools import lru_cache
from pathlib import Path

import yaml

from . import config

log = logging.getLogger("disha.data")
FOLLOW_UP = "follow_up"          # internal label for "ask a follow-up question" (care_type null in the data)
NO_URGENCY = "none"
SUFFIXES = {".json", ".jsonl", ".csv", ".yaml", ".yml"}


# ---------- generic readers ----------
def read_any(path: Path):
    s = path.suffix.lower()
    if s == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if s == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if s == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if s in (".yaml", ".yml"):
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported file type: {path}")


def _rows(obj, *keys) -> list[dict]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in keys:
            if isinstance(obj.get(k), list):
                return obj[k]
    raise ValueError(f"Cannot find a row list (looked for {keys}) in top-level {type(obj).__name__}")


def _pick(row: dict, aliases: list[str], default=None):
    lower = {k.lower(): v for k, v in row.items()}
    for a in aliases:
        if a in lower and lower[a] not in (None, ""):
            return lower[a]
    return default


def _listify(v) -> list[str]:
    if v is None or v == "":
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v]
    s = str(v)
    for sep in (";", "|", ","):
        if sep in s:
            return [x.strip() for x in s.split(sep) if x.strip()]
    return [s.strip()]


def _find(d: Path, globs: list[str]) -> Path | None:
    for g in globs:
        hits = sorted(p for p in d.glob(g) if p.suffix.lower() in SUFFIXES and "splits" not in p.parts)
        if hits:
            return hits[0]
    return None


DATASET_GLOBS = ["triage_dataset.jsonl", "*triage*.jsonl", "*train*", "*vignette*", "*cases*"]
CAREGIVER_GLOBS = ["caregivers.csv", "*caregiver*", "*provider*"]


@lru_cache(maxsize=1)
def active_data_dir() -> Path:
    d = config.DATA_DIR
    if not (d.is_dir() and _find(d, DATASET_GLOBS)):
        raise FileNotFoundError(f"No organiser dataset in {d}. Put triage_dataset.jsonl and caregivers.csv there (see data/README.md).")
    return d


def using_dev_data() -> bool:      # kept for scripts; there is no synthetic fallback any more
    return False


# ---------- taxonomy ----------
@dataclass
class Taxonomy:
    care_types: list[str]
    urgency_levels: list[str]            # ordered least -> most urgent
    emergency_label: str
    nurse_label: str | None
    definitions: dict[str, str]          # label -> definition
    red_flag_examples: list[str]
    red_flag_vocab: list[str]
    follow_up_definition: str
    matching: dict
    source: str

    @property
    def urgency_rank(self) -> dict[str, int]:
        return {u: i for i, u in enumerate(self.urgency_levels)}

    @property
    def immediate_label(self) -> str:
        return self.urgency_levels[-1]

    @property
    def routine_label(self) -> str:
        return self.urgency_levels[0]


@lru_cache(maxsize=1)
def load_taxonomy() -> Taxonomy:
    d = active_data_dir()
    cfg = yaml.safe_load((config.CONFIG_DIR / "taxonomy.yaml").read_text(encoding="utf-8"))
    rows = read_any(_find(d, DATASET_GLOBS))
    care = sorted({r["response"]["care_type"] for r in rows if r.get("response", {}).get("care_type")},
                  key=lambda c: list(cfg["care_types"]).index(c) if c in cfg["care_types"] else 99)
    seen_urg = {r["response"]["urgency"] for r in rows if r.get("response", {}).get("urgency")}
    urg = [u for u in cfg["urgency_order"] if u in seen_urg] + sorted(seen_urg - set(cfg["urgency_order"]))
    emergency = next((c for c in care if "emergency" in c.lower()), None)
    if emergency is None:
        raise ValueError("cannot identify the emergency care type in the dataset")
    nurse = next((c for c in care if "nurs" in c.lower()), None)
    defs = {**{k: " ".join(str(v).split()) for k, v in cfg["care_types"].items()}, **{k: v for k, v in cfg["urgency"].items()}}
    missing = [c for c in care if c not in defs]
    if missing:
        log.warning("taxonomy.yaml has no definition for %s", missing)
    return Taxonomy(care, urg, emergency, nurse, defs, [], list(cfg["red_flag_vocab"]), " ".join(cfg["follow_up"].split()),
                    cfg["matching"], "triage_dataset.jsonl + config/taxonomy.yaml")


# ---------- caregivers ----------
@dataclass
class Caregiver:
    id: str
    name: str
    role: str
    skills: list[str]
    lat: float
    lng: float
    languages: list[str]
    rating: float | None
    review_count: int | None = None
    gender: str | None = None
    max_radius_km: float = 10.0
    active: bool = True
    free_from: time | None = None
    free_until: time | None = None
    tier: str | None = None
    qualification: str | None = None
    experience_years: int | None = None
    base_area: str | None = None
    extra: dict = field(default_factory=dict)


def _parse_time(v) -> time | None:
    if not v:
        return None
    h, m = str(v).split(":")[:2]
    return time(int(h), int(m))


def _bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y")


@lru_cache(maxsize=1)
def load_caregivers() -> tuple[list[Caregiver], datetime | None]:
    """Returns (caregivers, dataset_now). The dataset defines no 'now', so the second item is None (wall clock is used)."""
    d = active_data_dir()
    p = _find(d, CAREGIVER_GLOBS)
    if p is None:
        raise FileNotFoundError(f"No caregiver file in {d}")
    out: list[Caregiver] = []
    for r in _rows(read_any(p), "caregivers", "providers", "items"):
        cid = _pick(r, ["caregiver_id", "id"])
        role = _pick(r, ["role", "type"])
        lat, lng = _pick(r, ["base_lat", "lat", "latitude"]), _pick(r, ["base_lng", "lng", "lon", "longitude"])
        if cid is None or role is None or lat is None or lng is None:
            raise ValueError(f"{p.name}: row lacks id/role/coordinates; columns = {list(r)}")
        rc = _pick(r, ["review_count", "reviews"])
        rating = _pick(r, ["rating"])
        g = _pick(r, ["gender", "sex"])
        out.append(Caregiver(
            id=str(cid), name=str(_pick(r, ["name"], cid)), role=str(role).lower(), skills=_listify(_pick(r, ["skills"])),
            lat=float(lat), lng=float(lng), languages=[x.lower() for x in _listify(_pick(r, ["languages", "language"]))],
            rating=float(rating) if rating is not None else None, review_count=int(rc) if rc else None,
            gender={"m": "male", "f": "female"}.get(str(g).lower(), str(g).lower()) if g else None,
            max_radius_km=float(_pick(r, ["max_radius_km", "radius_km"], 10)), active=_bool(_pick(r, ["active"], True)),
            free_from=_parse_time(_pick(r, ["free_from"])), free_until=_parse_time(_pick(r, ["free_until"])),
            tier=_pick(r, ["tier"]), qualification=_pick(r, ["qualification"]),
            experience_years=int(_pick(r, ["experience_years"])) if _pick(r, ["experience_years"]) is not None else None,
            base_area=_pick(r, ["base_area"])))
    return out, None


# ---------- labelled cases + splits ----------
@dataclass
class Case:
    id: str
    text: str
    care_type: str | None           # None = follow-up needed
    urgency: str | None
    extra: dict = field(default_factory=dict)

    @property
    def care_label(self) -> str:
        return self.care_type or FOLLOW_UP

    @property
    def urgency_label(self) -> str:
        return self.urgency or NO_URGENCY

    @property
    def location(self) -> dict | None:
        return (self.extra.get("request") or {}).get("location")

    @property
    def reference_matches(self) -> list[str]:
        return [m["caregiver_id"] for m in (self.extra.get("response") or {}).get("matched_caregivers", [])]


def _case_from_row(i: int, r: dict) -> Case:
    if "request" in r:                                   # nested (organiser JSONL, and our split files)
        resp = r.get("response", {})
        return Case(str(r.get("id", i)), r["request"]["user_input"], resp.get("care_type"), resp.get("urgency"), r)
    text = _pick(r, ["user_input", "text", "input", "query"])
    if text is None:
        raise ValueError(f"row {i} has no user_input; columns = {list(r)}")
    ct, ug = _pick(r, ["care_type"]), _pick(r, ["urgency"])
    loc = {"lat": float(r["lat"]), "lng": float(r["lng"])} if r.get("lat") else None
    return Case(str(_pick(r, ["id", "case_id"], i)), str(text), ct, ug, {**r, "request": {"user_input": text, "location": loc}})


def load_cases(path: Path | None = None) -> list[Case]:
    p = path or _find(active_data_dir(), DATASET_GLOBS)
    return [_case_from_row(i, r) for i, r in enumerate(_rows(read_any(p), "cases", "data", "items"))]


def splits_dir() -> Path:
    return active_data_dir() / "splits"


def make_splits(seed: int = 42, val_frac: float = 0.2) -> tuple[list[Case], list[Case]]:
    """Stratified 80/20 split by (care_type, urgency); follow-up cases form their own stratum."""
    groups: dict[tuple, list[Case]] = {}
    for c in load_cases():
        groups.setdefault((c.care_label, c.urgency_label), []).append(c)
    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda c: c.id)
        rng.shuffle(g)
        n_val = round(len(g) * val_frac) if len(g) >= 2 else 0
        val += g[:n_val]
        train += g[n_val:]
    sd = splits_dir()
    sd.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with (sd / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for c in sorted(rows, key=lambda c: c.id):
                f.write(json.dumps(c.extra, ensure_ascii=False) + "\n")
    load_split.cache_clear()
    return train, val


def make_unseen_split(seed: int = 42, val_frac: float = 0.25) -> tuple[list[Case], list[Case]]:
    """Harder split for generalisation: whole request *templates* (distinct texts) are held out, so no val text appears in train."""
    by_class: dict[tuple, dict[str, list[Case]]] = {}
    for c in load_cases():
        by_class.setdefault((c.care_label, c.urgency_label), {}).setdefault(c.text, []).append(c)
    rng = random.Random(seed)
    train, val = [], []
    for key in sorted(by_class):
        texts = sorted(by_class[key])
        rng.shuffle(texts)
        n_val = max(1, round(len(texts) * val_frac)) if len(texts) >= 3 else 0
        for i, t in enumerate(texts):
            (val if i < n_val else train).extend(by_class[key][t])
    sd = splits_dir()
    sd.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train_unseen", train), ("val_unseen", val)):
        with (sd / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for c in sorted(rows, key=lambda c: c.id):
                f.write(json.dumps(c.extra, ensure_ascii=False) + "\n")
    load_split.cache_clear()
    return train, val


TRAIN_SPLIT = "train"     # which split retrieval and the urgency prior learn from (evaluate --unseen switches it)


@lru_cache(maxsize=6)
def load_split(name: str) -> list[Case]:
    p = splits_dir() / f"{name}.jsonl"
    if not p.exists():
        make_unseen_split() if name.endswith("_unseen") else make_splits()
    return load_cases(p)


def label_distribution(cases: list[Case]) -> dict[tuple[str, str], int]:
    dist: dict[tuple[str, str], int] = {}
    for c in cases:
        dist[(c.care_label, c.urgency_label)] = dist.get((c.care_label, c.urgency_label), 0) + 1
    return dist


def load_multi_turn() -> list[dict]:
    p = active_data_dir() / "multi_turn_examples.jsonl"
    return read_any(p) if p.exists() else []
