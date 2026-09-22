# -*- coding: utf-8 -*-
"""Stream SEC companyfacts.zip into a long fact table for a whitelist of tags.

companyfacts.zip (about 1.4 GB) holds one JSON per CIK:

    {"cik": 320193, "entityName": "Apple Inc.",
     "facts": {"dei": {tag: {"label":..., "units": {unit: [entry, ...]}}},
               "us-gaap": {...}, "ifrs-full": {...}}}

    entry = {"start"?: "2017-07-02", "end": "2017-09-30", "val": 52579000000,
             "accn": "0000320193-18-000145", "fy": 2018, "fp": "FY",
             "form": "10-K", "filed": "2018-11-05", "frame"?: "CY2017Q3"}

Why a whitelist and not everything: the archive expands to roughly 15 GB of
JSON and this machine has 8 GB of RAM. Keeping only the concepts the model
needs (income statement, balance sheet, cash flow, shares) and only the forms
that enter the panel cuts the output to a few hundred MB of parquet.

Why every filing of a value is kept, not just the "frame" entry: the same
period is reported again in later filings (comparatives, restatements). A
point-in-time panel needs the FIRST filing of each value and its filed date;
the frame entry is whichever filing SEC chose to represent the period and can
embed a later restatement. De-duplication is xbrl_panel's job, with all
candidates on disk.

Output
    data/parquet/xbrl_raw/part-NNNN.parquet   long table, ~300 CIKs per part
    data/parquet/xbrl_entities.parquet         one row per CIK
    reports/logs/xbrl_extract.log

Run:  python -m sfv.xbrl_extract  [--limit N]  [--parts-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

from . import config as C
from .concepts import ALL_TAGS

PART_SIZE = 300          # CIKs per parquet part
UNITS_KEEP = {"USD", "shares", "USD/shares", "pure"}
COLUMNS = ["cik", "taxonomy", "tag", "unit", "start", "end", "val",
           "accn", "fy", "fp", "form", "filed", "frame", "n_filings", "which"]


TAGS: frozenset[str] = ALL_TAGS      # may be overridden from the command line


def _rows_for(cik: int, facts: dict) -> list[tuple]:
    """Flatten whitelisted facts of one company into tuples (COLUMNS order).

    A value for one (tag, unit, start, end) is typically reported again in
    every later filing that shows the period as a comparative. Only two of
    those reports carry information: the FIRST (point-in-time availability
    and the originally reported number) and the LAST (the restated number).
    Both are kept, labelled by `which`, with `n_filings` counting the total.
    A period reported once gets which == "only".
    """
    groups: dict[tuple, list] = {}
    for taxonomy in ("us-gaap", "dei"):
        block = facts.get(taxonomy)
        if not block:
            continue
        for tag, body in block.items():
            if tag not in TAGS:
                continue
            units = body.get("units") or {}
            for unit, entries in units.items():
                if unit not in UNITS_KEEP:
                    continue
                for e in entries:
                    form = e.get("form")
                    if form not in C.PANEL_FORMS:
                        continue
                    val = e.get("val")
                    filed = e.get("filed")
                    if val is None or not filed:
                        continue
                    key = (taxonomy, tag, unit, e.get("start"), e.get("end"))
                    groups.setdefault(key, []).append(
                        (filed, e.get("accn") or "", float(val), e.get("fy"),
                         e.get("fp"), form, e.get("frame")))
    out = []
    for (taxonomy, tag, unit, start, end), lst in groups.items():
        lst.sort(key=lambda t: (t[0], t[1]))      # by filed date, then accession
        n = len(lst)
        first, last = lst[0], lst[-1]
        if n == 1:
            picks = [(first, "only")]
        else:
            picks = [(first, "first"), (last, "last")]
        for (filed, accn, val, fy, fp, form, frame), which in picks:
            out.append((cik, taxonomy, tag, unit, start, end, val,
                        accn, fy, fp, form, filed, frame, n, which))
    return out


def _flush(rows: list[tuple], part_idx: int, parts_dir: Path) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["cik"] = df["cik"].astype("int64")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int32")
    for c in ("taxonomy", "tag", "unit", "fp", "form", "frame", "accn", "which"):
        df[c] = df[c].astype("string")
    df["n_filings"] = df["n_filings"].astype("int16")
    df.to_parquet(parts_dir / f"part-{part_idx:04d}.parquet", index=False)
    return len(df)


def _part_no(p: Path) -> int:
    return int(p.stem.split("-")[-1])


def renumber(parts_dir: Path) -> int:
    """Rename part files into one contiguous sequence, in the order the filers were read.

    Batches number their parts from the batch offset, so a four-batch run leaves
    part-0000..part-0016, part-5000.., part-10000.., part-15000... Those sort
    lexicographically as 0, 10000, 15000, 5000 - a different read order from a
    single pass, and the panel's first-reported tie-breaking depends on that
    order. Renumbering restores it.
    """
    parts = sorted(parts_dir.glob("part-*.parquet"), key=_part_no)
    tmp = [parts_dir / f"tmp-{i:05d}.parquet" for i in range(len(parts))]
    for src, dst in zip(parts, tmp):
        src.rename(dst)
    for i, f in enumerate(tmp):
        f.rename(parts_dir / f"part-{i:05d}.parquet")
    return len(parts)


def finish(parts_dir: Path, final_dir: Path, names: list, say, tags: str) -> None:
    """Write the manifest, merge the per-batch entity tables and swap the directory in."""
    prog = json.loads((parts_dir / "_progress.json").read_text(encoding="utf-8")) if (parts_dir / "_progress.json").exists() else {}
    if prog.get("done_to", 0) < prog.get("members", 1):
        raise SystemExit(f"batches cover only {prog.get('done_to')} of {prog.get('members')} members; "
                         f"run --start {prog.get('done_to')} before --finish")
    ent = sorted(parts_dir.glob("entities-*.parquet"), key=_part_no)
    if ent and not tags:
        pd.concat([pd.read_parquet(f) for f in ent], ignore_index=True).to_parquet(C.PQ / "xbrl_entities.parquet", index=False)
    for f in ent:
        f.unlink()
    renumber(parts_dir)
    n_parts = len(list(parts_dir.glob("part-*.parquet")))
    rows = sum(pd.read_parquet(f, columns=["cik"]).shape[0] for f in parts_dir.glob("part-*.parquet"))
    (parts_dir / "_manifest.json").write_text(json.dumps(
        {"written_at": pd.Timestamp.now().isoformat(timespec="seconds"), "parts": n_parts, "rows": int(rows),
         "members": prog.get("members"), "tags": tags or "all", "complete": True}, indent=1), encoding="utf-8")
    (parts_dir / "_progress.json").unlink(missing_ok=True)
    old_dir = final_dir.with_name(final_dir.name + ".old")
    if old_dir.exists():
        shutil.rmtree(old_dir)
    if final_dir.exists():
        final_dir.rename(old_dir)
    parts_dir.rename(final_dir)
    if old_dir.exists():
        shutil.rmtree(old_dir)
    say(f"complete: {n_parts} parts, {rows:,} rows -> {final_dir.name} (swapped in atomically)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(C.COMPANYFACTS_ZIP))
    ap.add_argument("--parts-dir", default=str(C.PQ / "xbrl_raw"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N CIKs (smoke test, or one batch with --start)")
    ap.add_argument("--start", type=int, default=0,
                    help="skip the first N companyfacts members and append to the staging directory. A full pass "
                         "over 20,000 filers takes about 25 minutes, longer than this machine will keep a job "
                         "alive, so it is run as consecutive batches: --start 0 --limit 5000, --start 5000 ..., "
                         "then --finish to write the manifest and swap the directory in")
    ap.add_argument("--finish", action="store_true",
                    help="the batches are done: write the manifest and swap the staging directory into place")
    ap.add_argument("--tags", default="", help="comma-separated tag list overriding the concept whitelist")
    ap.add_argument("--cik-file", default="", help="parquet/csv with a cik column: extract only these companies")
    a = ap.parse_args(argv)

    global TAGS
    if a.tags:
        TAGS = frozenset(t.strip() for t in a.tags.split(",") if t.strip())
    keep_ciks = None
    if a.cik_file:
        f = Path(a.cik_file)
        kc = pd.read_parquet(f, columns=["cik"]) if f.suffix == ".parquet" else pd.read_csv(f, usecols=["cik"])
        keep_ciks = set(kc["cik"].dropna().astype(int).unique())

    # Write into a staging directory and swap it in only when the pass finishes.
    # Deleting the old parts first and streaming new ones in their place looked
    # fine until a run was interrupted: the directory was left with a truncated
    # extract that every downstream staleness check happily accepted, and the
    # whole valuation chain was rebuilt on a fraction of the filings.
    final_dir = Path(a.parts_dir)
    parts_dir = final_dir.with_name(final_dir.name + ".building")
    if a.start == 0 and not a.finish:
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        parts_dir.mkdir(parents=True)
    elif not parts_dir.exists():
        raise SystemExit(f"{parts_dir} does not exist; start the run with --start 0 first")
    log = open(C.LOGS / "xbrl_extract.log", "a" if (a.start or a.finish) else "w", encoding="utf-8")

    def say(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    z = zipfile.ZipFile(a.zip)
    names = [n for n in z.namelist() if n.lower().endswith(".json")]
    if keep_ciks is not None:
        names = [n for n in names if n[3:13].isdigit() and int(n[3:13]) in keep_ciks]
    n_members = len(names)
    say(f"companyfacts members: {n_members:,}   whitelist tags: {len(TAGS)}"
        + (f"   batch: skip {a.start:,}" + (f", take {a.limit:,}" if a.limit else ", take the rest") if a.start or a.limit else ""))
    if a.finish:
        finish(parts_dir, final_dir, names, say, a.tags)
        log.close()
        return 0
    names = names[a.start:]

    rows: list[tuple] = []
    entities = []
    # part numbers carry the batch offset so consecutive batches never collide
    part_idx = a.start
    n_rows = n_ok = n_bad = n_in_part = 0
    t0 = time.time()
    for i, name in enumerate(names, 1):
        if a.limit and i > a.limit:
            break
        try:
            raw = z.read(name)
            j = json.loads(raw)
        except Exception as ex:  # corrupt member; count and continue
            n_bad += 1
            say(f"  bad member {name}: {type(ex).__name__}")
            continue
        cik = j.get("cik")
        facts = j.get("facts") or {}
        if cik is None:
            n_bad += 1
            continue
        r = _rows_for(int(cik), facts)
        rows.extend(r)
        gaap = facts.get("us-gaap") or {}
        entities.append({
            "cik": int(cik),
            "entity_name": j.get("entityName"),
            "n_tags_usgaap": len(gaap),
            "n_tags_ifrs": len(facts.get("ifrs-full") or {}),
            "n_rows_kept": len(r),
            "has_dei_shares": "EntityCommonStockSharesOutstanding" in (facts.get("dei") or {}),
        })
        n_ok += 1
        n_in_part += 1
        if n_in_part >= PART_SIZE:
            n_rows += _flush(rows, part_idx, parts_dir)
            rows.clear()
            part_idx += 1
            n_in_part = 0
        if i % 2000 == 0:
            el = time.time() - t0
            say(f"  {i:,}/{len(names):,}  {el/60:.1f} min  rows {n_rows + len(rows):,}  "
                f"parts {part_idx}  bad {n_bad}")
    n_rows += _flush(rows, part_idx, parts_dir)
    rows.clear()

    E = pd.DataFrame(entities)
    if len(E):
        E.to_parquet(parts_dir / f"entities-{a.start:06d}.parquet", index=False)
    done_to = a.start + (a.limit if a.limit else len(names))
    (parts_dir / "_progress.json").write_text(json.dumps(
        {"done_to": int(min(done_to, n_members)), "members": int(n_members), "tags": a.tags or "all"}, indent=1), encoding="utf-8")
    say(f"batch done: ciks {n_ok:,}  bad {n_bad}  rows {n_rows:,}  covered members {a.start:,}..{min(done_to, n_members):,} "
        f"of {n_members:,}  {(time.time() - t0)/60:.1f} min")
    if min(done_to, n_members) >= n_members:
        finish(parts_dir, final_dir, names, say, a.tags)
    say(f"entities with any us-gaap tag: {(E.n_tags_usgaap > 0).sum():,}   "
        f"with rows kept: {(E.n_rows_kept > 0).sum():,}")
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
