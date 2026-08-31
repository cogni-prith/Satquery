#!/usr/bin/env python3
"""Print the exact commands to fetch every dataset. Downloads nothing.

Deliberate: a script that starts a 100 GB transfer because someone ran it to see what it
did is a bad script. It prints commands, the destination each expands to, and the checksum
to verify against, and then stops. Copy the line you want.

`--verify` hashes what is already on disk against the recorded checksums; it also does not
download.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.utils.paths import data_root

#: name -> (destination under the data root, fetch command, sha256 of the archive or "").
#:
#: An empty checksum means it has not been recorded yet and `--verify` will say so rather
#: than pass. A blank that reports success is worse than no check at all.
DATASETS: dict[str, tuple[str, str, str]] = {
    "rsvqa_lr": (
        "rsvqa",
        "hf download RSVQA/RSVQA-LR --repo-type dataset --local-dir {dest}",
        "",
    ),
    "vrsbench": (
        "vrsbench",
        "HF_HUB_ENABLE_HF_TRANSFER=1 hf download xiang709/VRSBench --repo-type dataset "
        "--local-dir {dest}",
        "",
    ),
    "cdvqa": (
        "cdvqa",
        "# CDVQA question files: https://github.com/YZHJessica/CDVQA\n"
        "#   git clone https://github.com/YZHJessica/CDVQA {dest}/questions\n"
        "# Imagery is SECOND, fetched separately (see 'second' below).",
        "",
    ),
    "second": (
        "second",
        "# SECOND change-detection imagery. Request access, then unzip into {dest}.\n"
        "#   unzip -tq second.zip && unzip second.zip -d {dest}",
        "",
    ),
    "reben": (
        "reben",
        "HF_HUB_ENABLE_HF_TRANSFER=1 hf download BIFOLD-BigEarthNetMMv1-0/BigEarthNet-V2 "
        "--repo-type dataset --local-dir {dest}\n"
        "#   then: rico-hdl bigearthnet --dst {dest}/lmdb ...",
        "",
    ),
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=[], help="datasets to show; default all")
    parser.add_argument("--verify", action="store_true", help="hash local files, download none")
    args = parser.parse_args(argv)

    root = data_root()
    chosen = args.names or list(DATASETS)
    unknown = [name for name in chosen if name not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; known: {sorted(DATASETS)}")

    if args.verify:
        return _verify(root, chosen)

    print(f"# data root: {root}")
    print("# This script downloads nothing. Copy a command below and run it yourself.\n")
    for name in chosen:
        subdir, command, checksum = DATASETS[name]
        dest = root / subdir
        print(f"## {name}  ->  {dest}")
        print(command.format(dest=dest))
        print(f"# sha256: {checksum or 'not recorded yet'}")
        print(f"# present: {'yes' if dest.exists() else 'no'}\n")
    return 0


def _verify(root: Path, chosen: list[str]) -> int:
    failures = 0
    for name in chosen:
        subdir, _, checksum = DATASETS[name]
        dest = root / subdir
        if not dest.exists():
            print(f"{name}: MISSING at {dest}")
            failures += 1
        elif not checksum:
            print(f"{name}: present, but no checksum is recorded -- cannot verify")
            failures += 1
        else:
            actual = sha256(dest) if dest.is_file() else "<directory>"
            ok = actual == checksum
            print(f"{name}: {'OK' if ok else f'MISMATCH {actual}'}")
            failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
