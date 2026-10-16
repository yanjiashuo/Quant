"""Create LF-normalized copies of the deploy package.

The committed artifacts got CRLF line endings on Windows checkout (core.autocrlf=true),
which breaks LightGBM's text model parser. We copy the package to WORK/deploy_lf with
all lgb_model.txt normalized to LF, without mutating the committed repo.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from config import DEPLOY_DIR, WORK

OUT = WORK / "deploy_lf"


def normalize_txt(src: Path, dst: Path) -> None:
    raw = src.read_bytes().replace(b"\r\n", b"\n")
    dst.write_bytes(raw)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    n_txt = n_other = 0
    for src in DEPLOY_DIR.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(DEPLOY_DIR)
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".txt":
            normalize_txt(src, dst)
            n_txt += 1
        else:
            shutil.copy2(src, dst)
            n_other += 1
    print(f"normalized {n_txt} .txt files, copied {n_other} others -> {OUT}")

    # sanity: load every booster
    import lightgbm as lgb
    for mdir in sorted(OUT.iterdir()):
        txt = mdir / "lgb_model.txt"
        if txt.exists():
            b = lgb.Booster(model_file=str(txt))
            print(f"  {mdir.name}: OK trees={b.num_trees()} feat={b.num_feature()}")


if __name__ == "__main__":
    main()
