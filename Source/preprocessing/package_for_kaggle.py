from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAR_CANDIDATES = (
    Path(r"C:/Program Files/WinRAR/Rar.exe"),
    Path(r"C:/Program Files (x86)/WinRAR/Rar.exe"),
)


def find_rar() -> Path:
    found = shutil.which("rar") or shutil.which("Rar")
    if found:
        return Path(found)
    for candidate in RAR_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "no rar executable found; install WinRAR or use --format zip.\n"
        "(Python cannot write .rar -- the format is proprietary and every "
        "library for it can only read.)"
    )


try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def progress(iterable, desc, total=None):
    """A bar when tqdm is installed, the bare iterable when it is not."""

    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, mininterval=1.0, dynamic_ncols=True)


def report(target: Path, count: int, label: str, started: float) -> None:
    size = target.stat().st_size / 2**30
    print(
        f"  {label}: {count:,} files, {size:.2f} GB "
        f"in {time.time() - started:.0f}s -> {target.name}"
    )


def write_zip(target: Path, members: List[Tuple[Path, str]], label: str) -> None:
    started = time.time()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for source, arcname in progress(members, f"zip {label}", len(members)):
            archive.write(source, arcname)
    report(target, len(members), label, started)


def write_rar(
    target: Path, base: Path, names: List[str], prefix: str, label: str, count: int
) -> None:
    started = time.time()
    rar = str(find_rar())
    with tempfile.TemporaryDirectory(prefix=".rar-build-", dir=target.parent) as staging:
        candidate = Path(staging) / target.name
        command = [
            rar, "a", "-m0", "-r", "-ep1", f"-ap{prefix}", "-idq",
            str(candidate), *names,
        ]
        result = subprocess.run(command, cwd=base, capture_output=True, text=True)
        if result.returncode != 0 or not candidate.is_file():
            raise SystemExit(
                f"rar failed for {label} (exit {result.returncode})\n"
                f"{result.stdout}\n{result.stderr}"
            )
        subprocess.run([rar, "t", "-idq", str(candidate)], check=True)
        os.replace(candidate, target)
    report(target, count, label, started)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the build for Kaggle.")
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "Data" / "processed" / "img448"
    )
    parser.add_argument(
        "--code", type=Path, default=PROJECT_ROOT / "Source" / "training"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "kaggle_upload"
    )
    parser.add_argument(
        "--single", action="store_true", help="one archive for everything instead of per-source"
    )
    parser.add_argument(
        "--format",
        choices=("zip", "rar", "both"),
        default="rar",
        help="rar (default): use the notebook's extraction step; zip/both remain optional",
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="rebuild only dr448_code after editing Source/training; the 9 GB of "
             "image archives are left exactly as uploaded",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    images = data / "images"
    by_source, counts = {}, Counter()
    if not args.code_only:
        if not images.is_dir():
            raise SystemExit(f"no images/ under {data}; run build_dataset.py first")
        for name in ("manifest.csv", "splits.csv"):
            if not (data / name).is_file():
                raise SystemExit(f"missing {data / name}; run build_dataset.py and make_splits.py")
        for path in sorted(images.rglob("*.png")):
            by_source.setdefault(path.parent.name, []).append(path)
        counts = Counter({k: len(v) for k, v in by_source.items()})
        print(f"found {sum(counts.values()):,} images: {dict(sorted(counts.items()))}")

    code = args.code.expanduser().resolve()
    modules = sorted(code.glob("*.py"))
    if not modules:
        raise SystemExit(f"no .py files under {code}")

    tables = [(data / n, f"img448/{n}") for n in ("manifest.csv", "splits.csv")]
    formats = ("zip", "rar") if args.format == "both" else (args.format,)

    for suffix in formats:
        if args.code_only:
            pass
        elif suffix == "zip":
            print(f"\npackaging data (.{suffix})")
            if args.single:
                write_zip(
                    output / "dr448_data.zip",
                    tables
                    + [
                        (p, f"img448/images/{p.parent.name}/{p.name}")
                        for paths in by_source.values()
                        for p in paths
                    ],
                    "all",
                )
            else:
                write_zip(output / "dr448_tables.zip", tables, "tables")
                for source, paths in sorted(by_source.items()):
                    write_zip(
                        output / f"dr448_images_{source}.zip",
                        [(p, f"img448/images/{source}/{p.name}") for p in paths],
                        source,
                    )
        else:
            print(f"\npackaging data (.{suffix})")
            if args.single:
                write_rar(
                    output / "dr448_data.rar", data,
                    ["manifest.csv", "splits.csv", "images"],
                    "img448", "all", sum(counts.values()) + 2,
                )
            else:
                write_rar(
                    output / "dr448_tables.rar", data,
                    ["manifest.csv", "splits.csv"], "img448", "tables", 2,
                )
                for source, paths in sorted(by_source.items()):
                    write_rar(
                        output / f"dr448_images_{source}.rar", images,
                        [source], "img448/images", source, len(paths),
                    )

        print(f"packaging code (.{suffix})")
        if suffix == "zip":
            write_zip(
                output / "dr448_code.zip",
                [(p, f"Source/training/{p.name}") for p in modules],
                "code",
            )
        else:
            write_rar(
                output / "dr448_code.rar", code,
                [p.name for p in modules], "Source/training", "code", len(modules),
            )

    if args.code_only:
        print(
            f"\nUpload dr448_code.{formats[-1]} as a NEW VERSION of the code dataset.\n"
            "The data dataset stays as it is; import the updated notebook as well."
        )
        return 0

    produced = sorted(
        p for p in output.iterdir() if p.suffix.lower() in {".zip", ".rar"}
    )
    grand = sum(p.stat().st_size for p in produced) / 2**30
    print(f"\n{len(produced)} archives, {grand:.2f} GB total, in {output}")
    if "rar" in formats:
        print(
            "\nUse the updated Notebooks/kaggle_train_448.ipynb: it extracts raw "
            "RAR inputs into /tmp before finding tables, images and code."
        )
    upload_suffix = "rar" if "rar" in formats else "zip"
    print(
        "\nUpload as TWO Kaggle datasets:\n"
        f"  1. data  -- dr448_tables.{upload_suffix} + dr448_images_*.{upload_suffix}\n"
        f"             (or dr448_data.{upload_suffix} with --single)\n"
        f"  2. code  -- dr448_code.{upload_suffix}\n"
        "The notebook handles split archive branches and merges image roots with symlinks.\n"
        "Keeping code separate means editing a module never re-uploads 5 GB."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
