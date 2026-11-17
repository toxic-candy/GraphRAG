import argparse
from pathlib import Path

import pandas as pd


def _safe(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def build_bottom_layer(mimic_root: Path, out_dir: Path, max_files: int = 12) -> int:
    hosp = mimic_root / "hosp"
    d_icd_diag = pd.read_csv(hosp / "d_icd_diagnoses.csv")
    d_icd_proc = pd.read_csv(hosp / "d_icd_procedures.csv")
    d_lab = pd.read_csv(hosp / "d_labitems.csv")

    out_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0

    # Build compact ontology-like dictionary files.
    chunks = []

    diag_lines = []
    for _, row in d_icd_diag.head(2500).iterrows():
        code = _safe(row.get("icd_code"))
        ver = _safe(row.get("icd_version"))
        title = _safe(row.get("long_title"))
        if title:
            diag_lines.append(f"DIAGNOSIS code={code} icd_version={ver} name={title}")
    chunks.append(("diagnosis_dictionary", diag_lines))

    proc_lines = []
    for _, row in d_icd_proc.head(1800).iterrows():
        code = _safe(row.get("icd_code"))
        ver = _safe(row.get("icd_version"))
        title = _safe(row.get("long_title"))
        if title:
            proc_lines.append(f"PROCEDURE code={code} icd_version={ver} name={title}")
    chunks.append(("procedure_dictionary", proc_lines))

    lab_lines = []
    for _, row in d_lab.head(1800).iterrows():
        itemid = _safe(row.get("itemid"))
        label = _safe(row.get("label"))
        fluid = _safe(row.get("fluid"))
        category = _safe(row.get("category"))
        if label:
            lab_lines.append(
                f"LAB_TEST itemid={itemid} label={label} fluid={fluid} category={category}"
            )
    chunks.append(("lab_dictionary", lab_lines))

    for name, lines in chunks:
        if files_written >= max_files:
            break
        if not lines:
            continue
        path = out_dir / f"bottom_{files_written:02d}_{name}.txt"
        text = "medical dictionary entries:\n" + "\n".join(lines) + "\n"
        path.write_text(text, encoding="utf-8")
        files_written += 1

    return files_written


def build_middle_layer(
    txt_dir: Path,
    out_dir: Path,
    max_files: int | None = None,
) -> int:
    """
    Build the middle layer by copying pre-converted PMC paper .txt files into
    the pipeline's middle-layer staging directory.

    The .txt files must already exist — run ``convert_xml_to_txt.py`` first.

    Args:
        txt_dir:   Directory containing pre-converted .txt files
                   (default: middle-layer/pneumonia_corpus/txt/).
        out_dir:   Destination directory for middle-layer files consumed by the pipeline.
        max_files: Maximum number of files to stage. None means all available.

    Returns:
        Number of files staged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not txt_dir.exists():
        print(f"[middle layer] ERROR: txt directory not found: {txt_dir}")
        print(f"[middle layer] Run 'python convert_xml_to_txt.py' first to generate .txt files.")
        return 0

    txt_files = sorted(txt_dir.glob("*.txt"))
    if not txt_files:
        print(f"[middle layer] ERROR: No .txt files found in {txt_dir}")
        print(f"[middle layer] Run 'python convert_xml_to_txt.py' first to generate .txt files.")
        return 0

    print(f"[middle layer] Found {len(txt_files)} pre-converted .txt files in {txt_dir}")

    files_written = 0
    for src in txt_files:
        if max_files is not None and files_written >= max_files:
            break
        dst = out_dir / f"middle_{files_written:03d}_{src.name}"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        files_written += 1

    print(f"[middle layer] Staged {files_written} files → {out_dir}")
    return files_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local bottom/middle/top layer text datasets for the pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=None,
        help="Path to MIMIC dataset root (defaults to mimic-iv-3.1-pneumonia-100 or demo-2.2).",
    )
    parser.add_argument(
        "--middle-txt-dir",
        type=Path,
        default=Path("middle-layer/pneumonia_corpus/txt"),
        help=(
            "Directory containing pre-converted PMC .txt files for the middle layer. "
            "Run 'python convert_xml_to_txt.py' first to populate this directory."
        ),
    )
    parser.add_argument(
        "--max-middle",
        type=int,
        default=None,
        help="Maximum number of middle-layer files to stage into the pipeline. Default: all available.",
    )
    parser.add_argument("--top-path", type=Path, default=Path("dataset/mimic_demo_10"))
    parser.add_argument("--out-root", type=Path, default=Path("dataset/three_layer"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_root = args.out_root.resolve()
    bottom_dir = out_root / "bottom"
    middle_dir = out_root / "middle"
    top_dir = out_root / "top"

    if args.mimic_root is not None:
        mimic_root = args.mimic_root if args.mimic_root.is_absolute() else (repo_root / args.mimic_root).resolve()
    else:
        candidate = repo_root / "mimic-iv-3.1-pneumonia-100"
        mimic_root = candidate if candidate.exists() else (repo_root / "mimic-iv-clinical-database-demo-2.2")

    txt_dir = args.middle_txt_dir if args.middle_txt_dir.is_absolute() else (repo_root / args.middle_txt_dir).resolve()

    bottom_n = build_bottom_layer(mimic_root, bottom_dir)
    middle_n = build_middle_layer(txt_dir, middle_dir, max_files=args.max_middle)

    top_dir.mkdir(parents=True, exist_ok=True)
    top_n = 0
    for src in sorted(args.top_path.resolve().glob("*.txt")):
        dst = top_dir / src.name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        top_n += 1

    print(f"bottom_files={bottom_n}")
    print(f"middle_files={middle_n}")
    print(f"top_files={top_n}")
    print(f"output_root={out_root}")


if __name__ == "__main__":
    main()
