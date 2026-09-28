#!/usr/bin/env python3
import argparse
import gzip
import os
import subprocess
from pathlib import Path

import pandas as pd

PHYSIONET_BASE = "https://physionet.org/files/mimiciv/3.1"
FILES = [
    "hosp/admissions.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/d_icd_diagnoses.csv.gz",
    "hosp/patients.csv.gz",
    "hosp/procedures_icd.csv.gz",
    "hosp/d_icd_procedures.csv.gz",
    "hosp/labevents.csv.gz",
    "hosp/d_labitems.csv.gz",
    "hosp/prescriptions.csv.gz",
    "hosp/microbiologyevents.csv.gz",
]

PNEUMONIA_PREFIXES_9 = ["480", "481", "482", "483", "484", "485", "486"]
PNEUMONIA_PREFIXES_10 = ["J12", "J13", "J14", "J15", "J16", "J17", "J18"]


def normalize_icd(code):
    if pd.isna(code):
        return ""
    return str(code).strip().upper().replace(".", "")


def is_pneumonia_icd(code, icd_version):
    code = normalize_icd(code)
    version = str(icd_version).strip()
    if version == "9":
        return any(code.startswith(p) for p in PNEUMONIA_PREFIXES_9)
    if version == "10":
        return any(code.startswith(p) for p in PNEUMONIA_PREFIXES_10)
    return False


def download_with_wget(url, output_path, username, password):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "--user", username,
        "--password", password,
        "--tries=3",
        "--timeout=60",
        "-O", str(output_path),
        url,
    ]
    print("Downloading:", url)
    subprocess.run(cmd, check=True)
    print("Saved:", output_path)


def gunzip_file(gz_path, csv_path):
    with gzip.open(gz_path, "rb") as f_in:
        with open(csv_path, "wb") as f_out:
            f_out.write(f_in.read())
    print(f"Extracted: {csv_path}")


def load_csv(path):
    return pd.read_csv(path, low_memory=False)


def build_pneumonia_subset(source_root: Path, output_root: Path, limit: int = 100):
    hosp_in = source_root / "hosp"
    hosp_out = output_root / "hosp"
    hosp_out.mkdir(parents=True, exist_ok=True)

    d_icd_diag = load_csv(hosp_in / "d_icd_diagnoses.csv")
    d_icd_diag["icd_version"] = d_icd_diag["icd_version"].astype(str)
    d_icd_diag["icd_code_norm"] = d_icd_diag["icd_code"].map(normalize_icd)

    pneumonia_codes = d_icd_diag[
        d_icd_diag.apply(
            lambda r: is_pneumonia_icd(r["icd_code_norm"], r["icd_version"]),
            axis=1,
        )
    ].copy()

    if pneumonia_codes.empty:
        raise RuntimeError("No pneumonia ICD codes found in d_icd_diagnoses.csv")

    diagnoses = load_csv(hosp_in / "diagnoses_icd.csv")
    diagnoses["icd_version"] = diagnoses["icd_version"].astype(str)
    diagnoses["icd_code_norm"] = diagnoses["icd_code"].map(normalize_icd)

    matched = diagnoses.merge(
        pneumonia_codes[["icd_code", "icd_version"]].drop_duplicates(),
        on=["icd_code", "icd_version"],
        how="inner",
    )

    subject_ids = sorted(matched["subject_id"].dropna().astype(int).unique())[:limit]
    print(f"Matched pneumonia subjects: {len(subject_ids)}")
    print(f"Keeping first {limit} subjects")

    for file_name in [
        "admissions.csv",
        "diagnoses_icd.csv",
        "d_icd_diagnoses.csv",
        "patients.csv",
        "procedures_icd.csv",
        "d_icd_procedures.csv",
        "labevents.csv",
        "d_labitems.csv",
        "prescriptions.csv",
        "microbiologyevents.csv",
    ]:
        src = hosp_in / file_name
        dst = hosp_out / file_name
        df = load_csv(src)

        if "subject_id" in df.columns:
            df = df[df["subject_id"].isin(subject_ids)].copy()

        if file_name == "d_icd_diagnoses.csv":
            df = df[
                df.apply(
                    lambda r: is_pneumonia_icd(r["icd_code"], r["icd_version"]),
                    axis=1,
                )
            ].copy()

        df.to_csv(dst, index=False)
        print(f"Saved {len(df)} rows -> {dst}")

    print(f"\nDone. Subset written to: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Download MIMIC-IV and keep only pneumonia patients.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--source-root", type=Path, default=Path("./mimic-iv-3.1"))
    parser.add_argument("--output-root", type=Path, default=Path("./mimic-iv-3.1-pneumonia-100"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    source_root = args.source_root
    source_root.mkdir(parents=True, exist_ok=True)
    hosp_dir = source_root / "hosp"
    hosp_dir.mkdir(parents=True, exist_ok=True)

    for rel in FILES:
        url = f"{PHYSIONET_BASE}/{rel}"
        gz_path = hosp_dir / Path(rel).name
        csv_path = hosp_dir / Path(rel).name.replace(".gz", "")

        if not csv_path.exists():
            if not gz_path.exists():
                download_with_wget(url, gz_path, args.username, args.password)
            gunzip_file(gz_path, csv_path)

    build_pneumonia_subset(source_root, args.output_root, args.limit)


if __name__ == "__main__":
    main()