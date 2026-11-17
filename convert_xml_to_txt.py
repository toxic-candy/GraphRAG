"""
convert_xml_to_txt.py
---------------------
Standalone one-shot converter: PMC JATS XML → plain-text .txt files.

This script is NOT part of the pipeline. Run it once (or whenever the XML
corpus is updated) to produce the .txt files that the pipeline reads.

Usage
-----
  # Defaults: reads  middle-layer/pneumonia_corpus/xml/
  #           writes middle-layer/pneumonia_corpus/txt/
  python convert_xml_to_txt.py

  # Custom paths
  python convert_xml_to_txt.py \\
      --xml-dir  middle-layer/pneumonia_corpus/xml \\
      --txt-dir  middle-layer/pneumonia_corpus/txt

  # Limit output (useful for quick tests)
  python convert_xml_to_txt.py --max-files 20
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def _xml_text(elem) -> str:
    """Recursively collect all text content from an XML element, skipping tags."""
    return " ".join(elem.itertext()).strip()


def extract_pmc_xml(xml_path: Path) -> str | None:
    """
    Parse a single PMC JATS XML file and return structured plain text.

    Extracted sections (in order):
      - Title
      - Abstract  (graphical / toc abstracts are skipped)
      - Body sections (title + direct <p> children of each <sec>)

    Returns None if the file cannot be parsed or yields fewer than 200 chars.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None

    root = tree.getroot()

    # ---- title ----
    title_elem = root.find(".//article-title")
    title = re.sub(r"\s+", " ", _xml_text(title_elem)) if title_elem is not None else ""

    # ---- abstract ----
    abstract_parts: list[str] = []
    for abs_elem in root.findall(".//abstract"):
        if abs_elem.get("abstract-type", "") in ("graphical", "toc"):
            continue
        abstract_parts.append(re.sub(r"\s+", " ", _xml_text(abs_elem)))
    abstract = " ".join(abstract_parts)

    # ---- body sections ----
    body_parts: list[str] = []
    body = root.find(".//body")
    if body is not None:
        for sec in body.iter("sec"):
            sec_title_elem = sec.find("title")
            sec_title = (
                re.sub(r"\s+", " ", _xml_text(sec_title_elem))
                if sec_title_elem is not None
                else ""
            )
            # Only direct <p> children — avoids duplicating text from nested secs
            paras = [
                re.sub(r"\s+", " ", _xml_text(child))
                for child in sec
                if child.tag == "p"
            ]
            if paras:
                body_parts.append(
                    (f"{sec_title}: " if sec_title else "") + " ".join(paras)
                )

    # ---- assemble ----
    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if body_parts:
        parts.append("Full Text:\n" + "\n\n".join(body_parts))

    full_text = "\n\n".join(parts)
    return full_text if len(full_text) >= 200 else None


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------

def convert_all(xml_dir: Path, txt_dir: Path, max_files: int | None = None, min_chars: int = 500) -> None:
    """
    Convert every .xml file in *xml_dir* to a .txt file in *txt_dir*.

    Already-converted files whose .txt counterpart exists are skipped unless
    ``--overwrite`` is passed on the CLI.

    Args:
        xml_dir:    Directory containing PMC .xml files.
        txt_dir:    Output directory for .txt files.
        max_files:  Optional cap on the total number of files written.
        min_chars:  Documents shorter than this (after extraction) are skipped.
    """
    if not xml_dir.exists():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"[convert] No .xml files found in {xml_dir}")
        return

    txt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[convert] Source : {xml_dir}  ({len(xml_files)} files)")
    print(f"[convert] Output : {txt_dir}")
    if max_files:
        print(f"[convert] Limit  : {max_files} files")
    print()

    written = skipped_short = skipped_error = skipped_exists = 0

    for xml_path in xml_files:
        if max_files is not None and written >= max_files:
            break

        pmcid = xml_path.stem          # e.g. "PMC10961219"
        txt_path = txt_dir / f"{pmcid}.txt"

        if txt_path.exists():
            skipped_exists += 1
            continue

        text = extract_pmc_xml(xml_path)

        if text is None:
            skipped_error += 1
            print(f"  [SKIP-ERR ] {xml_path.name}")
            continue

        if len(text) < min_chars:
            skipped_short += 1
            print(f"  [SKIP-SHORT] {xml_path.name}  ({len(text)} chars)")
            continue

        txt_path.write_text(
            f"source=pmc_paper pmcid={pmcid}\n\n{text}\n",
            encoding="utf-8",
        )
        written += 1
        print(f"  [OK] {pmcid}.txt  ({len(text):,} chars)")

    print()
    print(f"[convert] Done — written={written}  skipped_exists={skipped_exists}"
          f"  skipped_short={skipped_short}  skipped_error={skipped_error}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PMC JATS XML papers to plain-text .txt files (run once, not part of pipeline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=Path("middle-layer/pneumonia_corpus/xml"),
        help="Directory containing PMC .xml source files.",
    )
    parser.add_argument(
        "--txt-dir",
        type=Path,
        default=Path("middle-layer/pneumonia_corpus/txt"),
        help="Output directory for converted .txt files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to convert (useful for quick tests). Default: all.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="Skip documents with fewer than this many characters after extraction.",
    )
    args = parser.parse_args()

    convert_all(
        xml_dir=args.xml_dir.resolve(),
        txt_dir=args.txt_dir.resolve(),
        max_files=args.max_files,
        min_chars=args.min_chars,
    )


if __name__ == "__main__":
    main()
