from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import OcrMacOptions, PdfPipelineOptions, TesseractOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


_UMLAUT_MAP = {
    "C196": "Ä",
    "C214": "Ö",
    "C218": "Ö",
    "C220": "Ü",
    "C228": "ä",
    "C246": "ö",
    "C252": "ü",
    "C223": "ß",
    "C216": "Ä",
    "C229": "ä",
    "C230": "ö",
    "C231": "ü",
    "C219": "Ü",
}

_OCR_FIXES = {
    "Eintri swahrscheinlichkeit": "Eintrittswahrscheinlichkeit",
    "mi el": "mittel",
    "Bes Practices": "Best Practices",
    "Interne of Things": "Internet of Things",
}


def _fix_german_umlauts(text: str) -> str:
    def repl(match: re.Match) -> str:
        code = match.group(1)
        return _UMLAUT_MAP.get(code, match.group(0))

    text = re.sub(r"/(C\d{3})", repl, text)
    text = re.sub(r"(?<![A-Za-z0-9])C(\d{3})", repl, text)
    for wrong, correct in _OCR_FIXES.items():
        text = text.replace(wrong, correct)
    text = re.sub(r"[ \\t]{2,}", " ", text)
    return text


def _normalize_json_strings(node: object) -> object:
    if isinstance(node, str):
        return _fix_german_umlauts(node)
    if isinstance(node, list):
        return [_normalize_json_strings(item) for item in node]
    if isinstance(node, dict):
        return {key: _normalize_json_strings(value) for key, value in node.items()}
    return node


def _build_ocr_options(engine: str, lang: list[str], force_full_page_ocr: bool):
    if engine == "mac":
        return OcrMacOptions(lang=lang, force_full_page_ocr=force_full_page_ocr)
    return TesseractOcrOptions(lang=lang, force_full_page_ocr=force_full_page_ocr)


def _canonical_page_map_for_standard(repo_root: Path, doc_stem: str) -> dict[int, str]:
    standards_dir = repo_root / "data" / "data_preprocessed" / "standards" / doc_stem
    if not standards_dir.is_dir():
        return {}

    per_page: dict[int, list[str]] = {}
    for chunk_path in sorted(standards_dir.glob("*.json")):
        if chunk_path.name == "_index.json":
            continue
        try:
            chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(chunk, dict):
            continue
        content = chunk.get("content")
        pages = chunk.get("pages")
        if not isinstance(content, str) or not content.strip() or not isinstance(pages, dict):
            continue
        start = pages.get("start")
        end = pages.get("end")
        if not isinstance(start, int):
            continue
        if not isinstance(end, int):
            end = start
        cleaned = _fix_german_umlauts(" ".join(content.split()))
        for page_no in range(start, end + 1):
            per_page.setdefault(page_no, []).append(cleaned)

    merged: dict[int, str] = {}
    for page_no, items in per_page.items():
        # keep order, drop exact duplicates
        seen: set[str] = set()
        unique = []
        for item in items:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        merged[page_no] = "\n\n".join(unique)
    return merged


def _extract_page_no_from_text_item(item: dict[str, Any]) -> int | None:
    prov = item.get("prov")
    if isinstance(prov, list):
        nums = [p.get("page_no") for p in prov if isinstance(p, dict) and isinstance(p.get("page_no"), int)]
        if nums:
            return min(nums)
    if isinstance(prov, dict) and isinstance(prov.get("page_no"), int):
        return prov["page_no"]
    return None


def _inject_canonical_text(doc_json_path: Path, repo_root: Path) -> bool:
    doc_stem = doc_json_path.stem
    page_map = _canonical_page_map_for_standard(repo_root, doc_stem)
    if not page_map:
        return False

    data = json.loads(doc_json_path.read_text(encoding="utf-8"))
    texts = data.get("texts")
    if not isinstance(texts, list):
        return False

    changed = False
    for item in texts:
        if not isinstance(item, dict):
            continue
        page_no = _extract_page_no_from_text_item(item)
        if page_no is None:
            continue
        canonical = page_map.get(page_no)
        if not canonical:
            continue
        item["canonical_text"] = canonical
        changed = True

    if changed:
        doc_json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def _postprocess_json(json_path: Path, args: argparse.Namespace, repo_root: Path) -> None:
    if args.pretty_json or args.normalize_text:
        try:
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
            if args.normalize_text:
                parsed = _normalize_json_strings(parsed)
            json_path.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    if args.inject_canonical:
        try:
            if _inject_canonical_text(json_path, repo_root):
                print(f"Injected canonical_text into {json_path.name}")
        except Exception:
            pass


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Convert PDFs via Docling to JSON/Markdown.")
    parser.add_argument(
        "--pdf-dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "data_raw"),
        help="Directory containing PDFs.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[2] / "data" / "data_docling_md"),
        help="Output directory for converted files.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "md", "both"],
        default="json",
        help="Output format. Use json to preserve provenance/page metadata.",
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="Rewrite JSON with UTF-8 characters (no unicode escapes) for readability.",
    )
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="Normalize common broken umlaut encodings like /C231 in exported JSON/Markdown.",
    )
    parser.add_argument(
        "--inject-canonical",
        action="store_true",
        help="After JSON export, inject canonical page text for standards from data/data_preprocessed/standards.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip conversion for files that already exist and only run selected postprocessing.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda", "auto"],
        default="cpu",
        help="Docling inference device.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR in Docling conversion.",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["tesseract", "mac"],
        default="tesseract",
        help="OCR backend. Use 'mac' for native macOS Vision OCR.",
    )
    parser.add_argument(
        "--ocr-lang",
        nargs="+",
        default=["eng", "deu"],
        help="OCR languages for Tesseract, e.g. --ocr-lang eng deu.",
    )
    parser.add_argument(
        "--force-full-page-ocr",
        action="store_true",
        help="Force OCR on full page instead of auto text-region detection.",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Force predictable memory behavior on macOS by defaulting to CPU.
    pdf_opts = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=args.device, num_threads=4),
        do_ocr=args.ocr,
        ocr_batch_size=1,
        layout_batch_size=1,
        table_batch_size=1,
        ocr_options=_build_ocr_options(args.ocr_engine, args.ocr_lang, args.force_full_page_ocr),
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        }
    )
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}")
        return

    for pdf in pdfs:
        json_path = out_dir / (pdf.stem + ".json")
        md_path = out_dir / (pdf.stem + ".md")

        if args.skip_existing:
            if args.format in ("json", "both") and json_path.exists():
                _postprocess_json(json_path, args, repo_root)
                print(f"Skipped conversion, updated {json_path}")
                if args.format == "json":
                    continue
            if args.format in ("md", "both") and md_path.exists() and args.format != "json":
                if args.normalize_text:
                    try:
                        current_md = md_path.read_text(encoding="utf-8")
                        md_path.write_text(_fix_german_umlauts(current_md), encoding="utf-8")
                    except Exception:
                        pass
                if args.format == "md" or (args.format == "both" and json_path.exists()):
                    print(f"Skipped conversion, kept {md_path}")
                    if args.format == "md" or (args.format == "both" and json_path.exists()):
                        continue

        result = converter.convert(str(pdf))
        document = getattr(result, "document", None)
        if document is None:
            print(f"Skipping {pdf.name}: no document output")
            continue

        if args.format in ("json", "both"):
            wrote_json = False
            if hasattr(document, "save_as_json"):
                document.save_as_json(str(json_path))
                wrote_json = True
            elif hasattr(document, "export_to_dict"):
                json_path.write_text(json.dumps(document.export_to_dict(), ensure_ascii=False), encoding="utf-8")
                wrote_json = True
            elif hasattr(document, "export_to_json"):
                raw_json = document.export_to_json()
                if isinstance(raw_json, str):
                    json_path.write_text(raw_json, encoding="utf-8")
                else:
                    json_path.write_text(json.dumps(raw_json, ensure_ascii=False), encoding="utf-8")
                wrote_json = True
            if wrote_json:
                _postprocess_json(json_path, args, repo_root)
                print(f"Wrote {json_path}")
            else:
                print(f"Skipping JSON for {pdf.name}: no json export")

        if args.format in ("md", "both"):
            if hasattr(document, "export_to_markdown"):
                markdown = document.export_to_markdown()
            elif hasattr(document, "export_to_text"):
                markdown = document.export_to_text()
            else:
                print(f"Skipping Markdown for {pdf.name}: no markdown/text export")
                continue
            markdown = _fix_german_umlauts(markdown)
            md_path.write_text(markdown, encoding="utf-8")
            print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
