"""
Sample N CS papers flagged as Highly Likely AI by Pangram, download PDFs,
extract main text (excluding appendix), and re-classify the full text via Pangram.

Usage:
  python -m scripts.sample_pangram_fulltext_cs --n 25 \
    --results-json data/results/cs/pangram_detection_results.json \
    --pdf-dir data/raw/pdfs/cs \
    --output data/results/cs/pangram_fulltext_reclassification.json

Requirements:
  - PANGRAM_API_KEY set in environment (for classification)
  - Dependencies from requirements.txt (requests, pymupdf, pandas, pyyaml, tqdm)
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

import pymupdf
import yaml

from src.detection.pangram_detector import PangramDetector


logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_pangram_results(results_json: Path) -> List[Dict]:
    """Load Pangram detection results JSON for CS papers."""
    with open(results_json, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of results")
    return data


def filter_highly_likely_ai(entries: List[Dict]) -> List[Dict]:
    """Filter entries where Pangram prediction indicates AI-generated (any confidence level except 'Unlikely AI')."""
    out = []
    for e in entries:
        pred = e.get('pangram_prediction', {})
        label = pred.get('prediction')
        if isinstance(label, str):
            label_lower = label.strip().lower()
            # Include any prediction that indicates AI, excluding only "Unlikely AI"
            if label_lower != 'unlikely ai':
                out.append(e)
    return out


def sample_entries(entries: List[Dict], n: int, seed: int) -> List[Dict]:
    random.seed(seed)
    if n >= len(entries):
        logger.info(f"Requested n={n} >= available={len(entries)}; taking all.")
        return entries
    return random.sample(entries, n)


def arxiv_pdf_url(arxiv_id: str) -> str:
    """
    Construct the arXiv PDF URL for a given arXiv ID.
    Supports new-style IDs (e.g., 2001.04425) and old-style (e.g., cs/0102012).
    """
    arxiv_id = str(arxiv_id).strip()
    # Old-style IDs contain a slash like 'cs/...' and use '/pdf/<id>.pdf'
    # New-style IDs are of the form 'YYMM.NNNNN' (possibly with version suffix)
    if '/' in arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    # Remove any version suffix (e.g., v2)
    arxiv_id = arxiv_id.replace('v', 'v')  # no-op, keep version if present; arxiv supports it
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def download_pdf(arxiv_id: str, dest_dir: Path, timeout: int = 60) -> Path:
    """Download the arXiv PDF to dest_dir if not already present."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = dest_dir / f"{str(arxiv_id).replace('/', '_')}.pdf"
    if pdf_path.exists():
        return pdf_path

    url = arxiv_pdf_url(arxiv_id)
    # logger.info(f"Downloading {arxiv_id} from {url}")
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    with open(pdf_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return pdf_path


APPENDIX_KEYWORDS = [
    '\nAppendix\n', '\nAppendices\n', '\nSupplementary\n', '\nSupplementary Material\n'
]

REFERENCES_KEYWORDS = [
    'References', 'Bibliography'
]


def extract_main_text(pdf_path: Path, max_pages: int = None) -> str:
    """
    Extract text from the PDF, stopping at the first page that appears to
    start the appendix (simple keyword heuristic). Optionally cap pages.

    Heuristic:
      - Scan each page's text; if the first ~1000 characters contain any of
        APPENDIX_KEYWORDS (case-insensitive) as a standalone word, cut before
        that page.
    """
    doc = pymupdf.open(str(pdf_path))
    texts: List[str] = []
    total_pages = doc.page_count
    stop_idx = total_pages  # exclusive

    for i in range(total_pages):
        if max_pages is not None and i >= max_pages:
            break
        page = doc.load_page(i)
        t = page.get_text("text")
        texts.append(t)

    # Join and normalize whitespace lightly
    full_text = "\n".join(texts)
    return full_text


FILTER_SECTION_REFERENCES_KEYWORDS = [
    'References', 'Bibliography', 'Appendix', 'Appendices', 'Supplementary', 'Supplementary Material', 'Acknowledgements', 'Acknowledgments',
    'REFERENCES', 'BIBLIOGRAPHY', 'APPENDIX', 'APPENDICES', 'SUPPLEMENTARY', 'SUPPLEMENTARY MATERIAL', 'ACKNOWLEDGEMENTS', 'ACKNOWLEDGMENTS'
]

def remove_sections(text: str) -> str:
    """
    Remove the References/Bibliography section from the text.
    Uses a simple heuristic: find the line containing a references keyword
    (case-insensitive) and truncate the text before it.
    """
    lines = text.split('\n')
    for i, line in enumerate(lines):
        # Check if line is a references header (e.g., "References", "Bibliography")
        if any(line == k or (k in line and len(k) <= 0.8 * len(line)) for k in FILTER_SECTION_REFERENCES_KEYWORDS):
            # Found references section; truncate before this line
            return '\n'.join(lines[:i])
    # No references section found; return full text
    return text


def trim_text_words(text: str, max_words: int) -> str:
    if max_words is None or max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def classify_full_texts(entries: List[Dict], texts: Dict[str, str], config: dict) -> Dict[str, Dict]:
    """Classify provided texts via PangramDetector; returns mapping arxiv_id -> result."""
    detector = PangramDetector(config)
    results: Dict[str, Dict] = {}
    for arxiv_id, text in tqdm(texts.items(), desc="Classifying full texts"):
        res = detector.classify_text(text)
        results[str(arxiv_id)] = res
    return results


def run(n: int, results_json: Path, pdf_dir: Path, output_path: Path, config_path: Path, seed: int, max_words: int):
    setup_logging()
    logger.info("Loading config and Pangram results...")
    config = load_config(str(config_path))

    entries = load_pangram_results(results_json)
    flagged = filter_highly_likely_ai(entries)
    if not flagged:
        logger.warning("No entries found with prediction 'Highly Likely AI'.")
        return

    sampled = sample_entries(flagged, n, seed)
    logger.info(f"Sampled {len(sampled)} papers from {len(flagged)} flagged entries")

    # Download PDFs and extract main text
    texts: Dict[str, str] = {}
    failures: List[Tuple[str, str]] = []  # (arxiv_id, reason)

    for e in tqdm(sampled, desc="Downloading & extracting"):
        arxiv_id = str(e.get('arxiv_id'))
        try:
            pdf_path = download_pdf(arxiv_id, pdf_dir)
            text = extract_main_text(pdf_path)
            text = remove_sections(text)
            text = trim_text_words(text, max_words)
            if not text or len(text.split()) < 200:
                failures.append((arxiv_id, 'extracted_text_too_short'))
                continue
            texts[arxiv_id] = text
        except Exception as ex:
            failures.append((arxiv_id, f"{type(ex).__name__}: {ex}"))

    logger.info(f"Prepared {len(texts)} texts; {len(failures)} failures")

    # Classify with Pangram
    if not os.getenv('PANGRAM_API_KEY', config.get('detection', {}).get('pangram', {}).get('api_key')):
        logger.error("PANGRAM_API_KEY not set; cannot classify. Set env var and re-run.")
        pangram_results = {}
    else:
        pangram_results = classify_full_texts(sampled, texts, config)

    # Persist outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for e in sampled:
        arxiv_id = str(e.get('arxiv_id'))
        rec = {
            'arxiv_id': arxiv_id,
            'paper_type': e.get('paper_type', 'unknown'),
            'original_pangram_prediction': e.get('pangram_prediction', {}),
            'reclassified_fulltext': pangram_results.get(arxiv_id, {}),
        }
        records.append(rec)

    with open(output_path, 'w') as f:
        json.dump(records, f, indent=2)

    # Save a companion log for failures
    if failures:
        fail_path = output_path.with_name(output_path.stem + "_failures.json")
        with open(fail_path, 'w') as f:
            json.dump([{ 'arxiv_id': a, 'reason': r } for a, r in failures], f, indent=2)
        logger.info(f"Saved failures log to {fail_path}")

    logger.info(f"Saved reclassification results to {output_path}")

    # Summary: count AI-generated papers based on prediction field
    ai_count = 0
    valid_classifications = 0
    breakdown_by_type = {}  # paper_type -> {'total': int, 'ai': int}
    
    for rec in records:
        reclassified = rec.get('reclassified_fulltext', {})
        if isinstance(reclassified, dict):
            prediction = reclassified.get('prediction')
            if prediction is not None:
                valid_classifications += 1
                paper_type = rec.get('paper_type', 'unknown')
                
                # Initialize tracking for this paper type if not seen
                if paper_type not in breakdown_by_type:
                    breakdown_by_type[paper_type] = {'total': 0, 'ai': 0}
                
                breakdown_by_type[paper_type]['total'] += 1
                
                # Check if prediction indicates AI-generated (any label except "Unlikely AI")
                if isinstance(prediction, str) and prediction.strip().lower() != 'unlikely ai':
                    ai_count += 1
                    breakdown_by_type[paper_type]['ai'] += 1
    
    if valid_classifications > 0:
        percentage = (ai_count / valid_classifications) * 100
        logger.info(f"\n{'='*60}")
        logger.info(f"SUMMARY: Full-text Pangram Reclassification Results")
        logger.info(f"{'='*60}")
        logger.info(f"Total papers reclassified: {valid_classifications}")
        logger.info(f"Papers classified as AI-generated: {ai_count}")
        logger.info(f"Percentage AI-generated: {percentage:.1f}%")
        logger.info(f"\nBreakdown by paper type:")
        for paper_type in sorted(breakdown_by_type.keys()):
            type_total = breakdown_by_type[paper_type]['total']
            type_ai = breakdown_by_type[paper_type]['ai']
            type_percentage = (type_ai / type_total * 100) if type_total > 0 else 0
            logger.info(f"  {paper_type}: {type_ai}/{type_total} AI-generated ({type_percentage:.1f}%)")
        logger.info(f"{'='*60}\n")
    else:
        logger.info("No valid classifications to summarize.")


def main():
    parser = argparse.ArgumentParser(description="Sample CS papers flagged as AI, reclassify full text via Pangram")
    parser.add_argument('--n', type=int, required=True, help='Number of papers to sample')
    parser.add_argument('--results-json', type=str, default='data/results/cs/pangram_detection_results.json', help='Path to Pangram results JSON for CS')
    parser.add_argument('--pdf-dir', type=str, default='data/raw/pdfs/cs', help='Directory to store downloaded PDFs')
    parser.add_argument('--output', type=str, default='data/results/cs/pangram_fulltext_reclassification.json', help='Output JSON path')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config YAML path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--max-words', type=int, default=50000, help='Max words to keep from extracted text')

    args = parser.parse_args()

    run(
        n=args.n,
        results_json=Path(args.results_json),
        pdf_dir=Path(args.pdf_dir),
        output_path=Path(args.output),
        config_path=Path(args.config),
        seed=args.seed,
        max_words=args.max_words,
    )


if __name__ == '__main__':
    main()
