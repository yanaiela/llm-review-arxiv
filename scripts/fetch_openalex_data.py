#!/usr/bin/env python3
"""
Phase 1: Fetch OpenAlex data for arxiv papers.

This script:
1. Loads paper metadata from all CS categories
2. Constructs arxiv DOIs and batch queries OpenAlex API
3. Extracts work-level data (topics, keywords, authorships)
4. Saves enriched data for downstream equity analysis
"""

import json
import os
import time
import argparse
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.parse
import urllib.error

# Configuration
OPENALEX_BASE_URL = "https://api.openalex.org"
BATCH_SIZE = 50  # Max DOIs per request
RATE_LIMIT_DELAY = 0.1  # 10 req/sec with polite pool
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N batches

# Categories to process
CATEGORIES = [
    "cs-AI", "cs-CL", "cs-CR", "cs-CV", "cs-CY",
    "cs-HC", "cs-IR", "cs-LG", "cs-RO", "cs-SE"
]


def load_paper_metadata(data_dir: Path) -> list[dict]:
    """Load paper metadata from all category JSON files."""
    all_papers = []
    
    for category in CATEGORIES:
        metadata_file = data_dir / "processed" / category / "paper_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                papers = json.load(f)
                # Add category to each paper
                for paper in papers:
                    paper["category"] = category
                all_papers.extend(papers)
                print(f"  Loaded {len(papers):,} papers from {category}")
        else:
            print(f"  Warning: {metadata_file} not found")
    
    return all_papers


def load_classification_results(data_dir: Path) -> dict[str, str]:
    """Load paper_type classifications from results files."""
    classifications = {}
    
    for category in CATEGORIES:
        results_file = data_dir / "results" / category / "pangram_detection_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
                for paper in results:
                    arxiv_id = str(paper.get("arxiv_id", ""))
                    paper_type = paper.get("paper_type", "unknown")
                    classifications[arxiv_id] = paper_type
    
    return classifications


def construct_arxiv_doi(arxiv_id) -> str:
    """Construct DOI URL from arxiv ID."""
    # Handle both numeric and string arxiv IDs
    arxiv_str = str(arxiv_id)
    # Some arxiv IDs might have version suffix like "2301.12969v1" - remove it
    if "v" in arxiv_str:
        arxiv_str = arxiv_str.split("v")[0]
    return f"https://doi.org/10.48550/arxiv.{arxiv_str}"


def batch_query_openalex(dois: list[str], email: str) -> dict:
    """Query OpenAlex for a batch of DOIs."""
    # Join DOIs with pipe separator
    doi_filter = "|".join(dois)
    
    # Build URL with select to limit response size
    params = {
        "filter": f"doi:{doi_filter}",
        "select": "id,doi,title,topics,keywords,primary_topic,cited_by_count,authorships,publication_year",
        "per-page": str(BATCH_SIZE),
        "mailto": email
    }
    
    url = f"{OPENALEX_BASE_URL}/works?{urllib.parse.urlencode(params)}"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"    HTTP Error {e.code}: {e.reason}")
        return {"results": []}
    except urllib.error.URLError as e:
        print(f"    URL Error: {e.reason}")
        return {"results": []}
    except Exception as e:
        print(f"    Error: {e}")
        return {"results": []}


def extract_work_data(work: dict) -> dict:
    """Extract relevant fields from OpenAlex work object."""
    # Extract topics with hierarchy
    topics = []
    for topic in work.get("topics") or []:
        if not topic:
            continue
        topic_id_raw = topic.get("id") or ""
        topics.append({
            "id": topic_id_raw.split("/")[-1] if topic_id_raw else "",
            "name": topic.get("display_name") or "",
            "score": topic.get("score") or 0,
            "subfield": (topic.get("subfield") or {}).get("display_name") or "",
            "field": (topic.get("field") or {}).get("display_name") or "",
            "domain": (topic.get("domain") or {}).get("display_name") or ""
        })
    
    # Extract keywords
    keywords = []
    for kw in work.get("keywords") or []:
        if not kw:
            continue
        keywords.append({
            "id": kw.get("id") or "",
            "name": kw.get("display_name") or "",
            "score": kw.get("score") or 0
        })
    
    # Extract primary topic
    primary_topic = None
    pt = work.get("primary_topic")
    if pt:
        pt_id_raw = pt.get("id") or ""
        primary_topic = {
            "id": pt_id_raw.split("/")[-1] if pt_id_raw else "",
            "name": pt.get("display_name") or "",
            "subfield": (pt.get("subfield") or {}).get("display_name") or "",
            "field": (pt.get("field") or {}).get("display_name") or ""
        }
    
    # Extract authorships
    authorships = []
    for auth in work.get("authorships", []):
        author = auth.get("author") or {}
        institutions = auth.get("institutions") or []
        
        # Handle null/missing author IDs
        author_id_raw = author.get("id") or ""
        author_id = author_id_raw.split("/")[-1] if author_id_raw else ""
        
        # Handle null/missing institution IDs
        inst_ids = []
        inst_names = []
        for inst in institutions:
            if inst:
                inst_id_raw = inst.get("id") or ""
                inst_ids.append(inst_id_raw.split("/")[-1] if inst_id_raw else "")
                inst_names.append(inst.get("display_name") or "")
        
        authorship = {
            "author_id": author_id,
            "author_name": author.get("display_name") or "",
            "orcid": author.get("orcid"),
            "position": auth.get("author_position") or "",
            "is_corresponding": auth.get("is_corresponding", False),
            "institution_ids": inst_ids,
            "institution_names": inst_names,
            "countries": auth.get("countries") or []
        }
        authorships.append(authorship)
    
    return {
        "openalex_id": work.get("id", "").split("/")[-1],
        "doi": work.get("doi", ""),
        "title": work.get("title", ""),
        "publication_year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count", 0),
        "topics": topics,
        "keywords": keywords,
        "primary_topic": primary_topic,
        "authorships": authorships
    }


def save_checkpoint(data: dict, output_dir: Path, checkpoint_num: int):
    """Save intermediate checkpoint."""
    checkpoint_file = output_dir / f"checkpoint_{checkpoint_num}.json"
    with open(checkpoint_file, "w") as f:
        json.dump(data, f)
    print(f"  Saved checkpoint {checkpoint_num}")


def load_checkpoint(output_dir: Path) -> tuple[dict, set]:
    """Load latest checkpoint if exists."""
    checkpoint_files = sorted(output_dir.glob("checkpoint_*.json"))
    if not checkpoint_files:
        return {}, set()
    
    latest = checkpoint_files[-1]
    print(f"  Loading checkpoint: {latest}")
    with open(latest) as f:
        data = json.load(f)
    
    processed_ids = set(data.get("processed_arxiv_ids", []))
    return data, processed_ids


def main():
    parser = argparse.ArgumentParser(description="Fetch OpenAlex data for arxiv papers")
    parser.add_argument("--email", required=True, help="Email for OpenAlex polite pool (10 req/sec)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of papers (0 = no limit)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = data_dir / "openalex"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Phase 1: Fetching OpenAlex Work Data")
    print("=" * 50)
    
    # Load paper metadata
    print("\n1. Loading paper metadata...")
    papers = load_paper_metadata(data_dir)
    print(f"   Total papers: {len(papers):,}")
    
    # Load classifications
    print("\n2. Loading paper type classifications...")
    classifications = load_classification_results(data_dir)
    print(f"   Classifications loaded: {len(classifications):,}")
    
    # Load checkpoint if resuming
    results = {"works": {}, "processed_arxiv_ids": []}
    processed_ids = set()
    if args.resume:
        print("\n3. Checking for checkpoint...")
        results, processed_ids = load_checkpoint(output_dir)
        print(f"   Previously processed: {len(processed_ids):,}")
    
    # Filter papers not yet processed
    papers_to_process = [p for p in papers if str(p.get("arxiv_id", "")) not in processed_ids]
    
    if args.limit > 0:
        papers_to_process = papers_to_process[:args.limit]
    
    print(f"\n4. Papers to process: {len(papers_to_process):,}")
    
    # Process in batches
    total_batches = (len(papers_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"   Batches: {total_batches:,}")
    print(f"   Estimated time: {total_batches * RATE_LIMIT_DELAY / 60:.1f} minutes")
    
    found_count = 0
    not_found_count = 0
    checkpoint_num = len(list(output_dir.glob("checkpoint_*.json")))
    
    print("\n5. Fetching from OpenAlex API...")
    for batch_idx in range(0, len(papers_to_process), BATCH_SIZE):
        batch = papers_to_process[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        
        # Construct DOIs for batch
        dois = [construct_arxiv_doi(p.get("arxiv_id")) for p in batch]
        
        # Query OpenAlex
        response = batch_query_openalex(dois, args.email)
        
        # Index results by DOI for matching
        doi_to_work = {}
        for work in response.get("results", []):
            if work.get("doi"):
                doi_to_work[work["doi"].lower()] = work
        
        # Match and extract data
        for paper in batch:
            arxiv_id = str(paper.get("arxiv_id", ""))
            doi = construct_arxiv_doi(arxiv_id).lower()
            
            if doi in doi_to_work:
                work_data = extract_work_data(doi_to_work[doi])
                work_data["arxiv_id"] = arxiv_id
                work_data["category"] = paper.get("category", "")
                work_data["paper_type"] = classifications.get(arxiv_id, "unknown")
                results["works"][arxiv_id] = work_data
                found_count += 1
            else:
                not_found_count += 1
            
            results["processed_arxiv_ids"].append(arxiv_id)
        
        # Progress update
        if batch_num % 50 == 0 or batch_num == total_batches:
            print(f"   Batch {batch_num}/{total_batches}: found={found_count:,}, not_found={not_found_count:,}")
        
        # Checkpoint
        if batch_num % CHECKPOINT_INTERVAL == 0:
            checkpoint_num += 1
            save_checkpoint(results, output_dir, checkpoint_num)
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Save final results
    print("\n6. Saving results...")
    
    # Save works
    works_file = output_dir / "works_enriched.json"
    with open(works_file, "w") as f:
        json.dump(results["works"], f, indent=2)
    print(f"   Saved {len(results['works']):,} works to {works_file}")
    
    # Extract unique authors and institutions for Phase 2 & 3
    unique_authors = {}
    unique_institutions = {}
    
    for arxiv_id, work in results["works"].items():
        for auth in work.get("authorships", []):
            author_id = auth.get("author_id", "")
            if author_id and author_id not in unique_authors:
                unique_authors[author_id] = {
                    "author_id": author_id,
                    "author_name": auth.get("author_name", ""),
                    "orcid": auth.get("orcid")
                }
            
            for inst_id, inst_name in zip(auth.get("institution_ids", []), auth.get("institution_names", [])):
                if inst_id and inst_id not in unique_institutions:
                    unique_institutions[inst_id] = {
                        "institution_id": inst_id,
                        "institution_name": inst_name
                    }
    
    # Save author list for Phase 2
    authors_file = output_dir / "authors_to_fetch.json"
    with open(authors_file, "w") as f:
        json.dump(list(unique_authors.values()), f, indent=2)
    print(f"   Saved {len(unique_authors):,} unique authors to {authors_file}")
    
    # Save institution list for Phase 3
    institutions_file = output_dir / "institutions_to_fetch.json"
    with open(institutions_file, "w") as f:
        json.dump(list(unique_institutions.values()), f, indent=2)
    print(f"   Saved {len(unique_institutions):,} unique institutions to {institutions_file}")
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Total papers processed: {len(results['processed_arxiv_ids']):,}")
    print(f"  Papers found in OpenAlex: {found_count:,} ({100*found_count/len(results['processed_arxiv_ids']):.1f}%)")
    print(f"  Papers not found: {not_found_count:,}")
    print(f"  Unique authors: {len(unique_authors):,}")
    print(f"  Unique institutions: {len(unique_institutions):,}")
    
    # Clean up checkpoints
    for cp in output_dir.glob("checkpoint_*.json"):
        cp.unlink()
    print("\n  Cleaned up checkpoint files.")
    print("\nPhase 1 complete! Run fetch_author_profiles.py next.")


if __name__ == "__main__":
    main()
