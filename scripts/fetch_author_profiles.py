#!/usr/bin/env python3
"""
Phase 2: Fetch full author profiles from OpenAlex.

This script:
1. Loads unique author IDs from Phase 1 output
2. Batch queries OpenAlex /authors endpoint
3. Extracts career stage indicators (h-index, works_count, career length)
4. Saves author profiles for equity analysis

Optimized for low memory usage by writing results incrementally.
"""

import json
import time
import argparse
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error

# Configuration
OPENALEX_BASE_URL = "https://api.openalex.org"
BATCH_SIZE = 50  # Max IDs per request
RATE_LIMIT_DELAY = 0.1  # 10 req/sec with polite pool
SAVE_INTERVAL = 50  # Save to disk every N batches to reduce memory


def batch_query_authors(author_ids: list[str], email: str) -> dict:
    """Query OpenAlex for a batch of author IDs."""
    # Construct full OpenAlex URLs for filter
    full_ids = [f"https://openalex.org/{aid}" for aid in author_ids]
    id_filter = "|".join(full_ids)
    
    params = {
        "filter": f"ids.openalex:{id_filter}",
        "select": "id,display_name,orcid,works_count,cited_by_count,summary_stats,counts_by_year,last_known_institutions,affiliations",
        "per-page": str(BATCH_SIZE),
        "mailto": email
    }
    
    url = f"{OPENALEX_BASE_URL}/authors?{urllib.parse.urlencode(params)}"
    
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


def extract_author_profile(author: dict) -> dict:
    """Extract relevant fields from OpenAlex author object."""
    # Get summary stats
    summary_stats = author.get("summary_stats", {})
    
    # Get counts by year to derive career start
    counts_by_year = author.get("counts_by_year", [])
    first_publication_year = None
    if counts_by_year:
        years_with_works = [c["year"] for c in counts_by_year if c.get("works_count", 0) > 0]
        if years_with_works:
            first_publication_year = min(years_with_works)
    
    # Extract last known institutions
    last_known_institutions = []
    for inst in author.get("last_known_institutions", []) or []:
        last_known_institutions.append({
            "id": inst.get("id", "").split("/")[-1],
            "name": inst.get("display_name", ""),
            "country_code": inst.get("country_code", ""),
            "type": inst.get("type", "")
        })
    
    # Extract affiliation history
    affiliations = []
    for aff in author.get("affiliations", []) or []:
        inst = aff.get("institution", {})
        affiliations.append({
            "institution_id": inst.get("id", "").split("/")[-1],
            "institution_name": inst.get("display_name", ""),
            "country_code": inst.get("country_code", ""),
            "type": inst.get("type", ""),
            "years": aff.get("years", [])
        })
    
    return {
        "author_id": author.get("id", "").split("/")[-1],
        "display_name": author.get("display_name", ""),
        "orcid": author.get("orcid"),
        "works_count": author.get("works_count", 0),
        "cited_by_count": author.get("cited_by_count", 0),
        "h_index": summary_stats.get("h_index", 0),
        "i10_index": summary_stats.get("i10_index", 0),
        "2yr_mean_citedness": summary_stats.get("2yr_mean_citedness", 0),
        "first_publication_year": first_publication_year,
        "last_known_institutions": last_known_institutions,
        # Skip full affiliations and counts_by_year to save memory
        # Just keep count of affiliations
        "num_affiliations": len(affiliations)
    }


def load_processed_ids(output_dir: Path) -> set:
    """Load set of already processed author IDs from existing output file."""
    profiles_file = output_dir / "author_profiles.jsonl"
    processed = set()
    if profiles_file.exists():
        with open(profiles_file) as f:
            for line in f:
                try:
                    profile = json.loads(line.strip())
                    processed.add(profile.get("author_id", ""))
                except:
                    continue
    return processed


def append_profiles(profiles: list[dict], output_dir: Path):
    """Append profiles to JSONL file (one JSON object per line)."""
    profiles_file = output_dir / "author_profiles.jsonl"
    with open(profiles_file, "a") as f:
        for profile in profiles:
            f.write(json.dumps(profile) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fetch OpenAlex author profiles")
    parser.add_argument("--email", required=True, help="Email for OpenAlex polite pool")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of authors (0 = no limit)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = data_dir / "openalex"
    
    print("Phase 2: Fetching OpenAlex Author Profiles (Memory-Optimized)")
    print("=" * 50)
    
    # Load authors to fetch
    authors_file = output_dir / "authors_to_fetch.json"
    if not authors_file.exists():
        print(f"Error: {authors_file} not found. Run fetch_openalex_data.py first.")
        return
    
    print("\n1. Loading authors to fetch...")
    with open(authors_file) as f:
        authors_to_fetch = json.load(f)
    print(f"   Total unique authors: {len(authors_to_fetch):,}")
    
    # Load already processed IDs if resuming
    processed_ids = set()
    if args.resume:
        print("\n2. Loading already processed authors...")
        processed_ids = load_processed_ids(output_dir)
        print(f"   Previously processed: {len(processed_ids):,}")
    
    # Filter authors not yet processed
    authors_to_process = [a for a in authors_to_fetch if a["author_id"] not in processed_ids]
    
    # Free memory from full list
    del authors_to_fetch
    
    if args.limit > 0:
        authors_to_process = authors_to_process[:args.limit]
    
    print(f"\n3. Authors to process: {len(authors_to_process):,}")
    
    # Process in batches
    total_batches = (len(authors_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"   Batches: {total_batches:,}")
    print(f"   Estimated time: {total_batches * RATE_LIMIT_DELAY / 60:.1f} minutes")
    
    found_count = len(processed_ids)  # Count previously found
    not_found_count = 0
    batch_profiles = []  # Buffer for batch writing
    
    print("\n4. Fetching from OpenAlex API...")
    for batch_idx in range(0, len(authors_to_process), BATCH_SIZE):
        batch = authors_to_process[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        
        # Get author IDs for batch
        author_ids = [a["author_id"] for a in batch]
        
        # Query OpenAlex
        response = batch_query_authors(author_ids, args.email)
        
        # Index results by ID
        id_to_author = {}
        for author in response.get("results", []):
            author_id = author.get("id", "").split("/")[-1]
            if author_id:
                id_to_author[author_id] = author
        
        # Extract profiles
        for author_info in batch:
            author_id = author_info["author_id"]
            
            if author_id in id_to_author:
                profile = extract_author_profile(id_to_author[author_id])
                batch_profiles.append(profile)
                found_count += 1
            else:
                # Store minimal info for authors not found
                batch_profiles.append({
                    "author_id": author_id,
                    "display_name": author_info.get("author_name", ""),
                    "orcid": author_info.get("orcid"),
                    "not_found": True
                })
                not_found_count += 1
        
        # Save to disk periodically to free memory
        if batch_num % SAVE_INTERVAL == 0 or batch_num == total_batches:
            append_profiles(batch_profiles, output_dir)
            batch_profiles = []  # Clear buffer
        
        # Progress update
        if batch_num % 100 == 0 or batch_num == total_batches:
            print(f"   Batch {batch_num}/{total_batches}: found={found_count:,}, not_found={not_found_count:,}")
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Save any remaining profiles
    if batch_profiles:
        append_profiles(batch_profiles, output_dir)
    
    print("\n5. Done!")
    
    # Summary statistics (read from file to avoid memory issues)
    print("\n" + "=" * 50)
    print("Summary:")
    total_processed = len(processed_ids) + len(authors_to_process)
    print(f"  Total authors processed: {total_processed:,}")
    print(f"  Authors found: {found_count:,}")
    print(f"  Authors not found: {not_found_count:,}")
    print(f"\n  Results saved to: {output_dir / 'author_profiles.jsonl'}")
    print("\nPhase 2 complete!")


if __name__ == "__main__":
    main()
