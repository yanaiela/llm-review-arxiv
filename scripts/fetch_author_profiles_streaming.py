#!/usr/bin/env python3
"""
Phase 2: Fetch full author profiles from OpenAlex.
STREAMING VERSION - minimal memory usage.

This script:
1. Streams through author IDs without loading all into memory
2. Batch queries OpenAlex /authors endpoint  
3. Writes results incrementally to JSONL
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
BATCH_SIZE = 50
RATE_LIMIT_DELAY = 0.1


def batch_query_authors(author_ids: list, email: str) -> dict:
    """Query OpenAlex for a batch of author IDs."""
    full_ids = [f"https://openalex.org/{aid}" for aid in author_ids]
    id_filter = "|".join(full_ids)
    
    params = {
        "filter": f"ids.openalex:{id_filter}",
        "select": "id,display_name,orcid,works_count,cited_by_count,summary_stats,last_known_institutions",
        "per-page": str(BATCH_SIZE),
        "mailto": email
    }
    
    url = f"{OPENALEX_BASE_URL}/authors?{urllib.parse.urlencode(params)}"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"    Error: {e}")
        return {"results": []}


def extract_author_profile(author: dict) -> dict:
    """Extract relevant fields from OpenAlex author object."""
    summary_stats = author.get("summary_stats") or {}
    
    last_known_institutions = []
    for inst in (author.get("last_known_institutions") or [])[:3]:  # Limit to 3
        if inst:
            inst_id = (inst.get("id") or "").split("/")[-1]
            last_known_institutions.append({
                "id": inst_id,
                "name": inst.get("display_name") or "",
                "country_code": inst.get("country_code") or "",
                "type": inst.get("type") or ""
            })
    
    author_id = (author.get("id") or "").split("/")[-1]
    
    return {
        "author_id": author_id,
        "display_name": author.get("display_name") or "",
        "orcid": author.get("orcid"),
        "works_count": author.get("works_count") or 0,
        "cited_by_count": author.get("cited_by_count") or 0,
        "h_index": summary_stats.get("h_index") or 0,
        "i10_index": summary_stats.get("i10_index") or 0,
        "2yr_mean_citedness": summary_stats.get("2yr_mean_citedness") or 0,
        "last_known_institutions": last_known_institutions
    }


def iter_json_array(filepath):
    """Iterate through a JSON array file without loading it all into memory."""
    import re
    with open(filepath) as f:
        content = ""
        brace_count = 0
        in_object = False
        
        for line in f:
            for char in line:
                if char == '{':
                    in_object = True
                    brace_count += 1
                    content += char
                elif char == '}':
                    brace_count -= 1
                    content += char
                    if brace_count == 0 and in_object:
                        try:
                            yield json.loads(content)
                        except:
                            pass
                        content = ""
                        in_object = False
                elif in_object:
                    content += char


def main():
    parser = argparse.ArgumentParser(description="Fetch OpenAlex author profiles (streaming)")
    parser.add_argument("--email", required=True, help="Email for OpenAlex polite pool")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = data_dir / "openalex"
    
    authors_file = output_dir / "authors_to_fetch.json"
    output_file = output_dir / "author_profiles.jsonl"
    
    print("Phase 2: Fetching OpenAlex Author Profiles (Streaming)")
    print("=" * 50)
    
    # Load already processed IDs
    processed_ids = set()
    if output_file.exists():
        print("\n1. Loading already processed authors...")
        with open(output_file) as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line.strip()).get("author_id", ""))
                except:
                    pass
        print(f"   Previously processed: {len(processed_ids):,}")
    
    # Process in streaming fashion
    print("\n2. Fetching from OpenAlex API...")
    
    found_count = 0
    not_found_count = 0
    batch = []
    batch_num = 0
    skipped = 0
    
    with open(output_file, "a") as f_out:
        for author_info in iter_json_array(authors_file):
            author_id = author_info.get("author_id", "")
            if not author_id or author_id in processed_ids:
                skipped += 1
                continue
            
            batch.append(author_info)
            
            if len(batch) >= BATCH_SIZE:
                batch_num += 1
                
                # Query OpenAlex
                author_ids = [a["author_id"] for a in batch]
                response = batch_query_authors(author_ids, args.email)
                
                # Index results
                id_to_author = {}
                for author in response.get("results", []):
                    aid = (author.get("id") or "").split("/")[-1]
                    if aid:
                        id_to_author[aid] = author
                
                # Write results
                for author_info in batch:
                    aid = author_info["author_id"]
                    if aid in id_to_author:
                        profile = extract_author_profile(id_to_author[aid])
                        found_count += 1
                    else:
                        profile = {
                            "author_id": aid,
                            "display_name": author_info.get("author_name", ""),
                            "not_found": True
                        }
                        not_found_count += 1
                    f_out.write(json.dumps(profile) + "\n")
                    f_out.flush()  # Ensure writes are saved
                
                # Clear batch
                batch = []
                
                # Progress
                if batch_num % 100 == 0:
                    print(f"   Batch {batch_num}: found={found_count:,}, not_found={not_found_count:,}")
                
                time.sleep(RATE_LIMIT_DELAY)
        
        # Process remaining
        if batch:
            batch_num += 1
            author_ids = [a["author_id"] for a in batch]
            response = batch_query_authors(author_ids, args.email)
            
            id_to_author = {}
            for author in response.get("results", []):
                aid = (author.get("id") or "").split("/")[-1]
                if aid:
                    id_to_author[aid] = author
            
            for author_info in batch:
                aid = author_info["author_id"]
                if aid in id_to_author:
                    profile = extract_author_profile(id_to_author[aid])
                    found_count += 1
                else:
                    profile = {
                        "author_id": aid,
                        "display_name": author_info.get("author_name", ""),
                        "not_found": True
                    }
                    not_found_count += 1
                f_out.write(json.dumps(profile) + "\n")
    
    print(f"\n   Final: {batch_num} batches, found={found_count:,}, not_found={not_found_count:,}, skipped={skipped:,}")
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  New authors found: {found_count:,}")
    print(f"  New authors not found: {not_found_count:,}")
    print(f"  Already processed (skipped): {skipped:,}")
    print(f"  Results saved to: {output_file}")
    print("\nPhase 2 complete!")


if __name__ == "__main__":
    main()
