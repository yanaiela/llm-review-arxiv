#!/usr/bin/env python3
"""
Phase 3: Fetch institution profiles from OpenAlex.

This script:
1. Loads unique institution IDs from Phase 1 output
2. Batch queries OpenAlex /institutions endpoint
3. Extracts prestige indicators (h-index, works_count, citations)
4. Saves institution profiles for equity analysis
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

# Global South countries (based on common definitions)
# Source: https://meta.wikimedia.org/wiki/List_of_countries_by_regional_classification
GLOBAL_SOUTH_COUNTRIES = {
    # Africa
    "DZ", "AO", "BJ", "BW", "BF", "BI", "CV", "CM", "CF", "TD", "KM", "CG", "CD", 
    "CI", "DJ", "EG", "GQ", "ER", "SZ", "ET", "GA", "GM", "GH", "GN", "GW", "KE", 
    "LS", "LR", "LY", "MG", "MW", "ML", "MR", "MU", "MA", "MZ", "NA", "NE", "NG", 
    "RW", "ST", "SN", "SC", "SL", "SO", "ZA", "SS", "SD", "TZ", "TG", "TN", "UG", 
    "ZM", "ZW",
    # Asia (excluding high-income)
    "AF", "BD", "BT", "KH", "IN", "ID", "IR", "IQ", "JO", "KZ", "KG", "LA", "LB", 
    "MY", "MV", "MN", "MM", "NP", "PK", "PH", "LK", "SY", "TJ", "TH", "TL", "TM", 
    "UZ", "VN", "YE", "PS",
    # Latin America & Caribbean
    "AR", "BZ", "BO", "BR", "CL", "CO", "CR", "CU", "DO", "EC", "SV", "GT", "GY", 
    "HT", "HN", "JM", "MX", "NI", "PA", "PY", "PE", "SR", "TT", "UY", "VE",
    # Oceania (excluding Australia, NZ)
    "FJ", "PG", "WS", "SB", "TO", "VU",
    # China (often included depending on definition)
    "CN"
}


def batch_query_institutions(institution_ids: list[str], email: str) -> dict:
    """Query OpenAlex for a batch of institution IDs."""
    # Construct full OpenAlex URLs for filter
    full_ids = [f"https://openalex.org/{iid}" for iid in institution_ids]
    id_filter = "|".join(full_ids)
    
    params = {
        "filter": f"ids.openalex:{id_filter}",
        "select": "id,display_name,country_code,type,works_count,cited_by_count,summary_stats,geo,ror",
        "per-page": str(BATCH_SIZE),
        "mailto": email
    }
    
    url = f"{OPENALEX_BASE_URL}/institutions?{urllib.parse.urlencode(params)}"
    
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


def extract_institution_profile(inst: dict) -> dict:
    """Extract relevant fields from OpenAlex institution object."""
    summary_stats = inst.get("summary_stats", {})
    geo = inst.get("geo", {})
    country_code = inst.get("country_code", "")
    
    return {
        "institution_id": inst.get("id", "").split("/")[-1],
        "display_name": inst.get("display_name", ""),
        "ror": inst.get("ror"),
        "country_code": country_code,
        "type": inst.get("type", ""),
        "works_count": inst.get("works_count", 0),
        "cited_by_count": inst.get("cited_by_count", 0),
        "h_index": summary_stats.get("h_index", 0),
        "i10_index": summary_stats.get("i10_index", 0),
        "2yr_mean_citedness": summary_stats.get("2yr_mean_citedness", 0),
        "is_global_south": country_code in GLOBAL_SOUTH_COUNTRIES,
        "geo": {
            "city": geo.get("city", ""),
            "region": geo.get("region"),
            "country": geo.get("country", ""),
            "latitude": geo.get("latitude"),
            "longitude": geo.get("longitude")
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch OpenAlex institution profiles")
    parser.add_argument("--email", required=True, help="Email for OpenAlex polite pool")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of institutions (0 = no limit)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = data_dir / "openalex"
    
    print("Phase 3: Fetching OpenAlex Institution Profiles")
    print("=" * 50)
    
    # Load institutions to fetch
    institutions_file = output_dir / "institutions_to_fetch.json"
    if not institutions_file.exists():
        print(f"Error: {institutions_file} not found. Run fetch_openalex_data.py first.")
        return
    
    print("\n1. Loading institutions to fetch...")
    with open(institutions_file) as f:
        institutions_to_fetch = json.load(f)
    print(f"   Total unique institutions: {len(institutions_to_fetch):,}")
    
    if args.limit > 0:
        institutions_to_fetch = institutions_to_fetch[:args.limit]
        print(f"   Limited to: {len(institutions_to_fetch):,}")
    
    # Process in batches
    total_batches = (len(institutions_to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n2. Processing in {total_batches} batches...")
    print(f"   Estimated time: {total_batches * RATE_LIMIT_DELAY / 60:.1f} minutes")
    
    results = {}
    found_count = 0
    not_found_count = 0
    
    print("\n3. Fetching from OpenAlex API...")
    for batch_idx in range(0, len(institutions_to_fetch), BATCH_SIZE):
        batch = institutions_to_fetch[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        
        # Get institution IDs for batch
        inst_ids = [i["institution_id"] for i in batch]
        
        # Query OpenAlex
        response = batch_query_institutions(inst_ids, args.email)
        
        # Index results by ID
        id_to_inst = {}
        for inst in response.get("results", []):
            inst_id = inst.get("id", "").split("/")[-1]
            if inst_id:
                id_to_inst[inst_id] = inst
        
        # Extract profiles
        for inst_info in batch:
            inst_id = inst_info["institution_id"]
            
            if inst_id in id_to_inst:
                profile = extract_institution_profile(id_to_inst[inst_id])
                results[inst_id] = profile
                found_count += 1
            else:
                # Store minimal info for institutions not found
                results[inst_id] = {
                    "institution_id": inst_id,
                    "display_name": inst_info.get("institution_name", ""),
                    "not_found": True
                }
                not_found_count += 1
        
        # Progress update
        if batch_num % 20 == 0 or batch_num == total_batches:
            print(f"   Batch {batch_num}/{total_batches}: found={found_count:,}, not_found={not_found_count:,}")
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Save final results
    print("\n4. Saving results...")
    
    profiles_file = output_dir / "institution_profiles.json"
    with open(profiles_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Saved {len(results):,} institution profiles to {profiles_file}")
    
    # Summary statistics
    valid_profiles = [p for p in results.values() if not p.get("not_found")]
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Total institutions processed: {len(results):,}")
    print(f"  Institutions found: {found_count:,} ({100*found_count/len(results):.1f}%)")
    print(f"  Institutions not found: {not_found_count:,}")
    
    if valid_profiles:
        # Type breakdown
        types = {}
        for p in valid_profiles:
            t = p.get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        print(f"\n  Institution types:")
        for t, count in sorted(types.items(), key=lambda x: -x[1])[:5]:
            print(f"    {t}: {count:,}")
        
        # Global South breakdown
        gs_count = sum(1 for p in valid_profiles if p.get("is_global_south"))
        print(f"\n  Global South institutions: {gs_count:,} ({100*gs_count/len(valid_profiles):.1f}%)")
        
        # Country breakdown (top 10)
        countries = {}
        for p in valid_profiles:
            c = p.get("country_code", "unknown")
            countries[c] = countries.get(c, 0) + 1
        print(f"\n  Top countries:")
        for c, count in sorted(countries.items(), key=lambda x: -x[1])[:10]:
            print(f"    {c}: {count:,}")
        
        # H-index stats
        h_indices = [p.get("h_index", 0) for p in valid_profiles]
        print(f"\n  H-index stats: median={sorted(h_indices)[len(h_indices)//2]}, max={max(h_indices)}")
    
    print("\nPhase 3 complete! All data fetched. Run equity_analysis.ipynb next.")


if __name__ == "__main__":
    main()
