# Generated Tables for Paper Appendix

This directory contains auto-generated LaTeX tables that replicate the format used in the paper's appendix.

## Files

### Main Output Files

1. **`appendix_all_tables_with_estimates.tex`** - Complete appendix section with all tables
   - Ready to include in your LaTeX paper with `\input{...}`
   - Contains tables for all four categories with estimated totals

2. **`all_tables.tex`** - Simpler version with sample counts only (no estimates)
   - Good for supplementary materials or presentations

### Individual Table Files

**With Estimates (matching paper format):**
- `appendix_table_cs.tex` - Computer Science
- `appendix_table_math.tex` - Mathematics
- `appendix_table_stat.tex` - Statistics
- `appendix_table_physics.tex` - Physics

**Without Estimates (simpler format):**
- `table_cs.tex`
- `table_math.tex`
- `table_stat.tex`
- `table_physics.tex`
- `table_summary.tex` - Summary across all categories

### CSV Data Files

- `stats_with_estimates_*.csv` - Statistics with estimated totals
- `stats_*.csv` - Sample counts only

## Table Format

The tables in the appendix format include:

| Year | Review Papers ||| Regular Papers |||
|------|------|------|------|------|------|------|
|      | Sample | Est. Total | % | Sample | Est. Total | % |

Where:
- **Sample**: Number of papers in our sampled dataset
- **Est. Total**: Estimated total papers on arXiv (extrapolated)
- **%**: Percentage of papers in that category

## Using in Your Paper

To include the complete appendix section in your paper:

```latex
\appendix

\input{data/tables/appendix_all_tables_with_estimates.tex}
```

Or to include individual tables:

```latex
\input{data/tables/appendix_table_cs.tex}
```

## Data Summary

### Computer Science
- Sample: 36,000 papers (6,000/year)
- Estimated Total: ~434,000 papers
- Review papers: 9.3% of total

### Mathematics
- Sample: 35,945 papers
- Estimated Total: ~238,000 papers
- Review papers: 5.5% of total

### Statistics
- Sample: 16,425 papers
- Estimated Total: ~61,000 papers
- Review papers: 11.4% of total

### Physics
- Sample: 35,677 papers
- Estimated Total: ~296,000 papers
- Review papers: 5.8% of total

## Regenerating Tables

To regenerate the tables with updated data:

```bash
# Simple tables (sample counts only)
python scripts/generate_appendix_tables.py

# Tables with estimated totals (matching paper appendix format)
python scripts/generate_appendix_tables_with_estimates.py
```

## Notes on Estimation

The estimated totals are extrapolated based on:
1. The sampling ratio between our dataset and arXiv's total submissions
2. ArXiv submission statistics from https://arxiv.org/stats/monthly_submissions
3. Known growth patterns in each category

The estimates are approximate but provide a reasonable extrapolation to the full arXiv corpus. You may want to update the `get_arxiv_estimate()` function in the script with more precise arXiv statistics for better accuracy.
