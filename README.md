# AI-Generated Content in Academic Papers: A Comparative Study

This project investigates whether review/survey papers contain more AI-generated content than regular research papers in the post-LLM era (2023-2025).

We would like to test the underlying hypothesis in arxiv's decision to stop uploading position papers: 
https://blog.arxiv.org/2025/10/31/attention-authors-updated-practice-for-review-articles-and-position-papers-in-arxiv-cs-category/?ref=404media.co

## Project Structure

```
.
├── README.md                  # This file
├── RESEARCH_PLAN.md          # Detailed research methodology
├── requirements.txt          # Python dependencies
├── config/
│   └── config.yaml           # Configuration parameters
├── src/
│   ├── data_collection/      # Scripts to download papers
│   ├── preprocessing/        # PDF extraction and cleaning
│   ├── classification/       # Paper type classification
│   ├── detection/            # AI content detection methods
│   ├── analysis/             # Statistical analysis
│   └── visualization/        # Plotting and reporting
├── data/
│   ├── raw/                  # Downloaded PDFs
│   ├── processed/            # Extracted text
│   └── results/              # Analysis outputs
├── notebooks/                # Jupyter notebooks for exploration
└── tests/                    # Unit tests
```

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Configure the project**:
   - Edit `config/config.yaml` with your API keys and parameters

3. **Run the pipeline**:
   ```bash
   python src/main.py --stage all
   ```

4. **View results**:
   - Statistical reports: `data/results/statistical_report.txt`
   - Visualizations: `data/results/figures/`

## Pipeline Stages

The analysis pipeline is modular and consists of these stages:

1. **collect**: Sample papers from Kaggle arXiv dataset
2. **preprocess**: Prepare metadata from sampled papers
3. **classify**: Identify review/survey vs regular papers
4. **detect**: Run AI content detection
5. **analyze**: Perform statistical analysis
6. **visualize**: Generate plots and charts

Run individual stages:
```bash
python src/main.py --stage classify
python src/main.py --stage detect
python src/main.py --stage analyze
```

## Key Components

### 1. Data Collection
- Samples papers from Kaggle arXiv dataset
- Filters by date ranges and keywords
- Organizes papers by type and period

### 2. Classification
- Identifies review/survey papers vs regular research papers
- Uses keywords, citation patterns, and structural features

### 3. AI Detection
- Multiple detection methods:
  - Perplexity analysis using language models
  - Linguistic feature analysis
  - Statistical markers
- Produces AI-likelihood scores (0-1)

### 4. Statistical Analysis
- Chi-square tests for proportion comparison
- T-tests for mean score comparison
- Regression analysis controlling for confounders
- Effect size calculations

### 5. Visualization
- Distribution plots
- Comparison charts
- Time series analysis
- Heatmaps by field and paper type

## Research Question

**Are review/survey papers more likely to contain AI-generated content than regular research papers?**

## Methodology Summary

- **Sample Size**: 500-1000 papers per category
- **Time Periods**: 
  - Pre-LLM: 2020-2022 (baseline)
  - Post-LLM: 2023-2025 (test period)
- **Detection**: Multiple methods with composite scoring
- **Analysis**: Multiple statistical tests with significance level α = 0.05

## Expected Outputs

1. Dataset of analyzed papers with AI detection scores
2. Statistical report with hypothesis test results
3. Visualizations showing comparative patterns
4. Publication-ready findings

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaboration, please open an issue on GitHub.
