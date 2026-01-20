"""
AI content detection using multiple methods.
"""

import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from .alpha_detector import AlphaDetector
from ..utils import MAX_TEXT_LENGTH

logger = logging.getLogger(__name__)


class AIContentDetector:
    """Detect AI-generated content using multiple methods."""
    
    def __init__(self, config: dict):
        self.config = config
        self.detection_config = config['detection']
        
        # Load perplexity model
        logger.info("Loading GPT-2 model for perplexity calculation...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        # Set pad_token for batch processing
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        
        # AI indicator phrases
        self.ai_phrases = self.detection_config['ai_indicator_phrases']
        
        # Alpha detector for distributional AI detection
        logger.info("Loading alpha estimation model...")
        self.alpha_detector = AlphaDetector(config)
    
    def calculate_perplexity(self, text: str, max_length: int = 512) -> float:
        """
        Calculate perplexity of text using GPT-2.
        Lower perplexity may indicate AI-generated content.

        Args:
            text: Input text
            max_length: Maximum sequence length

        Returns:
            Perplexity score
        """
        try:
            # Tokenize
            encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to(self.device)

            # Calculate loss
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss

            # Perplexity is exp(loss)
            perplexity = torch.exp(loss).item()
            return perplexity

        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float('inf')


def detect_ai_content(config: dict):
    """
    Detect AI-generated content in all papers.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Starting AI content detection")
    
    # Load paper data (similar to llm_classifier pattern)
    processed_dir = Path(config['output']['directories']['processed_data'])
    metadata_path = processed_dir / 'paper_metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"paper_metadata.csv not found in {processed_dir}. Run preprocess stage first."
        )
    text_df = pd.read_csv(metadata_path)
    
    # Ensure period column exists and fill any NaN values
    if 'period' not in text_df.columns:
        text_df['period'] = 'unknown'
    else:
        text_df['period'] = text_df['period'].fillna('unknown')
    
    # Load classification results for paper_type
    try:
        class_df = pd.read_csv(processed_dir / 'paper_classifications.csv')
        text_df = pd.merge(text_df, class_df[['arxiv_id', 'predicted_type']], on='arxiv_id', how='left')
        text_df['paper_type'] = text_df['predicted_type']
        text_df.drop(columns=['predicted_type'], inplace=True)
        logger.info("Merged classification results (paper_type information)")
    except Exception as e:
        logger.warning(f"Could not load classifications: {e}")
        text_df['paper_type'] = 'unknown'
    
    # Construct full_text from title + abstract
    logger.info("Constructing text from title and abstract")
    text_df['full_text'] = (
        text_df['title'].fillna('').astype(str) + '\n\n' + 
        text_df['abstract'].fillna('').astype(str)
    )
    
    # ========================================================================
    # STEP 1: Batch alpha estimation by group (period × paper_type)
    # ========================================================================
    logger.info("Step 1: Running batch alpha estimation by group...")

    # Initialize alpha detector
    alpha_detector = AlphaDetector(config)

    # Group papers by period and paper_type
    # We want to estimate alpha separately for each cohesive dataset
    # Store group-level alpha estimates
    group_alpha_estimates = {}

    for period in text_df['period'].unique():
        for paper_type in text_df['paper_type'].unique():
            # Get papers in this group
            group_mask = (text_df['period'] == period) & (text_df['paper_type'] == paper_type)
            group_df = text_df[group_mask]

            if len(group_df) == 0:
                continue

            logger.info(f"Processing group: period={period}, paper_type={paper_type} ({len(group_df)} papers)")

            # Prepare texts for this group
            group_texts = {}
            for _, row in group_df.iterrows():
                arxiv_id = row['arxiv_id']
                full_text = row.get('full_text', '')
                if full_text and not pd.isna(full_text):
                    # Use first MAX_TEXT_LENGTH characters for analysis
                    group_texts[arxiv_id] = full_text[:MAX_TEXT_LENGTH]

            if len(group_texts) == 0:
                logger.warning(f"No valid texts for group: period={period}, paper_type={paper_type}")
                continue

            # Run batch inference for this group
            logger.info(f"Running alpha estimation for {len(group_texts)} papers in group...")
            alpha_estimate, alpha_ci = alpha_detector.estimate_alpha_batch(group_texts)

            # Store the group's alpha estimate
            group_key = (period, paper_type)
            group_alpha_estimates[group_key] = (alpha_estimate, alpha_ci)

            logger.info(f"Group alpha estimate: {alpha_estimate:.4f} ± {alpha_ci:.4f}")

    logger.info(f"Completed alpha estimation for {len(group_alpha_estimates)} groups")

    # ========================================================================
    # STEP 1b: Additional batch alpha estimation by year × paper_type for post-LLM
    # ========================================================================
    logger.info("Step 1b: Running batch alpha estimation by year × paper_type for post-LLM years...")

    # Extract year from submission_date for post-LLM papers
    text_df['year'] = pd.to_datetime(text_df['submission_date'], errors='coerce').dt.year

    # Store year-specific alpha estimates for post-LLM papers
    year_group_alpha_estimates = {}

    # Only process post-LLM years 2023, 2024, 2025
    post_llm_years = [2023, 2024, 2025]

    for year in post_llm_years:
        for paper_type in text_df['paper_type'].unique():
            # Get papers in this year × paper_type group (only post-LLM period)
            group_mask = (
                (text_df['year'] == year) &
                (text_df['paper_type'] == paper_type) &
                (text_df['period'] == 'post_llm')
            )
            group_df = text_df[group_mask]

            if len(group_df) == 0:
                continue

            logger.info(f"Processing group: year={year}, paper_type={paper_type} ({len(group_df)} papers)")

            # Prepare texts for this group
            group_texts = {}
            for _, row in group_df.iterrows():
                arxiv_id = row['arxiv_id']
                full_text = row.get('full_text', '')
                if full_text and not pd.isna(full_text):
                    # Use first MAX_TEXT_LENGTH characters for analysis
                    group_texts[arxiv_id] = full_text[:MAX_TEXT_LENGTH]

            if len(group_texts) == 0:
                logger.warning(f"No valid texts for group: year={year}, paper_type={paper_type}")
                continue

            # Run batch inference for this group
            logger.info(f"Running alpha estimation for {len(group_texts)} papers in year-specific group...")
            alpha_estimate, alpha_ci = alpha_detector.estimate_alpha_batch(group_texts)

            # Store the year-specific group's alpha estimate
            year_group_key = (year, paper_type)
            year_group_alpha_estimates[year_group_key] = (alpha_estimate, alpha_ci)

            logger.info(f"Year-specific group alpha estimate: {alpha_estimate:.4f} ± {alpha_ci:.4f}")

    logger.info(f"Completed year-specific alpha estimation for {len(year_group_alpha_estimates)} groups")

    # ========================================================================
    # STEP 2: Perplexity calculation (uses GPU)
    # ========================================================================
    logger.info("Step 2: Running perplexity calculation...")

    detector = AIContentDetector(config)

    # Calculate perplexity for each paper individually
    perplexity_map = {}

    for _, row in tqdm(text_df.iterrows(), total=len(text_df), desc="Calculating perplexity"):
        arxiv_id = row['arxiv_id']
        full_text = row.get('full_text', '')

        if full_text and not pd.isna(full_text):
            perplexity = detector.calculate_perplexity(full_text[:MAX_TEXT_LENGTH])
            perplexity_map[arxiv_id] = perplexity

    # ========================================================================
    # STEP 3: Combine all metrics into detection results
    # ========================================================================
    logger.info("Step 3: Combining metrics into final detection results...")

    detection_results = []

    for idx, row in tqdm(text_df.iterrows(), total=len(text_df), desc="Creating detection results"):
        arxiv_id = row['arxiv_id']
        full_text = row.get('full_text', '')

        if not full_text or pd.isna(full_text):
            logger.warning(f"No text for {arxiv_id}")
            continue

        # Get pre-computed values
        period = row.get('period', 'unknown')
        paper_type = row.get('paper_type', 'unknown')
        year = row.get('year', None)
        group_key = (period, paper_type)
        alpha_estimate, alpha_ci = group_alpha_estimates.get(group_key, (0.0, 0.0))
        perplexity = perplexity_map.get(arxiv_id, float('inf'))

        # Get year-specific alpha estimate for post-LLM papers
        year_alpha_estimate = None
        year_alpha_ci = None
        if period == 'post_llm' and year in [2023, 2024, 2025]:
            year_group_key = (year, paper_type)
            if year_group_key in year_group_alpha_estimates:
                year_alpha_estimate, year_alpha_ci = year_group_alpha_estimates[year_group_key]

        # Calculate composite scores
        try:
            results = {
                'arxiv_id': arxiv_id,
                'paper_type': paper_type,
                'period': period,
                'year': year if not pd.isna(year) else None,
                'perplexity': perplexity,
                'alpha_estimate': alpha_estimate,
                'alpha_ci_half_width': alpha_ci,
                'year_alpha_estimate': year_alpha_estimate,
                'year_alpha_ci_half_width': year_alpha_ci,
            }
            detection_results.append(results)

        except Exception as e:
            logger.error(f"Error creating detection results for {arxiv_id}: {e}")
    
    # Save results
    df_results = pd.DataFrame(detection_results)
    results_dir = Path(config['output']['directories']['results'])
    results_dir.mkdir(parents=True, exist_ok=True)

    df_results.to_csv(results_dir / 'ai_detection_results.csv', index=False)
    df_results.to_json(results_dir / 'ai_detection_results.json', orient='records', indent=2)
    logger.info(f"Saved detection results to {results_dir}")
    
    # Summary statistics
    logger.info(f"Analyzed {len(df_results)} papers")
    
    return df_results
