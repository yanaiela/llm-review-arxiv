"""
Alpha estimation detector for measuring fraction of AI-generated content.
Integrates the MLE-based alpha estimation from "Mapping the Increasing Use of LLMs in Scientific Papers".
"""

import logging
from pathlib import Path
import pandas as pd
import re
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

try:
    from .alpha_estimation.MLE import MLE
except ImportError:
    logger.warning("Alpha estimation MLE module not available")
    MLE = None


class AlphaDetector:
    """
    Estimates the fraction (alpha) of text that is AI-generated or substantially modified by AI.
    Uses distributional detection rather than instance-level detection.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.distribution_path = Path(__file__).parent / 'alpha_estimation' / 'CS.parquet'
        
        if not self.distribution_path.exists():
            logger.error(f"Distribution file not found: {self.distribution_path}")
            return
            
        try:
            if MLE is not None:
                self.model = MLE(str(self.distribution_path))
                logger.info("Alpha estimation model loaded successfully")
            else:
                logger.warning("MLE class not available, alpha estimation will be skipped")
        except Exception as e:
            logger.error(f"Error loading alpha estimation model: {e}")
            self.model = None
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Simple tokenization similar to the MLE paper.
        Converts text to lowercase and splits on non-alphanumeric characters.
        
        Args:
            text: Raw text to tokenize
            
        Returns:
            List of tokens
        """
        if not text or not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Split on non-alphanumeric characters and filter out empty strings
        tokens = re.findall(r'\b[a-z]+\b', text)
        
        return tokens
    
    def prepare_text_for_inference(self, text: str) -> List[List[str]]:
        """
        Prepares text for alpha inference by tokenizing into sentences.
        
        Args:
            text: Raw text content
            
        Returns:
            List of tokenized sentences
        """
        if not text or not isinstance(text, str):
            return [[]]
        
        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+', text)
        
        # Tokenize each sentence
        tokenized_sentences = []
        for sent in sentences:
            tokens = self.tokenize_text(sent)
            if len(tokens) > 1:  # Filter out very short sentences
                tokenized_sentences.append(tokens)
        
        return tokenized_sentences if tokenized_sentences else [[]]
    
    def estimate_alpha_batch(self, texts: Dict[str, str]) -> Tuple[float, float]:
        """
        Estimates alpha (fraction of AI-generated content) for a batch of texts.
        Returns a single alpha estimate for the entire cohesive dataset.

        Args:
            texts: Dictionary mapping paper IDs to raw text

        Returns:
            Tuple of (alpha estimate, confidence interval half-width) for the entire dataset
        """
        if self.model is None:
            logger.warning("Alpha estimation model not available")
            return (0.0, 0.0)

        if not texts:
            return (0.0, 0.0)

        try:
            # Prepare all texts as tokenized sentences
            rows = []
            for paper_id, text in texts.items():
                if not text or not isinstance(text, str):
                    continue

                tokenized_sentences = self.prepare_text_for_inference(text)

                if not tokenized_sentences or all(len(s) <= 1 for s in tokenized_sentences):
                    continue

                # Each sentence is a separate row
                for sent in tokenized_sentences:
                    rows.append({
                        'inference_sentence': sent
                    })

            if not rows:
                logger.warning("No valid texts for alpha estimation")
                return (0.0, 0.0)

            # Create dataframe in the format expected by MLE
            temp_df = pd.DataFrame(rows)

            # Save to temporary parquet file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                temp_path = tmp.name
                temp_df.to_parquet(temp_path, index=False)

            try:
                # Run inference on entire batch
                alpha_estimate, ci_half_width = self.model.inference(
                    temp_path,
                    exploded_data=True
                )

                return (float(alpha_estimate), float(ci_half_width))
            finally:
                # Clean up temporary file
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Error estimating alpha in batch: {e}")
            return (0.0, 0.0)

    def estimate_alpha(self, text: str, n_bootstrap: int = 100) -> Tuple[float, float]:
        """
        Estimates alpha (fraction of AI-generated content) for given text.

        Args:
            text: Raw text to analyze
            n_bootstrap: Number of bootstrap samples for confidence interval

        Returns:
            Tuple of (alpha estimate, confidence interval half-width)
        """
        # Use batch method for consistency
        return self.estimate_alpha_batch({'single': text})
