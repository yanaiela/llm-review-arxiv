"""
Alpha estimation module for detecting AI-generated content fraction.
Based on "Mapping the Increasing Use of LLMs in Scientific Papers" methodology.
"""

from .MLE import MLE
from .estimation import estimate_text_distribution

__all__ = ['MLE', 'estimate_text_distribution']
