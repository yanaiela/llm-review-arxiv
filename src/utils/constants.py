"""
Constants used throughout the AI review paper detection pipeline.

This module centralizes magic numbers and configuration values to improve
code maintainability and prevent duplication.
"""

# Text processing
MAX_TEXT_LENGTH = 10000  # Maximum text length for AI detection processing

# Detection thresholds
AI_CONTENT_THRESHOLD = 0.15  # Threshold for flagging high AI content

# LLM parameters
DEFAULT_MAX_TOKENS = 200  # Default maximum tokens for LLM responses
DEFAULT_RANDOM_SEED = 42  # Default random seed for reproducibility

# MLE parameters
DEFAULT_LOG_PROB_OOV = -13.8  # Default log probability for out-of-vocabulary words
DEFAULT_BOOTSTRAP_SAMPLES = 1000  # Default number of bootstrap samples for MLE

# Statistics
POST_LLM_START_YEAR = 2023  # Year when LLM era began (ChatGPT release)
