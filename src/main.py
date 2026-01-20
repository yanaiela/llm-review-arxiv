"""
Main entry point for the AI review paper detection study.
"""

import argparse
import logging
from pathlib import Path
import yaml

from src.data_collection.arxiv_sampler import sample_arxiv_papers
from src.preprocessing.kaggle_prepare_metadata import prepare_kaggle_metadata
from src.classification.llm_classifier import classify_papers_llm
from src.detection.detect_ai_content import detect_ai_content
from src.detection.pangram_detector import detect_pangram_content
from src.analysis.statistical_analysis import run_statistical_analysis
from src.visualization.create_plots import create_visualizations


def setup_logging():
    """Configure logging for the project."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ai_review_study.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config: dict):
    """Create necessary directories for the project."""
    for dir_name in config['output']['directories'].values():
        Path(dir_name).mkdir(parents=True, exist_ok=True)


def main():
    """Main pipeline for the study."""
    parser = argparse.ArgumentParser(
        description="AI-Generated Content in Review Papers Study"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['collect', 'preprocess', 'classify', 'detect', 'analyze', 'visualize', 'all'],
        default='all',
        help='Which stage of the pipeline to run'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download if papers already exist'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='ollama/gemma2',
        help='LLM model to use for classification (default: ollama/gemma2)'
    )
    parser.add_argument(
        '--llm-api-base',
        type=str,
        default=None,
        help='Optional OpenAI-compatible base URL for local vLLM server (e.g. http://localhost:8000/v1)'
    )
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        default=['cs'],
        help='arXiv categories to filter by (e.g., cs, cs.AI, cs.LG). Default: cs (all CS papers)'
    )
    parser.add_argument(
        '--detection-method',
        type=str,
        choices=['alpha', 'pangram'],
        default='alpha',
        help='Detection method: alpha (distributional) or pangram (per-paper API)'
    )

    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting AI review paper detection study")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create category-specific directory structure
    category_str = '_'.join(args.categories).replace('.', '-')
    logger.info(f"Using categories: {args.categories}")
    logger.info(f"Category identifier: {category_str}")
    
    # Update output directories to be category-specific
    for key in config['output']['directories']:
        original_path = config['output']['directories'][key]
        # Add category as subdirectory
        path_obj = Path(original_path)
        new_path = path_obj / category_str
        config['output']['directories'][key] = str(new_path)
    
    # Store categories in config for downstream use
    config['categories'] = args.categories
    
    create_directories(config)
    
    # Use LLM-based classifier
    logger.info(f"Using LLM-based classifier with model: {args.llm_model}")
    if args.llm_api_base:
        logger.info(f"LLM API base: {args.llm_api_base}")
    classify_func = lambda cfg: classify_papers_llm(cfg, model=args.llm_model, api_base=args.llm_api_base)

    # Select detection method(s)
    logger.info(f"Using detection method: {args.detection_method}")
    if args.detection_method == 'alpha':
        detect_func = detect_ai_content
    elif args.detection_method == 'pangram':
        detect_func = detect_pangram_content
    else:
        raise ValueError("Invalid detection method specified.")

    # Create analysis function with detection method
    analyze_func = lambda cfg: run_statistical_analysis(cfg, detection_method=args.detection_method)

    # Create visualization function with detection method
    visualize_func = lambda cfg: create_visualizations(cfg, detection_method=args.detection_method)

    # Run pipeline stages
    stages = {
        'collect': sample_arxiv_papers,
        'preprocess': prepare_kaggle_metadata,
        'classify': classify_func,
        'detect': detect_func,
        'analyze': analyze_func,
        'visualize': visualize_func,
    }
    
    if args.stage == 'all':
        stages_to_run = stages.keys()
    else:
        stages_to_run = [args.stage]
    
    for stage_name in stages_to_run:
        if stage_name == 'collect' and args.skip_download:
            logger.info("Skipping data collection as requested")
            continue
        
        logger.info(f"Running stage: {stage_name}")
        try:
            stages[stage_name](config)
            logger.info(f"Completed stage: {stage_name}")
        except Exception as e:
            logger.error(f"Error in stage {stage_name}: {str(e)}", exc_info=True)
            raise
    
    logger.info("Study pipeline completed successfully")


if __name__ == "__main__":
    main()
