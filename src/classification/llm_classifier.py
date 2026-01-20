"""
LLM-based paper classification using LiteLLM.
Classifies papers as review/survey vs. other types.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import json
from tqdm import tqdm

# Suppress litellm logging completely
os.environ["LITELLM_LOG"] = "ERROR"
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False

# Disable litellm's internal loggers
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Proxy").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

from litellm import completion, batch_completion

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

logger = logging.getLogger(__name__)


class LLMPaperClassifier:
    """Classify papers using LLM (via LiteLLM).

    Supports local vLLM OpenAI-compatible servers by using a model name
    prefixed with 'vllm/' (e.g. 'vllm/mistral' or 'vllm/your-hf-model').
    If that prefix is detected, an OpenAI-compatible base URL is used.
    Default base URL can be provided via VLLM_BASE_URL env var or api_base arg.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        api_base: str | None = None
    ):
        """Initialize LLM classifier.

        Args:
            model: LiteLLM model name (e.g., 'gpt-4o-mini', 'claude-3-haiku-20240307', 'vllm/mistral')
            temperature: Generation temperature (0 = deterministic)
            api_base: Optional OpenAI-compatible base URL (used for local vLLM)
        """

        self.original_model = model
        self.temperature = temperature
        self.api_base = None
        self.use_vllm = False  # Track if we're using vLLM

        # Alias map for convenience / non-standard inputs
        self._alias_map = {
            # Simple alias
            "gemma2": "ollama/gemma2",
            # Common user inputs for gemma 2 instruct models
            "google/gemma-2-2b-i": "huggingface/google/gemma-2-2b-it",
            "google/gemma-2-2b-it": "huggingface/google/gemma-2-2b-it",
            "google/gemma-2-7b-it": "huggingface/google/gemma-2-7b-it",
        }

        # Resolve alias if provided
        if model in self._alias_map:
            logging.getLogger(__name__).info(f"Resolving model alias '{model}' -> '{self._alias_map[model]}'")
            model = self._alias_map[model]

        # Detect vLLM usage
        if model.startswith("vllm/"):
            self.use_vllm = True
            # Extract model name and use openai/ prefix for litellm compatibility
            model_name = model.split("/", 1)[1]
            # Use openai/ prefix so litellm treats it as OpenAI-compatible
            self.model = f"openai/{model_name}"
            # Resolve API base
            self.api_base = api_base or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            # Set dummy API key for local vLLM (required by litellm but not used)
            if not os.getenv("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = "dummy-key-for-local-vllm"
            logging.getLogger(__name__).info(
                f"Using local vLLM server at {self.api_base} with model '{model_name}' (via openai/ prefix, no system role)"
            )
        else:
            self.model = model

        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for classification."""
        return """You are an expert academic paper classifier. Your task is to classify papers into one of two categories:

1. **Review/Survey**: Papers that primarily review, survey, synthesize existing research, or present positions/perspectives on a field. These papers:
   - Systematically review existing literature
   - Provide comprehensive overviews of a research area
   - Summarize and compare multiple existing approaches
   - Present position papers or perspectives on research directions
   - Often have titles containing words like "survey", "review", "overview", "primer", "position", "perspective"

2. **Other**: All other papers including:
   - Original research papers presenting new methods, experiments, or results
   - Technical reports
   - Case studies
   - Papers presenting new datasets or benchmarks
   - Papers focused primarily on novel empirical results

You must respond with ONLY a JSON object in this exact format:
{"classification": "review", "reasoning": "brief explanation"}

OR

{"classification": "other", "reasoning": "brief explanation"}

The classification field must be either "review" or "other" (lowercase).
Keep the reasoning brief (1-2 sentences)."""

    def _build_user_prompt(self, title: str, abstract: str) -> str:
        """Build the user prompt with paper details."""
        return f"""Please classify the following paper:

Title: {title}

Abstract: {abstract}

Classify this paper as either "review" or "other" and respond with the JSON format specified."""

    def classify_paper(self, title: str, abstract: str) -> Dict:
        """
        Classify a single paper.
        
        Args:
            title: Paper title
            abstract: Paper abstract
        
        Returns:
            Dictionary with classification, confidence, and reasoning
        """
        try:
            # For vLLM, combine system prompt with user message (no system role support)
            if self.use_vllm:
                combined_prompt = f"{self.system_prompt}\n\n{self._build_user_prompt(title, abstract)}"
                messages = [{"role": "user", "content": combined_prompt}]
            else:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self._build_user_prompt(title, abstract)}
                ]
            
            completion_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 200
            }
            # Inject api_base if using local vLLM
            if self.api_base:
                completion_kwargs["api_base"] = self.api_base

            try:
                response = completion(**completion_kwargs)
            except Exception as e:
                msg = str(e)
                # Fallback: provider missing, attempt inference
                if "LLM Provider NOT provided" in msg:
                    # For vLLM we already injected api_base; if still failing, raise with guidance
                    if self.api_base:
                        raise RuntimeError(
                            f"vLLM server reachable but provider not recognized. Ensure litellm version is recent and api_base '{self.api_base}' exposes OpenAI /v1/chat/completions. Original error: {msg}"
                        )
                    raise
                else:
                    raise
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            # Handle cases where model wraps JSON in markdown code blocks
            if content.startswith("```"):
                # Extract JSON from markdown code block
                lines = content.split("\n")
                content = "\n".join([l for l in lines if l and not l.startswith("```")])
            
            result = json.loads(content)
            
            # Validate and normalize
            classification = result.get("classification", "").lower()
            if classification not in ["review", "other"]:
                logger.warning(f"Invalid classification '{classification}', defaulting to 'other'")
                classification = "other"
            
            return {
                "classification": classification,
                "reasoning": result.get("reasoning", "")
            }
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {content[:200]}")
            logger.error(f"Error: {e}")
            # Heuristic extraction
            lowered = content.lower()
            review_keywords = ["survey", "surveys", "review", "overview", "primer", "position", "perspective"]
            classification = "review" if any(k in lowered for k in review_keywords) else "other"
            # Attempt to extract reasoning fragment
            reasoning = "heuristic keyword match after JSON parse failure"
            import re
            reason_match = re.search(r"reasoning\s*[:=]\s*['\"]([^'\"]{10,200})['\"]", content)
            if reason_match:
                reasoning = reason_match.group(1).strip()
            return {
                "classification": classification,
                "reasoning": reasoning
            }
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return {
                "classification": "other",
                "reasoning": f"Error: {str(e)}"
            }


def evaluate_llm_classifier(
    eval_csv_path: str = "data/eval/maria-eval.csv",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    output_path: str = None
) -> Dict:
    """
    Evaluate LLM classifier on ground truth data.
    
    Args:
        eval_csv_path: Path to evaluation CSV with MARIA_ANNOTATION column
        model: LiteLLM model name
        temperature: Temperature for generation
        output_path: Optional path to save detailed results
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating LLM classifier (model={model}) on {eval_csv_path}")
    
    # Load evaluation data
    df = pd.read_csv(eval_csv_path)
    logger.info(f"Loaded {len(df)} papers for evaluation")
    logger.info(f"Ground truth distribution: {df['MARIA_ANNOTATION'].value_counts().to_dict()}")
    
    # Initialize classifier
    classifier = LLMPaperClassifier(model=model, temperature=temperature)
    
    # Classify each paper
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying papers"):
        title = str(row.get('title', ''))
        abstract = str(row.get('abstract', ''))
        ground_truth = int(row['MARIA_ANNOTATION'])
        
        # Get LLM classification
        llm_result = classifier.classify_paper(title, abstract)
        
        # Convert to binary: review=1, other=0
        predicted = 1 if llm_result['classification'] == 'review' else 0
        
        results.append({
            'arxiv_id': row['arxiv_id'],
            'title': title,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'llm_classification': llm_result['classification'],
            'reasoning': llm_result['reasoning']
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    y_true = results_df['ground_truth']
    y_pred = results_df['predicted']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=['Other', 'Review'], zero_division=0)
    
    # Compile metrics
    metrics = {
        'model': model,
        'temperature': temperature,
        'total_papers': len(df),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Model: {model}")
    logger.info(f"Total papers: {len(df)}")
    logger.info(f"\nAccuracy:  {accuracy:.3f}")
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall:    {recall:.3f}")
    logger.info(f"F1 Score:  {f1:.3f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"                Predicted")
    logger.info(f"              Other  Review")
    logger.info(f"Actual Other   {cm[0][0]:4d}   {cm[0][1]:4d}")
    logger.info(f"       Review  {cm[1][0]:4d}   {cm[1][1]:4d}")
    logger.info(f"\n{report}")
    logger.info("=" * 80)
    
    # Save detailed results if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved detailed results to {output_path}")
        
        # Save metrics
        metrics_path = output_path.parent / f"{output_path.stem}_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_metrics = {k: v if not isinstance(v, type(cm)) else v.tolist() 
                          for k, v in metrics.items()}
            json.dump(json_metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
    
    return metrics


def classify_papers_llm(config: dict, model: str = "ollama/gemma2", api_base: str | None = None) -> pd.DataFrame:
    """
    Classify papers using LLM (pipeline-compatible function).
    
    Args:
        config: Configuration dictionary
        model: LiteLLM model name (default: ollama/gemma2)
    
    Returns:
        DataFrame with classifications
    """
    logger.info(f"Running LLM-based classification with model: {model}")
    
    # Load extracted text (includes title, abstract, and full_text)
    processed_dir = Path(config['output']['directories']['processed_data'])
    extracted_text_path = processed_dir / 'extracted_text.csv'
    metadata_path = processed_dir / 'paper_metadata.csv'

    if extracted_text_path.exists():
        logger.info(f"Loading extracted text from {extracted_text_path}")
        df = pd.read_csv(extracted_text_path)
    else:
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Neither extracted_text.csv nor paper_metadata.csv found in {processed_dir}. Run preprocess stage first."
            )
        logger.warning(
            "Extracted text file missing; falling back to metadata only (title + abstract). "
            "Classification will be based solely on these fields."
        )
        df = pd.read_csv(metadata_path)

    # Shuffle to avoid ordering bias and limit size for cost control
    original_count = len(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # df = df.head(1000)
    logger.info(f"Loaded {original_count} rows; using {len(df)} for LLM classification")
    
    # Initialize classifier
    classifier = LLMPaperClassifier(model=model, temperature=0.0, api_base=api_base)
    
    batch_size = 20
    classifications = []
    logger.info(f"Processing {len(df)} papers in batches of {batch_size}")
    for batch_start in tqdm(range(0, len(df), batch_size), desc="Batching LLM calls"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        prompts = []
        arxiv_ids = []
        rows = []
        for _, row in batch_df.iterrows():
            arxiv_id = row['arxiv_id']
            title = str(row.get('title', ''))
            abstract = str(row.get('abstract', ''))
            prompts.append(classifier._build_user_prompt(title, abstract))
            arxiv_ids.append(arxiv_id)
            rows.append(row)
        # Build messages for batch_completion
        if classifier.use_vllm:
            messages = [[{"role": "user", "content": classifier.system_prompt + "\n\n" + prompt}] for prompt in prompts]
        else:
            messages = [[{"role": "system", "content": classifier.system_prompt}, {"role": "user", "content": prompt}] for prompt in prompts]
        batch_kwargs = {
            "model": classifier.model,
            "messages": messages,
            "temperature": classifier.temperature,
            "max_tokens": 200
        }
        if classifier.api_base:
            batch_kwargs["api_base"] = classifier.api_base
        try:
            responses = batch_completion(**batch_kwargs)
        except Exception as e:
            logger.error(f"Batch LLM call failed: {e}")
            responses = [None] * len(messages)
        for i, response in enumerate(responses):
            row = rows[i]
            arxiv_id = arxiv_ids[i]
            if response is None:
                classification = {
                    'arxiv_id': arxiv_id,
                    'predicted_type': 'other',
                    'reasoning': 'Batch LLM error'
                }
                classifications.append(classification)
                continue
            try:
                content = response.choices[0].message.content.strip()
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join([l for l in lines if l and not l.startswith("```")])
                result = json.loads(content)
                classification_label = result.get("classification", "other").lower()
                predicted_type = 'review' if classification_label == 'review' else 'regular'
                classification = {
                    'arxiv_id': arxiv_id,
                    'predicted_type': predicted_type,
                    'reasoning': result.get('reasoning', '')
                }
            except Exception as e:
                classification = {
                    'arxiv_id': arxiv_id,
                    'predicted_type': 'other',
                    'reasoning': f'LLM parse error: {str(e)}'
                }
            classifications.append(classification)
    df_class = pd.DataFrame(classifications)
    df_class.to_csv(processed_dir / 'paper_classifications.csv', index=False)
    
    # Validation: check agreement with original labels if available
    if 'paper_type' in df.columns:
        df_class_merged = pd.merge(df[['arxiv_id', 'paper_type']], df_class, on='arxiv_id')
        known_papers = df_class_merged['paper_type'] != 'unknown'
        if known_papers.sum() > 0:
            agreement = (df_class_merged.loc[known_papers, 'paper_type'] == 
                        df_class_merged.loc[known_papers, 'predicted_type']).mean()
            logger.info(f"Agreement with original labels: {agreement:.2%}")
        else:
            logger.info("No pre-labeled papers to validate against")
    else:
        logger.info("No original paper_type labels available")
    
    logger.info(f"LLM classification complete: {len(df_class)} papers")
    logger.info(f"Predicted distribution: {df_class['predicted_type'].value_counts().to_dict()}")
    
    return df_class


def main():
    """Main function for running evaluation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LLM-based paper classifier")
    parser.add_argument(
        '--eval-csv',
        type=str,
        default='data/eval/maria-eval.csv',
        help='Path to evaluation CSV file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='LiteLLM model name (e.g., gpt-4o-mini, claude-3-haiku-20240307)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for generation (0 = deterministic)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/results/llm_classification_results.csv',
        help='Path to save detailed results'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run evaluation
    metrics = evaluate_llm_classifier(
        eval_csv_path=args.eval_csv,
        model=args.model,
        temperature=args.temperature,
        output_path=args.output
    )
    
    return metrics


if __name__ == "__main__":
    main()
