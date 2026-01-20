# LLM-Based Paper Classification

This script uses LiteLLM to classify papers as review/survey vs. other types using large language models.

## Installation

```bash
pip install litellm scikit-learn
```

## Usage

### Basic Usage

```bash
python src/classification/llm_classifier.py
```

### Custom Model

```bash
# Using GPT-4o-mini (default)
python src/classification/llm_classifier.py --model gpt-4o-mini

# Using Claude
python src/classification/llm_classifier.py --model claude-3-haiku-20240307

# Using local model via Ollama
python src/classification/llm_classifier.py --model ollama/llama2
```

### Custom Parameters

```bash
python src/classification/llm_classifier.py \
    --eval-csv data/eval/maria-eval.csv \
    --model gpt-4o-mini \
    --temperature 0.0 \
    --output data/results/llm_classification_results.csv
```

## Environment Variables

Set your API keys as environment variables:

```bash
# For OpenAI models
export OPENAI_API_KEY="your-key-here"

# For Anthropic models
export ANTHROPIC_API_KEY="your-key-here"
```

## Output

The script generates:
1. **Console output**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
2. **Results CSV**: Detailed predictions for each paper
3. **Metrics JSON**: All evaluation metrics in JSON format

## Evaluation Data Format

The evaluation CSV should have these columns:
- `arxiv_id`: Paper identifier
- `title`: Paper title
- `abstract`: Paper abstract
- `MARIA_ANNOTATION`: Ground truth label (1 = review/survey, 0 = other)

## Example Output

```
================================================================================
EVALUATION RESULTS
================================================================================
Model: gpt-4o-mini
Total papers: 60

Accuracy:  0.950
Precision: 0.978
Recall:    0.957
F1 Score:  0.967

Confusion Matrix:
                Predicted
              Other  Review
Actual Other     13      1
       Review      2     44

              precision    recall  f1-score   support

       Other       0.87      0.93      0.90        14
      Review       0.98      0.96      0.97        46

    accuracy                           0.95        60
   macro avg       0.92      0.94      0.93        60
weighted avg       0.95      0.95      0.95        60
================================================================================
```

## Python API

You can also use the classifier programmatically:

```python
from src.classification.llm_classifier import LLMPaperClassifier, evaluate_llm_classifier

# Classify a single paper
classifier = LLMPaperClassifier(model="gpt-4o-mini", temperature=0.0)
result = classifier.classify_paper(
    title="A Survey of Deep Learning",
    abstract="This paper reviews recent advances in deep learning..."
)
print(result)  # {'classification': 'review', 'confidence': 0.95, 'reasoning': '...'}

# Run full evaluation
metrics = evaluate_llm_classifier(
    eval_csv_path="data/eval/maria-eval.csv",
    model="gpt-4o-mini",
    output_path="data/results/llm_results.csv"
)
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

## Supported Models

LiteLLM supports 100+ models. Common options:

**OpenAI:**
- `gpt-4o-mini` (recommended, cost-effective)
- `gpt-4o`
- `gpt-3.5-turbo`

**Anthropic:**
- `claude-3-haiku-20240307`
- `claude-3-sonnet-20240229`
- `claude-3-opus-20240229`

**Local (via Ollama):**
- `ollama/llama2`
- `ollama/mistral`

See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for full list.
