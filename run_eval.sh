uv run python src/classification/llm_classifier.py --eval-csv ./data/eval/eval.csv --model ollama/gemma2 --output data/results/classification_results/gemma2_eval.csv
uv run python src/classification/llm_classifier.py --eval-csv ./data/eval/eval.csv --model ollama/llama3.3 --output data/results/classification_results/llama3.3_eval.csv
uv run python src/classification/llm_classifier.py --eval-csv ./data/eval/eval.csv --model ollama/gpt-oss --output data/results/classification_results/gpt-oss_eval.csv
uv run python src/classification/llm_classifier.py --eval-csv ./data/eval/eval.csv --model gpt-4o-mini --output data/results/classification_results/gpt4o-mini_eval.csv

uv run python src/classification/llm_classifier.py --eval-csv ./data/eval/test.csv --model ollama/gemma2 --output data/results/classification_results/gemma2_test.csv
uv run python src/classification/llm_classifier.py --eval-csv ./data/eval/test.csv --model ollama/llama3.3 --output data/results/classification_results/llama3.3_test.csv
uv run python src/classification/llm_classifier.py --eval-csv ./data/eval/test.csv --model ollama/gpt-oss --output data/results/classification_results/gpt-oss_test.csv
uv run python src/classification/llm_classifier.py --eval-csv ./data/eval/test.csv --model gpt-4o-mini --output data/results/classification_results/gpt4o-mini_test.csv