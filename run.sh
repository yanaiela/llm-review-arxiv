CUDA_VISIBLE_DEVICES=1 uv run python -m src.main --llm-model vllm/google/gemma-2-2b-it --llm-api-base http://localhost:8000/v1 --stage all --categories cs
CUDA_VISIBLE_DEVICES=1 uv run python -m src.main --llm-model vllm/google/gemma-2-2b-it --llm-api-base http://localhost:8000/v1 --stage all --categories math
CUDA_VISIBLE_DEVICES=1 uv run python -m src.main --llm-model vllm/google/gemma-2-2b-it --llm-api-base http://localhost:8000/v1 --stage all --categories stat
CUDA_VISIBLE_DEVICES=1 uv run python -m src.main --llm-model vllm/google/gemma-2-2b-it --llm-api-base http://localhost:8000/v1 --stage all --categories physics
