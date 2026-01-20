# AI detection module

from .detect_ai_content import detect_ai_content
from .pangram_detector import detect_pangram_content, PangramDetector

__all__ = ['detect_ai_content', 'detect_pangram_content', 'PangramDetector']
