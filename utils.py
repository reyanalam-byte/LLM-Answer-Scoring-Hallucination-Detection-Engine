import re
import numpy as np

# Strongly confident words
CONFIDENT_WORDS = [
    "definitely", "certainly", "always",
    "never", "guaranteed", "undoubtedly",
    "clearly", "without doubt"
]

# Hedging words that reduce overconfidence
HEDGE_WORDS = [
    "maybe", "perhaps", "possibly",
    "might", "could", "i think", "i guess",
    "likely", "seems", "suggests"
]

def overconfidence_penalty(text: str) -> float:
    """
    Context-aware overconfidence penalty.

    - Counts confident words, reduces penalty with hedging words.
    - Scales by answer length for smooth normalization.
    - Returns a value between 0 and 1.
    """
    text_lower = text.lower()
    
    confident_count = sum(bool(re.search(rf'\b{re.escape(word)}\b', text_lower)) 
                          for word in CONFIDENT_WORDS)
    
    hedge_count = sum(bool(re.search(rf'\b{re.escape(word)}\b', text_lower)) 
                      for word in HEDGE_WORDS)
    
    base_penalty = 0.15
    num_words = max(len(text_lower.split()), 1)
    
    penalty = (confident_count * base_penalty) / np.log1p(num_words)
    
    hedge_factor = 1 - min(hedge_count * 0.1, 0.5)
    penalty *= hedge_factor
    
    return min(max(penalty, 0.0), 1.0)
