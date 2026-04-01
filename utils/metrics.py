import math
from nltk.translate.bleu_score import corpus_bleu

def calculate_perplexity(loss):
    """Calculates perplexity or loss for text generation."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')

def calculate_bleu(candidate_corpus, reference_corpus):
    """
    Calculates BLEU score for machine translation using NLTK.
    candidate_corpus: list of lists of tokens (e.g., [['the', 'cat', 'sat'], ...])
    reference_corpus: list of lists of lists of tokens (e.g., [[['the', 'cat', 'sat on', 'mat']], ...])
    """
    return corpus_bleu(reference_corpus, candidate_corpus)