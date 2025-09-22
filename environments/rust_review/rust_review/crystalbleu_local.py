"""
Minimal CrystalBLEU implementation without external NLTK or fractions.Fraction
dependencies. Provides a corpus_bleu function compatible with the usage in
rust_review.rust_review.crystalbleu_reward, including support for an `ignoring`
set of n-grams.
"""

import math
import sys
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Set, Tuple


def _generate_ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _ngrams_ignoring(
    tokens: Sequence[str], n: int, ignoring: Optional[Iterable[Tuple[str, ...]]] = None
) -> List[Tuple[str, ...]]:
    ngrams = _generate_ngrams(tokens, n)
    if not ignoring:
        return ngrams
    ignoring_set: Set[Tuple[str, ...]] = set(ignoring)
    return [ng for ng in ngrams if ng not in ignoring_set]


def _closest_ref_length(references: List[List[str]], hyp_len: int) -> int:
    # Choose the reference length closest to hyp_len; tie-break by using shorter
    closest = None
    best_key = None
    for ref in references:
        ref_len = len(ref)
        key = (abs(ref_len - hyp_len), ref_len)
        if best_key is None or key < best_key:
            best_key = key
            closest = ref_len
    return closest if closest is not None else 0


def _brevity_penalty(ref_total_len: int, hyp_total_len: int) -> float:
    if hyp_total_len == 0:
        return 0.0
    if hyp_total_len > ref_total_len:
        return 1.0
    return math.exp(1.0 - (ref_total_len / hyp_total_len))


def corpus_bleu(
    list_of_references: List[List[List[str]]],
    hypotheses: List[List[str]],
    weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),
    ignoring: Optional[Iterable[Tuple[str, ...]]] = None,
) -> float:
    """
    Compute corpus-level BLEU/CrystalBLEU with optional ignoring set of n-grams.

    - list_of_references: [[ref1_for_h0, ref2_for_h0, ...], [ref1_for_h1, ...], ...]
      where each ref is a token list (List[str]).
    - hypotheses: [hyp0_tokens, hyp1_tokens, ...]
    - weights: weights for n-gram orders (default BLEU-4 uniform)
    - ignoring: iterable of n-gram tuples to be excluded when counting

    Returns a float in [0, 1].
    """
    if len(list_of_references) != len(hypotheses):
        raise ValueError("Number of hypotheses and reference sets must match")

    # Accumulate corpus-level numerators/denominators for modified precision
    p_numerators: Counter[int] = Counter()
    p_denominators: Counter[int] = Counter()

    hyp_total_len = 0
    ref_total_len = 0

    max_order = len(weights)

    for references, hypothesis in zip(list_of_references, hypotheses):
        hyp_len = len(hypothesis)
        hyp_total_len += hyp_len
        ref_total_len += _closest_ref_length(references, hyp_len)

        for n in range(1, max_order + 1):
            hyp_counts = Counter(_ngrams_ignoring(hypothesis, n, ignoring))
            if not hyp_counts:
                # denominator at least 1 (to avoid division by zero later)
                p_denominators[n] += 1
                continue

            max_ref_counts: Counter[Tuple[str, ...]] = Counter()
            for ref in references:
                ref_counts = Counter(_ngrams_ignoring(ref, n, ignoring))
                # take elementwise max across references
                for ng, cnt in ref_counts.items():
                    if cnt > max_ref_counts[ng]:
                        max_ref_counts[ng] = cnt

            # clipped counts
            overlap = 0
            for ng, cnt in hyp_counts.items():
                overlap += min(cnt, max_ref_counts.get(ng, 0))

            p_numerators[n] += int(overlap)
            p_denominators[n] += int(sum(hyp_counts.values()))

    # If no unigram overlap at all, return 0 directly (as in NLTK's behavior)
    if p_numerators[1] == 0:
        return 0.0

    # Compute precision terms with simple smoothing for zero counts
    precisions: List[float] = []
    for n in range(1, max_order + 1):
        denom = p_denominators[n]
        if denom <= 0:
            precisions.append(sys.float_info.min)
            continue
        num = p_numerators[n]
        if num == 0:
            # method0-style minimal positive to avoid log(0)
            precisions.append(sys.float_info.min)
        else:
            precisions.append(num / denom)

    # Brevity penalty and geometric mean of precisions
    bp = _brevity_penalty(ref_total_len, hyp_total_len)
    s = 0.0
    for w, p in zip(weights, precisions):
        # guard against p <= 0 due to underflow
        p_safe = p if p > 0.0 else sys.float_info.min
        s += w * math.log(p_safe)
    score = bp * math.exp(s)
    # Ensure numeric stability
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)
