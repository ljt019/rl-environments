import os
import threading

import verifiers as vf
from datasets import load_dataset

from .custom_parser import CustomParser
from .utils import (
    get_code_from_applied_comments,
    run_cargo_build,
    run_cargo_clippy,
    run_cargo_tests,
    setup_client,
)

SYSTEM_PROMPT = """
You are an expert code reviewer. You will be given code to review and should provide constructive feedback.

Format your response as follows:
{format_str}

Focus on the most important issues first. Be constructive and educational.
"""


def load_environment(
    review_applicator_model: str,
    review_applicator_base_url: str = "https://openrouter.ai/api/v1",
    review_applicator_api_key: str | None = None,
    dataset_name: str = "ljt019/rust-review-singleturn-3250",
) -> vf.SingleTurnEnv:
    """
    Load the rust code review environment.

    Args:
        review_applicator_model: Model name for the review applicator LLM
        review_applicator_base_url: Base URL for the review applicator LLM API (defaults to OpenRouter)
        review_applicator_api_key: API key for the review applicator LLM (if None, uses OPENROUTER_API_KEY env var)

    Returns:
        SingleTurnEnv: The configured rust review environment

    Environment Variables:
        OPENROUTER_API_KEY: API key for the review applicator LLM (used if review_applicator_api_key is None)
    """
    api_key = review_applicator_api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "API key is required for the review applicator LLM. "
            "Either provide review_applicator_api_key parameter or set OPENROUTER_API_KEY environment variable."
        )

    review_applicator_client = setup_client(
        api_base_url=review_applicator_base_url,
        api_key=api_key,
        timeout=600.0,
        max_connections=100,
        max_keepalive_connections=50,
        max_retries=3,
    )

    dataset = load_dataset(dataset_name)

    parser = CustomParser()

    def minimum_issues_found_reward(completion, **kwargs):
        print("[RUST_REVIEW] minimum_issues_found_reward: start")
        """
        Counts the number of issues in 'gold_comments' in info of state

        - If no issues are expected (zero gold_comments), reward 1.0 only if the model returns zero comments.
        - Otherwise, reward 1.0 if the number of comments >= expected issues; else 0.0.
        """
        state = kwargs["state"]
        print("[RUST_REVIEW] minimum_issues_found_reward: parsed state")
        gold_comments = state.get("info", {}).get("gold_comments", [])

        # Use parser to extract comments consistently
        comments = parser.parse_answer(completion)
        print(f"[RUST_REVIEW] minimum_issues_found_reward: parsed comments={len(comments)} expected={len(gold_comments)}")
        expected_issues = len(gold_comments)

        if expected_issues == 0:
            result = 1.0 if len(comments) == 0 else 0.0
            print(f"[RUST_REVIEW] minimum_issues_found_reward: returning {result}")
            return result
        result = 1.0 if len(comments) >= expected_issues else 0.0
        print(f"[RUST_REVIEW] minimum_issues_found_reward: returning {result}")
        return result

    _st_lock = threading.Lock()
    _st_model = {"model": None}

    def _get_st_model():
        import torch
        from sentence_transformers import SentenceTransformer

        with _st_lock:
            if _st_model["model"] is None:
                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
                )
                model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
                _st_model["model"] = model
            return _st_model["model"]

    def _safe_encode(texts):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        def _encode_current(t):
            model = _get_st_model()
            emb = model.encode(
                t,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=False,
            )
            return np.atleast_2d(emb)

        try:
            return _encode_current(texts)
        except Exception:
            # Fallback: reinitialize on CPU and retry once
            with _st_lock:
                _st_model["model"] = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            try:
                return _encode_current(texts)
            except Exception:
                return None

    def semantic_similarity_reward(completion, **kwargs):
        print("[RUST_REVIEW] semantic_similarity_reward: start")
        """
        Returns semantic similarity between predicted and gold comments as a scalar in [0, 1].
        Uses symmetric max-cosine matching (average of per-side maxima).
        """
        import numpy as np

        state = kwargs["state"]
        pred_comments_raw = parser.parse_answer(completion)
        print(f"[RUST_REVIEW] semantic_similarity_reward: parsed pred={pred_comments_raw}")
        gold_comments_raw = state.get("info", {}).get("gold_comments", [])
        print(f"[RUST_REVIEW] semantic_similarity_reward: gold={gold_comments_raw}")

        # Clean and validate comments
        pred_comments = [str(c).strip() for c in (pred_comments_raw or []) if isinstance(c, str) and str(c).strip()]
        gold_comments = [str(c).strip() for c in (gold_comments_raw or []) if isinstance(c, str) and str(c).strip()]

        # If both are empty, perfect agreement
        if not pred_comments and not gold_comments:
            print("[RUST_REVIEW] semantic_similarity_reward: returning 1.0 (both empty)")
            return 1.0
        # If only one side is empty, no semantic agreement
        if not pred_comments or not gold_comments:
            print("[RUST_REVIEW] semantic_similarity_reward: returning 0.0 (one empty)")
            return 0.0

        pred_emb = _safe_encode(pred_comments)
        gold_emb = _safe_encode(gold_comments)
        if pred_emb is None or gold_emb is None:
            print("[RUST_REVIEW] semantic_similarity_reward: encode failed -> 0.0")
            return 0.0

        # Ensure 2D arrays
        pred_emb = np.atleast_2d(pred_emb)
        gold_emb = np.atleast_2d(gold_emb)

        # Cosine similarity via dot product of normalized embeddings
        sim = pred_emb @ gold_emb.T  # shape: [num_pred, num_gold]

        # Symmetric max matching: precision and recall components
        precision = float(sim.max(axis=1).mean())  # each pred matched to best gold
        recall = float(sim.max(axis=0).mean())  # each gold matched to best pred
        score = (precision + recall) / 2.0
        print(f"[RUST_REVIEW] semantic_similarity_reward: returning {score}")
        # Bound to [0, 1]
        return max(0.0, min(1.0, score))

    def crystalbleu_reward(completion, **kwargs):
        print("[RUST_REVIEW] crystalbleu_reward: start")
        """CoRAL-style reward: Compare refined code with ground truth using CrystalBLEU"""
        import re
        from collections import Counter

        from .crystalbleu_local import corpus_bleu

        state = kwargs["state"]
        refined_code = get_code_from_applied_comments(
            review_applicator_model, review_applicator_client, completion, state
        )
        print("[RUST_REVIEW] crystalbleu_reward: got refined_code" if refined_code else "[RUST_REVIEW] crystalbleu_reward: no refined_code")
        if not refined_code:
            # If no refined code (should be rare now), return 0.0 instead of raising
            print("[RUST_REVIEW] crystalbleu_reward: returning 0.0 (no refined code)")
            return 0.0

        gold_code = state.get("info", {}).get("gold_code", "")
        if not gold_code:
            print("[RUST_REVIEW] crystalbleu_reward: missing gold_code")
            raise ValueError("Missing gold_code in state.info for CrystalBLEU computation.")

        def simple_ngrams(tokens, n):
            """Simple n-gram implementation without NLTK dependency"""
            return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

        def tokenize_rust_code(code):
            """Simple Rust code tokenization"""
            # Remove comments and strings for cleaner tokenization
            code = re.sub(r"//.*?\n", " ", code)  # Remove line comments
            code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)  # Remove block comments
            code = re.sub(r'"[^"]*"', "STRING", code)  # Replace string literals
            code = re.sub(r"'[^']*'", "CHAR", code)  # Replace char literals

            # Tokenize by splitting on whitespace and common delimiters
            tokens = re.findall(r"\w+|[{}();,.\[\]<>!=&|+-/*%^~]", code)
            return [token.lower() for token in tokens if token.strip()]

        # Tokenize both codes
        refined_tokens = tokenize_rust_code(refined_code)
        gold_tokens = tokenize_rust_code(gold_code)

        if not refined_tokens or not gold_tokens:
            return 0.0

        # Build ignoring set from gold/reference code only (Rust-only background)
        # Avoids leakage from the candidate into the ignoring set
        background_tokens = gold_tokens
        # Adaptive k with sane bounds
        k = min(500, max(50, len(background_tokens) // 4))
        all_ngrams = []
        for n in range(1, 5):  # 1-grams to 4-grams
            all_ngrams.extend(simple_ngrams(background_tokens, n))

        frequencies = Counter(all_ngrams)
        trivially_shared_ngrams = {ng for ng, _ in frequencies.most_common(k)}

        # Calculate CrystalBLEU following the paper's approach
        # corpus_bleu expects list of list-of-references and list of candidates
        references = [[gold_tokens]]
        candidates = [refined_tokens]

        crystalbleu_score = corpus_bleu(references, candidates, ignoring=trivially_shared_ngrams)
        result = float(crystalbleu_score)
        print(f"[RUST_REVIEW] crystalbleu_reward: returning {result}")
        return result

    def cargo_build_reward(completion, **kwargs):
        print("[RUST_REVIEW] cargo_build_reward: start")
        """Reward for successful compilation after applying review comments"""
        state = kwargs["state"]
        refined_code = get_code_from_applied_comments(
            review_applicator_model, review_applicator_client, completion, state
        )
        print("[RUST_REVIEW] cargo_build_reward: got refined_code" if refined_code else "[RUST_REVIEW] cargo_build_reward: no refined_code")
        if not refined_code:
            return 0.0
        result = 1.0 if run_cargo_build(refined_code) else 0.0
        print(f"[RUST_REVIEW] cargo_build_reward: returning {result}")
        return result

    def cargo_test_reward(completion, **kwargs):
        print("[RUST_REVIEW] cargo_test_reward: start")
        """Reward for tests passing after applying review comments"""
        state = kwargs["state"]
        refined_code = get_code_from_applied_comments(
            review_applicator_model, review_applicator_client, completion, state
        )
        print("[RUST_REVIEW] cargo_test_reward: got refined_code" if refined_code else "[RUST_REVIEW] cargo_test_reward: no refined_code")
        if not refined_code:
            return 0.0
        result = 1.0 if run_cargo_tests(refined_code) else 0.0
        print(f"[RUST_REVIEW] cargo_test_reward: returning {result}")
        return result

    def cargo_clippy_reward(completion, **kwargs):
        print("[RUST_REVIEW] cargo_clippy_reward: start")
        """Reward for fewer clippy warnings after applying review comments"""
        state = kwargs["state"]
        refined_code = get_code_from_applied_comments(
            review_applicator_model, review_applicator_client, completion, state
        )
        print("[RUST_REVIEW] cargo_clippy_reward: got refined_code" if refined_code else "[RUST_REVIEW] cargo_clippy_reward: no refined_code")
        if not refined_code:
            return 0.0
        result = 1.0 if run_cargo_clippy(refined_code) else 0.0
        print(f"[RUST_REVIEW] cargo_clippy_reward: returning {result}")
        return result

    # Reward ordering and weights prioritize correctness-based signals per the paper:
    # - CrystalBLEU (refined vs gold) carries highest weight among correctness metrics
    # - Cargo build/test/clippy ensure functional quality
    # - Semantic similarity supports alignment to gold comments without overfitting wording
    # - Coverage (minimum issues) and formatting have small impact
    rubric = vf.Rubric(
        funcs=[
            crystalbleu_reward,
            cargo_build_reward,
            cargo_test_reward,
            cargo_clippy_reward,
            semantic_similarity_reward,
            minimum_issues_found_reward,
            parser.get_format_reward_func(),
        ],
        weights=[0.35, 0.15, 0.15, 0.05, 0.20, 0.07, 0.03],
        parser=parser,
    )

    vf_env = vf.SingleTurnEnv(
        system_prompt=SYSTEM_PROMPT.format(format_str=parser.get_format_str()),
        dataset=dataset["train"],
        eval_dataset=dataset["test"],
        parser=parser,
        rubric=rubric,
    )

    return vf_env
