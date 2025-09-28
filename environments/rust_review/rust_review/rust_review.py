import asyncio
import os
import re
import threading

import numpy as np
from datasets import load_dataset

import verifiers as vf

from .custom_parser import CustomParser
from .utils import get_code_from_applied_comments, run_cargo_command, setup_client

SYSTEM_PROMPT = """
You are an expert code reviewer. You will be given code to review and should provide constructive feedback.

Format your response as follows:
{format_str}

Focus on the most important issues first. Be constructive and educational.
"""

DATASET_NAME_DEFAULT = "ljt019/rust-review-singleturn-3250"


def load_environment(
    review_applicator_model: str,
    review_applicator_base_url: str = "https://openrouter.ai/api/v1",
    review_applicator_api_key: str | None = None,
    dataset_name: str = DATASET_NAME_DEFAULT,
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
        """Reward 1.0 when at least the expected number of issues are flagged."""
        state = kwargs["state"]
        gold_comments = state.get("info", {}).get("gold_comments", [])
        comment_count = len(parser.parse_answer(completion) or [])
        expected_issues = len(gold_comments)

        if expected_issues == 0:
            return 1.0 if comment_count == 0 else 0.0
        return 1.0 if comment_count >= expected_issues else 0.0

    _st_lock = threading.Lock()
    _st_model = {"model": None}
    _encode_lock_holder = {"lock": None}
    _encode_lock_init_lock = threading.Lock()

    def _get_st_model(force_cpu: bool = False):
        from sentence_transformers import SentenceTransformer

        with _st_lock:
            if force_cpu or _st_model["model"] is None:
                _st_model["model"] = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            return _st_model["model"]

    def _log_encode_failure(stage: str, exc: Exception):
        print(f"[_safe_encode] {stage} failed with {exc}")

    def _get_encode_lock():
        with _encode_lock_init_lock:
            if _encode_lock_holder["lock"] is None:
                _encode_lock_holder["lock"] = asyncio.Lock()
            return _encode_lock_holder["lock"]

    async def _safe_encode(texts):
        async def _encode(model):
            emb = await asyncio.to_thread(
                model.encode,
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=False,
            )
            return np.atleast_2d(emb)

        model = _get_st_model()
        encode_lock = _get_encode_lock()
        async with encode_lock:
            try:
                return await _encode(model)
            except Exception as exc:
                _log_encode_failure("initial encode", exc)
                model = _get_st_model(force_cpu=True)
                try:
                    return await _encode(model)
                except Exception as exc_retry:
                    _log_encode_failure("retry", exc_retry)
                    return None

    def _normalize_comments(raw_comments):
        return [
            str(comment).strip()
            for comment in (raw_comments or [])
            if isinstance(comment, str) and str(comment).strip()
        ]

    def _simple_ngrams(tokens, n):
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _tokenize_rust_code(code):
        """Tokenize Rust code for CrystalBLEU comparisons."""
        code = re.sub(r"//.*?\n", " ", code)
        code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
        code = re.sub(r'"[^"]*"', "STRING", code)
        code = re.sub(r"'[^']*'", "CHAR", code)

        tokens = re.findall(r"\w+|[{}();,\.\[\]<>!=&|+-/*%^~]", code)
        return [token.lower() for token in tokens if token.strip()]

    async def semantic_similarity_reward(completion, **kwargs):
        state = kwargs["state"]
        pred_comments = _normalize_comments(parser.parse_answer(completion))
        gold_comments = _normalize_comments(state.get("info", {}).get("gold_comments", []))

        if not pred_comments and not gold_comments:
            return 1.0  # both are empty (code doesn't need review)

        if not pred_comments or not gold_comments:
            return 0.0  # if one is empty and other is not, can't be similar

        pred_emb = await _safe_encode(pred_comments)
        gold_emb = await _safe_encode(gold_comments)

        if pred_emb is None or gold_emb is None:
            print("[semantic_similarity_reward] returning 0.0 (encode failed)")
            return 0.0

        sim = pred_emb @ gold_emb.T

        precision = float(sim.max(axis=1).mean())
        recall = float(sim.max(axis=0).mean())

        score = (precision + recall) / 2.0
        return max(0.0, min(1.0, score))

    async def crystalbleu_reward(completion, **kwargs):
        """Compare refined code with ground truth using CrystalBLEU."""
        from collections import Counter

        from .crystalbleu_local import corpus_bleu

        state = kwargs["state"]

        refined_code = await get_code_from_applied_comments(
            review_applicator_model, review_applicator_client, completion, state
        )
        gold_code = state.get("info", {}).get("gold_code", "")

        if not refined_code:
            print("[crystalbleu_reward] returning 0.0 (no refined code)")
            return 0.0

        refined_tokens = _tokenize_rust_code(refined_code)
        gold_tokens = _tokenize_rust_code(gold_code)

        if not refined_tokens or not gold_tokens:
            print("[crystalbleu_reward] returning 0.0 (no refined or gold tokens)")
            return 0.0

        background_tokens = gold_tokens

        k = min(500, max(50, len(background_tokens) // 4))
        all_ngrams = []
        for n in range(1, 5):  # 1-grams to 4-grams
            all_ngrams.extend(_simple_ngrams(background_tokens, n))

        frequencies = Counter(all_ngrams)
        trivially_shared_ngrams = {ng for ng, _ in frequencies.most_common(k)}

        references = [[gold_tokens]]
        candidates = [refined_tokens]

        crystalbleu_score = corpus_bleu(references, candidates, ignoring=trivially_shared_ngrams)
        result = float(crystalbleu_score)
        return result

    async def cargo_build_reward(completion, **kwargs):
        """Reward for successful compilation after applying review comments."""
        import asyncio as _asyncio

        state = kwargs["state"]
        refined_code = await get_code_from_applied_comments(
            review_applicator_model, review_applicator_client, completion, state
        )

        if not refined_code:
            return 0.0

        success = await _asyncio.to_thread(run_cargo_command, "build", refined_code)
        return 1.0 if success else 0.0

    async def cargo_test_reward(completion, **kwargs):
        """Reward for tests passing after applying review comments."""
        import asyncio as _asyncio

        state = kwargs["state"]
        refined_code = await get_code_from_applied_comments(
            review_applicator_model, review_applicator_client, completion, state
        )

        if not refined_code:
            return 0.0

        success = await _asyncio.to_thread(run_cargo_command, "test", refined_code)
        return 1.0 if success else 0.0

    async def cargo_clippy_reward(completion, **kwargs):
        """Reward for fewer clippy warnings after applying review comments."""
        import asyncio as _asyncio

        state = kwargs["state"]
        refined_code = await get_code_from_applied_comments(
            review_applicator_model, review_applicator_client, completion, state
        )

        if not refined_code:
            return 0.0

        success = await _asyncio.to_thread(run_cargo_command, "clippy", refined_code)
        return 1.0 if success else 0.0

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
