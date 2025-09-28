import asyncio
import os
import re
import threading

import numpy as np
import torch
from datasets import load_dataset
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

import verifiers as vf

from .custom_parser import CustomParser
from .utils import (
    extract_rust_code_from_state,
    get_code_from_applied_comments,
    run_cargo_command,
    setup_client,
)

ONNX_EMBED_MODEL = "EmbeddedLLM/all-MiniLM-L6-v2-onnx-o3-cpu"
ONNX_PROVIDER = "CPUExecutionProvider"

SYSTEM_PROMPT = """
You are an expert code reviewer. You will be given code to review and should provide constructive feedback.

Format your response as follows:
{format_str}

Focus on the most important issues first. Be constructive and educational.
Ground every comment in concrete code snippets from the actual submission.
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

    def _normalize_comments(raw_comments):
        return [
            str(comment).strip()
            for comment in (raw_comments or [])
            if isinstance(comment, str) and str(comment).strip()
        ]

    def _diagnostic_key(diag: str) -> str:
        return re.sub(r"\s+", " ", str(diag).strip()).lower()

    def _score_diagnostics(baseline: list[str], current: list[str]) -> float:
        baseline_set = {key for key in (_diagnostic_key(d) for d in baseline) if key}
        current_set = {key for key in (_diagnostic_key(d) for d in current) if key}

        if not baseline_set and not current_set:
            return 1.0
        if not baseline_set:
            # Penalize new diagnostics proportional to their count
            return max(0.0, 1.0 - 0.25 * len(current_set))

        resolved = baseline_set - current_set
        remaining = baseline_set & current_set
        new_issues = current_set - baseline_set

        resolved_ratio = len(resolved) / max(len(baseline_set), 1)
        remaining_ratio = len(remaining) / max(len(baseline_set), 1)
        new_ratio = len(new_issues) / max(len(current_set), 1) if current_set else 0.0

        reward = resolved_ratio
        reward -= 0.6 * remaining_ratio
        reward -= 0.9 * new_ratio
        return float(max(0.0, min(1.0, reward)))

    async def _get_baseline_diagnostics(command: str, state) -> list[str]:
        cache = state.setdefault("_cargo_baseline", {})
        if command in cache:
            return cache[command]

        original_code = state.get("original_code") or extract_rust_code_from_state(state)
        if not original_code:
            cache[command] = []
            return cache[command]

        import asyncio as _asyncio

        _, diagnostics = await _asyncio.to_thread(run_cargo_command, command, original_code)
        cache[command] = diagnostics
        return diagnostics

    async def _encode_texts(texts: list[str]) -> np.ndarray | None:
        if not texts:
            return None
        return await _safe_encode(texts)

    def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size == 0 or b.size == 0:
            return np.zeros((len(a), len(b)), dtype=np.float32)
        return np.asarray(a @ b.T, dtype=np.float32)

    def _match_comments(sim_matrix: np.ndarray, threshold: float = 0.65) -> set[tuple[int, int]]:
        matches = set()
        if sim_matrix.size == 0:
            return matches

        used_rows: set[int] = set()
        used_cols: set[int] = set()

        flat_indices = np.argsort(sim_matrix, axis=None)[::-1]
        rows, cols = sim_matrix.shape
        for idx in flat_indices:
            r = idx // cols
            c = idx % cols
            if r in used_rows or c in used_cols:
                continue
            if sim_matrix[r, c] < threshold:
                break
            matches.add((r, c))
            used_rows.add(r)
            used_cols.add(c)
        return matches

    def _extract_identifiers(comment: str) -> set[str]:
        code_spans = re.findall(r"`([^`]+)`", comment)
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", comment)
        return {token for token in (*code_spans, *words) if len(token) > 2}

    def _ground_comment(comment: str, original_code: str) -> bool:
        if not original_code:
            return False
        identifiers = _extract_identifiers(comment)
        if not identifiers:
            # fallback: ensure comment length and mentions exist
            return True
        lowered_code = original_code.lower()
        for identifier in identifiers:
            if identifier.lower() in lowered_code:
                return True
        return False

    async def comment_alignment_reward(completion, **kwargs):
        state = kwargs["state"]
        gold_comments = _normalize_comments(state.get("info", {}).get("gold_comments", []))
        pred_comments = _normalize_comments(parser.parse_answer(completion))

        if not gold_comments and not pred_comments:
            return 1.0
        if not pred_comments:
            return 0.0

        gold_emb = await _encode_texts(gold_comments)
        pred_emb = await _encode_texts(pred_comments)

        if gold_emb is None or pred_emb is None:
            return 0.0

        sim_matrix = _cosine_similarity_matrix(pred_emb, gold_emb)
        matches = _match_comments(sim_matrix)

        original_code = extract_rust_code_from_state(state) or ""

        grounded_matches = set()
        for pred_idx, gold_idx in matches:
            comment = pred_comments[pred_idx]
            if _ground_comment(comment, original_code):
                grounded_matches.add((pred_idx, gold_idx))

        true_positives = len(grounded_matches)
        false_positives = len(pred_comments) - true_positives

        if true_positives == 0:
            penalty = max(0.0, false_positives / max(len(pred_comments), 1))
            return max(0.0, 0.0 - 0.2 * penalty)

        precision = true_positives / len(pred_comments)
        recall = true_positives / max(len(gold_comments), 1)
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        penalty = 0.2 * (false_positives / max(len(pred_comments), 1))
        return max(0.0, f1 - penalty)

    _st_lock = threading.Lock()
    _st_model = {"model": None}
    _encode_lock_holder = {"lock": None}
    _encode_lock_init_lock = threading.Lock()

    _encoder_lock = threading.Lock()
    _encoder_model = {"model": None}
    _encoder_tokenizer = {"tokenizer": None}

    def _get_st_model():
        # force_cpu kept for API compatibility, ONNX model is CPU-only here
        with _encoder_lock:
            if _encoder_model["model"] is None:
                _encoder_tokenizer["tokenizer"] = AutoTokenizer.from_pretrained(ONNX_EMBED_MODEL)
                _encoder_model["model"] = ORTModelForFeatureExtraction.from_pretrained(
                    ONNX_EMBED_MODEL,
                    provider=ONNX_PROVIDER,
                )
            return _encoder_model["model"], _encoder_tokenizer["tokenizer"]

    def _log_encode_failure(stage: str, exc: Exception):
        print(f"[_safe_encode] {stage} failed with {exc}")

    def _get_encode_lock():
        with _encode_lock_init_lock:
            if _encode_lock_holder["lock"] is None:
                _encode_lock_holder["lock"] = asyncio.Lock()
            return _encode_lock_holder["lock"]

    async def _safe_encode(texts):
        async def _encode(model, tokenizer):
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="np",
                max_length=getattr(model.config, "max_position_embeddings", 512),
            )
            # Run ONNX model in background thread
            outputs = await asyncio.to_thread(model, **inputs)
            token_embeddings = outputs.last_hidden_state
            # Attention mask for pooling
            att_mask = inputs["attention_mask"][..., None].astype(np.float32)
            pooled = (token_embeddings * att_mask).sum(axis=1) / np.clip(att_mask.sum(axis=1), 1e-9, None)
            # Normalize to unit length
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            normalized = pooled / np.clip(norms, 1e-9, None)
            return normalized.astype(np.float32)

        model, tokenizer = _get_st_model()
        return await _encode(model, tokenizer)

    def _simple_ngrams(tokens, n):
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _tokenize_rust_code(code: str) -> list[str]:
        """Tokenize Rust source for CrystalBLEU comparisons."""
        code = re.sub(r"//.*?\n", " ", code)
        code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
        code = re.sub(r'"[^"]*"', "STRING", code)
        code = re.sub(r"'[^']*'", "CHAR", code)
        tokens = re.findall(r"\w+|[{}();,\.\[\]<>!=&|+-/*%^~]", code)
        return [token.lower() for token in tokens if token.strip()]

    def _to_builtin(obj):
        """Recursively convert tensors/arrays to plain Python types."""
        if torch is not None and isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu()
            return obj.item() if obj.ndim == 0 else obj.tolist()
        if torch is not None and isinstance(obj, torch.Size):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return obj.item() if obj.ndim == 0 else obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {key: _to_builtin(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [_to_builtin(value) for value in obj]
        if isinstance(obj, tuple):
            return tuple(_to_builtin(value) for value in obj)
        return obj

    def sanitize_for_broadcast(obj):
        return _to_builtin(obj)

    async def semantic_similarity_reward(completion, **kwargs) -> int | float:
        """
        Calculate semantic similarity between predicted and gold comments using sentence embeddings.
        """
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

        sim = np.asarray(pred_emb @ gold_emb.T, dtype=np.float32)

        precision = float(sim.max(axis=1).mean())
        recall = float(sim.max(axis=0).mean())

        score = (precision + recall) / 2.0
        score = max(0.0, min(1.0, score))
        return score

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

        success, diagnostics = await _asyncio.to_thread(run_cargo_command, "build", refined_code)
        baseline = await _get_baseline_diagnostics("build", state)
        if success and not diagnostics:
            return 1.0
        return _score_diagnostics(baseline, diagnostics)

    async def cargo_test_reward(completion, **kwargs):
        """Reward for tests passing after applying review comments."""
        import asyncio as _asyncio

        state = kwargs["state"]
        refined_code = await get_code_from_applied_comments(
            review_applicator_model, review_applicator_client, completion, state
        )

        if not refined_code:
            return 0.0

        success, diagnostics = await _asyncio.to_thread(run_cargo_command, "test", refined_code)
        baseline = await _get_baseline_diagnostics("test", state)
        if success and not diagnostics:
            return 1.0
        return _score_diagnostics(baseline, diagnostics)

    async def cargo_clippy_reward(completion, **kwargs):
        """Reward for fewer clippy warnings after applying review comments."""
        import asyncio as _asyncio

        state = kwargs["state"]
        refined_code = await get_code_from_applied_comments(
            review_applicator_model, review_applicator_client, completion, state
        )

        if not refined_code:
            return 0.0

        success, diagnostics = await _asyncio.to_thread(run_cargo_command, "clippy", refined_code)
        baseline = await _get_baseline_diagnostics("clippy", state)
        if success and not diagnostics:
            return 1.0
        return _score_diagnostics(baseline, diagnostics)

    rubric = vf.Rubric(
        funcs=[
            crystalbleu_reward,
            cargo_build_reward,
            cargo_test_reward,
            cargo_clippy_reward,
            semantic_similarity_reward,
            comment_alignment_reward,
            parser.get_format_reward_func(),
        ],
        weights=[0.30, 0.15, 0.15, 0.05, 0.15, 0.17, 0.03],
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
