# Rust Code Review

### Overview
- **Environment ID**: `rust-review`
- **Short description**: Single-turn environment where models provide constructive code review feedback on Rust code with issues.
- **Tags**: code, rust, review, single-turn
- **Socials**: [Github @ljt019](https://github.com/ljt019), [Hf @ljt019](https://huggingface.co/ljt019), [X @Ljt019117161](https://x.com/Ljt019117161)

### Datasets
- **Primary dataset(s)**: 
  - [rust-review-coral](https://huggingface.co/datasets/ljt019/rust-review-coral): Rust code review dataset with constructive feedback examples

### Task & Scoring
- **Type**: single-turn code review
- **Parser**: Extracts review comments from structured feedback format
- **Rubric overview**: Weighted scoring based on CrystalBLEU similarity, cargo validation, semantic similarity, and issue coverage

### Quickstart

Run an evaluation with default setting:
```bash
uv run vf-eval rust-review
```

Browse results
```bash
uv run vf-tui
```

## Environment Arguments

| Arg               | Type | Default         | Description                                           |
| ----------------- | ---- | --------------- | ----------------------------------------------------- |
| `coder_model`     | str  | (required)      | Model name for the code reviewer LLM                 |
| `coder_base_url`  | str  | `https://openrouter.ai/api/v1` | Base URL for the coder LLM API                       |
| `coder_api_key`   | str  | `None`          | API key for the coder LLM (uses OPENROUTER_API_KEY env var if None) |

---

## Metrics

| Metric                            | Weight | Meaning                                         |
| --------------------------------- | ------ | ----------------------------------------------- |
| `reward`                          | -      | Final weighted rubric score (0.0 to 1.0)       |
| `crystalbleu_reward`              | 0.35   | CrystalBLEU similarity between refined and gold code |
| `semantic_similarity_reward`      | 0.20   | Semantic similarity between predicted and gold comments |
| `cargo_build_reward`              | 0.15   | Refined code compiles successfully             |
| `cargo_test_reward`               | 0.15   | Refined code passes tests                      |
| `cargo_clippy_reward`             | 0.05   | Refined code passes clippy linting            |
| `minimum_issues_found_reward`     | 0.07   | Appropriate number of issues identified        |
| `format_reward`                   | 0.03   | Review follows proper formatting guidelines    |

---