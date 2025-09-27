from __future__ import annotations

"""
Rust Code Review Dataset Generation Pipeline

Generates training data by:
  1. Creating buggy Rust code via LLM
  2. Running cargo diagnostics (build/clippy/test)
  3. Generating review comments from diagnostics
  4. Creating "gold" fixed code via review applicator
  5. Packaging everything into HuggingFace dataset

Architecture:
  • Async pipeline with configurable concurrency
  • Retry logic for LLM failures and cargo timeouts
  • Checkpoint saves every N examples
  • Clean/buggy example ratio balancing
"""

import asyncio
import json
import logging
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Literal, Optional

from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetNotFoundError
from openai import AsyncOpenAI
from rust_review import custom_parser
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS & CONSTANTS
# =============================================================================


CODE_WRITER_SYSTEM_PROMPT = """
You are a junior Rust developer who makes common programming mistakes. Given the following question, write a Rust function that attempts to complete the task but contains bugs. Do NOT write perfect code - you must introduce realistic bugs that cargo tools would catch.

Then write unit tests for the function you defined. Write multiple unit tests for the function. The tests should be a simple line delimited list of assert! or assert_eq! statements. When writing the unit tests you can have comments specifying what you are testing in plain english. The tests should use super::*.

An example output should look like the following:

```rust
/// Find the maximum value in a vector
/// Returns the largest element
fn find_max(numbers: Vec<i32>) -> i32 {
    let mut max_val = 0;  // Bug: should initialize to first element or i32::MIN
    for num in numbers {
        if num >= max_val {  // Bug: should be > for proper comparison
            max_val = num;
        }
    }
    max_val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_max() {
        // Test with positive numbers
        assert_eq!(find_max(vec![1, 5, 3, 9, 2]), 9);
        // Test with negative numbers
        assert_eq!(find_max(vec![-5, -1, -10, -3]), -1);
        // Test with single element
        assert_eq!(find_max(vec![42]), 42);
    }
}
```

CRITICAL: You MUST introduce bugs into your code. Do NOT write perfect code. Your code should always contain bugs and logical issues that cargo clippy, cargo build, or cargo test would catch.

Examples of bugs to include:
- Null pointer dereferences and uninitialized variables
- Buffer overflows and memory leaks
- Race conditions in concurrent code
- Integer overflow and underflow errors
- Logic errors in conditionals and loops

The tests should be correct and would pass if the function were implemented properly.

Make sure to only respond with a single  ```rust``` block. The unit tests must be defined inside the mod tests {} module. Make sure to import any standard library modules that you need. Do not add a main function.
"""


CODE_REVIEWER_SYSTEM_PROMPT = """
You are a thoughtful human code reviewer. You are given Rust code plus internal notes about failing cargo checks. Treat the notes as private hints, not content to repeat. Do NOT reference cargo, clippy, test names, panic traces, or diagnostic IDs. Write review comments exactly as if you noticed the issues by reading the code yourself.

Instructions:
1. For each unique issue hinted at by the notes (and any severe bug you spot), write exactly one concise review comment.
2. Explain the problem in your own words, grounded in code-understandable symptoms. Mention relevant lines or behaviors; do not echo the diagnostic text.
3. Offer actionable advice—briefly describe how to fix or improve the code. Small snippets are fine if they clarify the fix.
4. After covering the hinted issues, take a quick pass for any additional bug you can confidently infer from the code alone. Add an extra comment only if it’s clearly high-value; it’s fine to leave none.
5. Skip nits or subjective style tweaks. If nothing significant is wrong, return an empty review.

Follow this format exactly:
{format_str}

Example 1 (extra bug spotted):
<think>
The hints flag an out-of-bounds access in the loop. While scanning the code I also notice the accumulator never resets, which will corrupt later iterations.
</think>

<review>
<comment>**Bounds check (line 27)**: `values[i + 1]` is read without confirming `i + 1 < values.len()`. On the final iteration this will panic. Guard the access or stop the loop earlier.</comment>
<comment>**Accumulator reset (line 33)**: `sum` carries over between outer iterations. Initialize it to zero inside the loop so each pass works from a clean slate.</comment>
</review>

Example 2 (no additional issues):
<think>
The hint about the unused variable checks out. After reviewing the rest of the function, I don't see any other substantive bugs worth flagging.
</think>

<review>
<comment>**Remove unused variable (line 12)**: `buffer_len` is never read, so it just adds noise. Drop the binding or use it where it adds value.</comment>
</review>

Example 3 (nothing worth flagging):
<think>
The notes are empty. I read through the helper function and nothing jumps out as unsafe or incorrect, so I'll leave the review blank.
</think>

<review>
</review>

Example 4 (two hinted issues only):
<think>
Both hints identify real problems: one is an off-by-one in the loop bounds, the other a panic from `unwrap`. I'll echo those fixes and stop there.
</think>

<review>
<comment>**Loop upper bound (line 18)**: Iterating with `<= items.len()` runs one step past the vector and will panic. Use `< items.len()` so the final index stays in range.</comment>
<comment>**Unchecked unwrap (line 26)**: Calling `value.unwrap()` will panic when the parse fails. Handle the `Err` case explicitly—return an error or fall back to a default instead of unwrapping.</comment>
</review>

Example 5 (extra bug when hints are absent):
<think>
No diagnostics were provided, but while reading I noticed `denominator` can stay zero before division. That's a runtime crash worth surfacing.
</think>

<review>
<comment>**Guard division (line 31)**: `denominator` starts at zero and is never updated when the filter doesn't match, so `total / denominator` will panic. Return early or bail out when the denominator remains zero.</comment>
</review>

Example 6 (rewrite lint-style hints in plain English):
<think>
The notes complain about cloning and needless borrows. They're real inefficiencies, so I'll translate them into a concise performance review.
</think>

<review>
<comment>**Avoid the extra clone (line 14)**: `names.clone()` copies the entire vector even though you only read from it. Iterate over `&names` directly to skip the duplication.</comment>
<comment>**Borrow directly (line 21)**: `path.push_str(&segment)` takes `&&str` before writing. Call `push_str(segment)` so you borrow just once.</comment>
</review>

Example 7 (merge similar hints):
<think>
Two hints point to redundant `clone()` calls on the same data structure. I'll summarize them together so the developer sees it as one cleanup.
</think>

<review>
<comment>**Drop redundant clones (lines 18-22)**: `records.clone()` and `records.clone().iter()` each duplicate the vector even though you only read from it. Iterate over `&records` and reuse the borrowed items instead of cloning twice.</comment>
</review>

Example 8 (actionable fix with snippet):
<think>
The diagnostics note a borrow checker error and a panic from unchecked indexing. Both stem from mutating the vector while iterating. I'll explain the root cause and sketch the safe pattern.
</think>

<review>
<comment>**Mutate after collecting (lines 30-38)**: `for item in items { items.push(process(item)); }` re-borrows `items` mutably while it’s already borrowed for iteration, which triggers the borrow error and can even panic when the length changes mid-loop. Collect the processed values into a temporary vector and extend afterward, e.g.

```rust
let processed: Vec<_> = items.iter().map(process).collect();
items.extend(processed);
```

This keeps the original iteration immutable and avoids the runtime crash.</comment>
</review>
"""


REVIEW_APPLICATOR_SYSTEM_PROMPT = """
You are a code editor that applies ONLY the specific changes mentioned in review comments. You must:

CRITICAL RULES:
1. Apply ONLY the exact changes specified in the review comments
2. Do NOT make any additional improvements, optimizations, or fixes beyond what's explicitly mentioned
3. Do NOT add new functionality, change variable names, or restructure code unless specifically requested
4. Do NOT fix other issues you might notice - only address the exact feedback given
5. If a comment is unclear or impossible to implement, leave that part of the code unchanged

Your job is to be a precise code editor, not a code improver. Apply the minimum changes necessary to address the specific feedback.

Return ONLY the modified Rust code in a single ```rust code block. Do not include explanations.
"""


REVIEW_APPLICATOR_PROMPT = """
Original Code:
```rust
{code}
```

Review Comments (apply ONLY these specific changes):
{comments}

IMPORTANT: Apply only the exact changes mentioned in the review comments above. Do not make any other modifications to the code. Return the minimally modified code in a ```rust block.
"""


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass
class LLMConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    code_writer_model: str = "openai/gpt-4.1-nano"
    code_writer_temperature: float = 0.1
    code_writer_max_tokens: int = 4000
    code_reviewer_model: str = "x-ai/grok-4"
    code_reviewer_temperature: float = 0.1
    code_reviewer_max_tokens: int = 4000
    review_applicator_model: str = "openai/gpt-4.1-nano"
    review_applicator_temperature: float = 0.0
    review_applicator_max_tokens: int = 4000
    llm_retry_attempts: int = 3
    llm_retry_backoff: int = 30
    llm_retry_cooldown: float = 0.5


@dataclass
class DatasetConfig:
    dataset_id: str = "ljt019/rust-17000"
    target_dataset_hub: str = "ljt019/rust-review-hq"
    num_examples: int = 10_000
    no_comment_ratio: float = 0.1


@dataclass
class CargoConfig:
    outputs_base_dir: Path = field(default_factory=lambda: Path("outputs") / "tests")
    cargo_timeout: int = 180
    cargo_test_timeout: int = 60


@dataclass
class RuntimeConfig:
    reprocess_retry_limit: int = 6
    checkpoint_every: int = 50
    enable_timing: bool = True
    timing_log_every: int = 1
    max_concurrent_api_calls: int = 10


# =============================================================================
# DATA STRUCTURES & UTILITIES
# =============================================================================


@dataclass
class ExampleInfo:
    cargo_outputs: list[str]
    gold_comments: list[str]
    gold_code: str
    tests: str

    def to_dict(self) -> dict:
        return {
            "cargo_outputs": self.cargo_outputs,
            "gold_comments": self.gold_comments,
            "gold_code": self.gold_code,
            "tests": self.tests,
        }


@dataclass
class ExampleRecord:
    question: str
    info: ExampleInfo

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "info": self.info.to_dict(),
        }


@dataclass
class CargoDiagnostics:
    messages: list[str]


@dataclass
class PipelineContext:
    llm_config: LLMConfig
    cargo_config: CargoConfig
    runtime_config: RuntimeConfig
    client: AsyncOpenAI
    timer: Optional["TimingCollector"] = None


REVIEW_PROMPT_TEMPLATE = """Please review the following code and provide feedback on any issues you find.

Here's the code for you to review:

```rust
{code}
```
"""


CARGO_COMMENT_PROMPT_TEMPLATE = """```rust
{code}
```

Cargo diagnostics:
{diagnostics}
"""


@dataclass(frozen=True)
class Prompts:
    code_writer_system: str = CODE_WRITER_SYSTEM_PROMPT
    code_reviewer_system: str = CODE_REVIEWER_SYSTEM_PROMPT
    review_applicator_system: str = REVIEW_APPLICATOR_SYSTEM_PROMPT
    review_applicator_prompt_template: str = REVIEW_APPLICATOR_PROMPT
    review_template: str = REVIEW_PROMPT_TEMPLATE
    cargo_comment_template: str = CARGO_COMMENT_PROMPT_TEMPLATE

    def build_review_question(self, code: str) -> str:
        return self.review_template.format(code=code)

    def build_code_reviewer_prompt(self, code: str, cargo_outputs: list[str]) -> str:
        diagnostics_text = "\n".join(f"- {diagnostic}" for diagnostic in cargo_outputs) if cargo_outputs else "- (none)"
        return self.cargo_comment_template.format(code=code, diagnostics=diagnostics_text)

    def build_review_applicator_prompt(self, original_code: str, comments: list[str]) -> str:
        comments_text = "\n".join(f"- {comment}" for comment in comments)
        return self.review_applicator_prompt_template.format(code=original_code, comments=comments_text)


@dataclass
class WorkItem:
    prompt: str
    kind: Literal["dataset", "reprocess"] = "dataset"
    retry_count: int = 0
    code: Optional[str] = None
    tests: Optional[str] = None

    def is_reprocess(self) -> bool:
        return self.kind == "reprocess"


class ExampleQueue:
    def __init__(self, items: Optional[list[WorkItem]] = None) -> None:
        self._queue: Deque[WorkItem] = deque(items or [])

    def push(self, item: WorkItem) -> None:
        self._queue.append(item)

    def pop(self) -> Optional[WorkItem]:
        if not self._queue:
            return None
        return self._queue.popleft()

    def __bool__(self) -> bool:
        return bool(self._queue)

    def __len__(self) -> int:
        return len(self._queue)


class TimingCollector:
    def __init__(self) -> None:
        self._totals: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._stack: list[tuple[str, float]] = []
        self._completed_iterations: int = 0

    @dataclass
    class _Stage:
        collector: "TimingCollector"
        name: str

        def __enter__(self) -> None:
            self.collector._start(self.name)

        def __exit__(self, exc_type, exc, tb) -> None:
            self.collector._stop(self.name)

    def stage(self, name: str) -> "TimingCollector._Stage":
        return self._Stage(self, name)

    def _start(self, name: str) -> None:
        self._stack.append((name, time.perf_counter()))

    def _stop(self, name: str) -> None:
        if not self._stack:
            return
        stack_name, start = self._stack.pop()
        if stack_name != name:
            return
        elapsed = time.perf_counter() - start
        self._totals[name] = self._totals.get(name, 0.0) + elapsed
        self._counts[name] = self._counts.get(name, 0) + 1

    def snapshot(self) -> dict[str, dict[str, float]]:
        report: dict[str, dict[str, float]] = {}
        for key, total in sorted(self._totals.items()):
            count = self._counts.get(key, 0)
            avg = total / count if count else 0.0
            report[key] = {
                "total": total,
                "count": float(count),
                "avg": avg,
            }
        return report

    def log_summary(self, header: str = "Timing summary") -> None:
        if not self._totals:
            return
        logger.info(header)
        for key, metrics in self.snapshot().items():
            logger.info(
                "  %s: total=%.2fs count=%d avg=%.3fs",
                key,
                metrics["total"],
                int(metrics["count"]),
                metrics["avg"],
            )

    def mark_iteration_complete(self) -> None:
        self._completed_iterations += 1

    @property
    def completed_iterations(self) -> int:
        return self._completed_iterations


# =============================================================================
# PIPELINE IMPLEMENTATION
# =============================================================================


class ExampleProcessor:
    def __init__(self, context: PipelineContext, prompts: Prompts) -> None:
        self.context = context
        self.prompts = prompts
        self._parser = custom_parser.CustomParser()
        self._iteration_counter = 0

    async def process(self, item: WorkItem) -> tuple[Optional[ExampleRecord], Optional[WorkItem]]:
        preserved_tests = item.tests or "" if item.is_reprocess() else ""
        if item.is_reprocess():
            rust_code_full = item.code
            tests = preserved_tests
            if rust_code_full is None:
                logger.warning("Reprocess item missing code payload, skipping")
                return None, None
        else:
            with self._time("llm_code_writer"):
                response = await get_response(
                    self.context,
                    self.prompts.code_writer_system,
                    item.prompt,
                    model=self.context.llm_config.code_writer_model,
                    temperature=self.context.llm_config.code_writer_temperature,
                    max_tokens=self.context.llm_config.code_writer_max_tokens,
                )
            if response is None:
                return None, None

            rust_code_full = extract_rust_code(response)
            if rust_code_full is None:
                return None, None

        rust_code_without_tests, parsed_tests = separate_tests_from_code(rust_code_full)
        if parsed_tests:
            tests = parsed_tests
        elif item.is_reprocess():
            tests = preserved_tests
        else:
            tests = parsed_tests

        if tests and not self._validate_tests(tests):
            return None, None

        with self._time("cargo_diagnostics"):
            diagnostics = get_cargo_outputs(self.context, rust_code_full)
        if diagnostics is None:
            return None, None
        cargo_outputs = diagnostics.messages

        gold_comments = await self._generate_gold_comments(rust_code_without_tests, cargo_outputs)
        if not gold_comments and cargo_outputs:
            return None, None

        if gold_comments:
            gold_code, reprocess_code = await self._generate_gold_code(rust_code_without_tests, tests, gold_comments)
            if reprocess_code:
                retry_count = item.retry_count + 1
                if retry_count >= self.context.runtime_config.reprocess_retry_limit:
                    logger.warning("Skipping example after %s failed attempts", retry_count)
                    return None, None
                reprocess_item = WorkItem(
                    prompt=self.prompts.build_review_question(reprocess_code),
                    kind="reprocess",
                    retry_count=retry_count,
                    code=reprocess_code,
                    tests=tests,
                )
                return None, reprocess_item
            if not gold_code:
                return None, None
        else:
            gold_code = rust_code_without_tests

        review_prompt = self.prompts.build_review_question(rust_code_without_tests)

        info = ExampleInfo(
            cargo_outputs=cargo_outputs,
            gold_comments=gold_comments,
            gold_code=gold_code,
            tests=tests,
        )

        if self.context.timer:
            self.context.timer.mark_iteration_complete()

        self._iteration_counter += 1
        self._maybe_log_iteration_timing()

        return ExampleRecord(question=review_prompt, info=info), None

    def _validate_tests(self, tests: str) -> bool:
        test_validation_code = f"""
fn dummy_main() {{}}

{tests}
"""
        with self._time("test_validation"):
            test_cargo_outputs = get_cargo_outputs(self.context, test_validation_code)

        if test_cargo_outputs is None:
            logger.warning("Skipping example due to cargo failure during test validation")
            return False

        if test_cargo_outputs and test_cargo_outputs.messages:
            logger.warning(
                "Skipping example due to test compilation errors: %s",
                test_cargo_outputs.messages[:1],
            )
            return False

        return True

    async def _generate_gold_comments(self, code: str, cargo_outputs: list[str]) -> list[str]:
        prompt = self.prompts.build_code_reviewer_prompt(code, cargo_outputs)
        with self._time("llm_code_reviewer"):
            response = await get_response(
                self.context,
                self.prompts.code_reviewer_system,
                prompt,
                model=self.context.llm_config.code_reviewer_model,
                temperature=self.context.llm_config.code_reviewer_temperature,
                max_tokens=self.context.llm_config.code_reviewer_max_tokens,
            )

        if not response:
            return []

        comments = self._parser.parse_answer(response)

        if not comments:
            logger.debug(
                "Reviewer response produced no comments. Raw response: %s",
                response.strip(),
            )

        return comments

    async def _generate_gold_code(
        self,
        original_code: str,
        tests: str,
        gold_comments: list[str],
    ) -> tuple[Optional[str], Optional[str]]:
        if not gold_comments:
            return None, None

        prompt = self.prompts.build_review_applicator_prompt(original_code, gold_comments)
        with self._time("llm_review_applicator"):
            response = await get_response(
                self.context,
                self.prompts.review_applicator_system,
                prompt,
                model=self.context.llm_config.review_applicator_model,
                temperature=self.context.llm_config.review_applicator_temperature,
                max_tokens=self.context.llm_config.review_applicator_max_tokens,
            )

        if not response:
            return None, None

        refined_code = extract_rust_code(response)

        if not refined_code:
            return None, None

        combined_code = combine_code_with_tests(refined_code, tests)
        logger.info("Validating gold code with cargo (build/clippy/tests)")
        with self._time("cargo_gold_validate"):
            diagnostics = get_cargo_outputs(self.context, combined_code)

        if diagnostics is None:
            logger.warning("Gold code validation failed (cargo error), requeueing for reprocessing")
            return None, refined_code

        failing_tests = [msg for msg in diagnostics.messages if msg.startswith("[test_failure]")]
        if failing_tests:
            logger.warning("Gold code failed cargo tests, requeueing: %s", failing_tests[:1])
            return None, refined_code

        compile_errors = [msg for msg in diagnostics.messages if msg.startswith("[E")]
        if compile_errors:
            logger.warning("Gold code produced compiler errors, requeueing: %s", compile_errors[:1])
            return None, refined_code

        return refined_code, None

    def _time(self, name: str):
        if self.context.timer is None:
            return nullcontext()
        return self.context.timer.stage(name)

    def _maybe_log_iteration_timing(self) -> None:
        timer = self.context.timer
        config = self.context.runtime_config
        if not timer:
            return
        if config.timing_log_every <= 0:
            return
        if self.context.timer.completed_iterations % config.timing_log_every != 0:
            return
        timer.log_summary(f"Timing summary after {self.context.timer.completed_iterations} completed iterations")


def build_context(llm_config: LLMConfig, cargo_config: CargoConfig, runtime_config: RuntimeConfig) -> PipelineContext:
    client = AsyncOpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=llm_config.base_url)
    timer = TimingCollector() if runtime_config.enable_timing else None
    return PipelineContext(
        llm_config=llm_config,
        cargo_config=cargo_config,
        runtime_config=runtime_config,
        client=client,
        timer=timer,
    )


def extract_rust_code(response) -> Optional[str]:
    if isinstance(response, list):
        text = "\n".join([msg.get("content", "") for msg in response if msg.get("role") == "assistant"])
    else:
        text = response

    pattern = r"```rust\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)

    return match.group(1) if match else None


def separate_tests_from_code(rust_code: str) -> tuple[str, str]:
    test_pattern = r"(#\[cfg\(test\)\]\s*mod\s+tests\s*\{.*?\n\})"
    test_match = re.search(test_pattern, rust_code, re.DOTALL)

    if test_match:
        tests = test_match.group(1)
        code_without_tests = rust_code[: test_match.start()] + rust_code[test_match.end() :]
        return code_without_tests.strip(), tests.strip()
    else:
        return rust_code.strip(), ""


def combine_code_with_tests(code: str, tests: str) -> str:
    if not tests:
        return code
    return f"{code}\n\n{tests}"


def setup_project(context: PipelineContext, code: str) -> Path:
    base_dir = context.cargo_config.outputs_base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    project_dir = base_dir / f"temp_rust_project_{uuid.uuid4()}"
    src_dir = project_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    cargo_toml = """
    [package]
    name = "rust-project"
    version = "0.1.0"
    edition = "2021"
    
    [dependencies]
    """
    (project_dir / "Cargo.toml").write_text(cargo_toml, encoding="utf-8")

    main_rs = f"""
    #![allow(dead_code)]
    {code}
    
    fn main() {{
        println!("Hello World");
    }}
    """
    (src_dir / "main.rs").write_text(main_rs, encoding="utf-8")

    return project_dir


def parse_cargo_json_messages(stdout: str) -> list[str]:
    diagnostics = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if obj.get("reason") != "compiler-message":
            continue

        msg = obj.get("message", {})

        level = msg.get("level")

        if level not in ("error", "warning"):
            continue

        code_obj = msg.get("code") or {}
        code = code_obj.get("code") if code_obj.get("code") else level

        spans = msg.get("spans", [])
        location = None
        if spans:
            primary_span = next((s for s in spans if s.get("is_primary")), None)
            if primary_span:
                file_name = primary_span.get("file_name")
                line_start = primary_span.get("line_start")
                column_start = primary_span.get("column_start")
                if all([file_name, line_start, column_start]):
                    location = f"{file_name}:{line_start}:{column_start}"

        if location:
            formatted_diag = f"[{code}] {level} {location} {msg.get('message', '')}".strip()
        else:
            formatted_diag = f"[{code}] {level} {msg.get('message', '')}".strip()

        diagnostics.append(formatted_diag)

    return diagnostics


def run_cargo_json(context: PipelineContext, project_dir: Path, args: list[str], timeout: int) -> Optional[list[str]]:
    try:
        result = subprocess.run(
            args + ["--message-format", "json"],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Cargo command %s timed out after %ss", " ".join(args), timeout)
        return None
    if result.stdout is None:
        return None
    return parse_cargo_json_messages(result.stdout)


def run_cargo_build(context: PipelineContext, project_dir: Path) -> Optional[list[str]]:
    return run_cargo_json(context, project_dir, ["cargo", "build"], context.cargo_config.cargo_timeout)


def run_cargo_clippy(context: PipelineContext, project_dir: Path) -> Optional[list[str]]:
    return run_cargo_json(context, project_dir, ["cargo", "clippy"], context.cargo_config.cargo_timeout)


def run_cargo_tests(context: PipelineContext, project_dir: Path) -> Optional[list[str]]:
    try:
        result = subprocess.run(
            ["cargo", "test"],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=context.cargo_config.cargo_test_timeout,
        )

        diagnostics = []
        if result.returncode != 0:
            for line in result.stdout.split("\n"):
                line = line.strip()
                if "panicked at" in line:
                    location_match = re.search(r"panicked at ([^:]+:\d+:\d+)", line)
                    location = location_match.group(1) if location_match else ""
                    diagnostics.append(f"[test_failure] error {location} {line}")

        return diagnostics
    except subprocess.TimeoutExpired:
        logger.warning("Cargo test timed out, skipping")
        return None


def get_cargo_outputs(context: PipelineContext, code: str) -> Optional[CargoDiagnostics]:
    project_dir = setup_project(context, code)
    diagnostics: list[str] = []
    try:
        clippy_result = run_cargo_clippy(context, project_dir)
        build_result = run_cargo_build(context, project_dir)
        test_result = run_cargo_tests(context, project_dir)

        if clippy_result is None or build_result is None or test_result is None:
            return None

        diagnostics.extend(clippy_result)
        diagnostics.extend(build_result)
        diagnostics.extend(test_result)

        seen = set()
        unique_diagnostics = []
        for diag in diagnostics:
            match = re.search(r"\[([A-Z]\d+)\].*?([^\\/:]+:\d+:\d+)", diag.strip())
            key = f"{match.group(1)}|{match.group(2)}" if match else diag.strip()

            if key not in seen:
                unique_diagnostics.append(diag)
                seen.add(key)

        diagnostics = unique_diagnostics
    finally:
        try:
            shutil.rmtree(project_dir)
        except PermissionError:
            pass
        tests_dir = context.cargo_config.outputs_base_dir
        try:
            if tests_dir.exists() and not any(tests_dir.iterdir()):
                tests_dir.rmdir()
        except OSError:
            pass
    return CargoDiagnostics(messages=diagnostics)


async def get_response(
    context: PipelineContext,
    system_prompt: str,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Optional[str]:
    target_model = model
    for attempt in range(context.llm_config.llm_retry_attempts):
        try:
            response = await context.client.chat.completions.create(
                model=target_model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            await asyncio.sleep(context.llm_config.llm_retry_cooldown)
            return response.choices[0].message.content
        except Exception as e:
            if attempt == context.llm_config.llm_retry_attempts - 1:
                logger.error("API call failed after %s attempts: %s", context.llm_config.llm_retry_attempts, e)
                return None
            wait_time = context.llm_config.llm_retry_backoff * (attempt + 1)
            logger.warning(
                "API call failed (attempt %s/%s), retrying in %ss...",
                attempt + 1,
                context.llm_config.llm_retry_attempts,
                wait_time,
            )
            await asyncio.sleep(wait_time)
    return None


async def main() -> None:
    # =============================================================================
    # COMMAND-LINE ENTRYPOINT
    # =============================================================================

    import argparse

    parser = argparse.ArgumentParser(description="Generate Rust code review dataset")
    parser.add_argument(
        "--code_reviewer_model", type=str, help="Override the code reviewer model (default: x-ai/grok-4)"
    )
    parser.add_argument(
        "--target_dataset_hub", type=str, help="Override the target dataset hub (default: ljt019/rust-review-hq)"
    )
    args = parser.parse_args()

    llm_config = LLMConfig()
    dataset_config = DatasetConfig()
    cargo_config = CargoConfig()
    runtime_config = RuntimeConfig()

    if args.code_reviewer_model:
        llm_config.code_reviewer_model = args.code_reviewer_model
    if args.target_dataset_hub:
        dataset_config.target_dataset_hub = args.target_dataset_hub

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    logger.info("=== Data Generation Configuration ===")
    logger.info("Code reviewer model: %s", llm_config.code_reviewer_model)
    logger.info("Target dataset hub: %s", dataset_config.target_dataset_hub)
    logger.info("Max concurrent API calls: %s", runtime_config.max_concurrent_api_calls)
    logger.info("Target examples: %s", dataset_config.num_examples)
    logger.info("=====================================")

    context = build_context(llm_config, cargo_config, runtime_config)

    logger.info("Loading dataset %s", dataset_config.dataset_id)
    dataset = load_dataset(dataset_config.dataset_id, split="train", streaming=False)
    assert isinstance(dataset, Dataset)

    total_available = len(dataset)
    num_examples = min(dataset_config.num_examples, total_available)

    all_indices = list(range(total_available))
    random.shuffle(all_indices)
    selected_indices = all_indices[:num_examples]

    pipeline = ExampleProcessor(context, Prompts())
    initial_questions = [dataset[i]["question"] for i in selected_indices]

    results: list[ExampleRecord] = []

    try:
        existing_dataset_candidate = load_dataset(
            dataset_config.target_dataset_hub,
            split="train",
            streaming=False,
        )
        if not isinstance(existing_dataset_candidate, Dataset):
            logger.warning(
                "Expected a Dataset from %s but got %s; skipping preload",
                dataset_config.target_dataset_hub,
                type(existing_dataset_candidate).__name__,
            )
        elif isinstance(existing_dataset_candidate, Dataset):
            logger.info(
                "Loaded %s existing examples from %s",
                len(existing_dataset_candidate),
                dataset_config.target_dataset_hub,
            )
            initial_existing_records: list[ExampleRecord] = []
            for entry in existing_dataset_candidate:
                if not isinstance(entry, dict):
                    logger.warning(
                        "Unexpected entry type %s in existing dataset; skipping",
                        type(entry).__name__,
                    )
                    continue

                entry_info: Any = entry.get("info", {})
                if isinstance(entry_info, str):
                    try:
                        entry_info = json.loads(entry_info)
                    except json.JSONDecodeError:
                        logger.warning("Failed to decode info field for existing example; skipping")
                        continue

                if not isinstance(entry_info, dict):
                    logger.warning("Unexpected info field type (%s); skipping", type(entry_info).__name__)
                    continue

            info = ExampleInfo(
                cargo_outputs=entry_info.get("cargo_outputs", []),
                gold_comments=entry_info.get("gold_comments", []),
                gold_code=entry_info.get("gold_code", ""),
                tests=entry_info.get("tests", ""),
            )

            initial_existing_records.append(
                ExampleRecord(
                    question=entry.get("question", ""),
                    info=info,
                )
            )
            results.extend(initial_existing_records)
    except DatasetNotFoundError:
        logger.info(
            "No existing dataset found at %s; starting from empty base",
            dataset_config.target_dataset_hub,
        )
    except Exception as exc:
        logger.warning(
            "Failed to load existing dataset from %s: %s",
            dataset_config.target_dataset_hub,
            exc,
        )

    logger.info(
        "Processing %s new prompts (starting with %s existing examples)",
        len(initial_questions),
        len(results),
    )

    work_queue = ExampleQueue([WorkItem(prompt=question) for question in initial_questions])

    semaphore = asyncio.Semaphore(runtime_config.max_concurrent_api_calls)

    async def process_single_item(item: WorkItem) -> tuple[Optional[ExampleRecord], Optional[WorkItem]]:
        async with semaphore:
            with context.timer.stage("iteration") if context.timer else nullcontext():
                return await pipeline.process(item)

    with tqdm(total=len(initial_questions), desc="Processing examples") as pbar:
        processed_originals = 0
        pending_tasks = set()

        while work_queue or pending_tasks:
            while (
                len(pending_tasks) < runtime_config.max_concurrent_api_calls
                and work_queue
                and len(results) < num_examples
            ):
                item = work_queue.pop()
                if item is None:
                    break

                task = asyncio.create_task(process_single_item(item))
                setattr(task, "item", item)
                pending_tasks.add(task)

            if not pending_tasks:
                break

            done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                item = getattr(task, "item")
                result, reprocess_item = await task

                if item.kind == "dataset":
                    processed_originals += 1
                    pbar.update(1)

                if result is not None:
                    results.append(result)

                    if len(results) % runtime_config.checkpoint_every == 0:
                        logger.info("Uploading checkpoint at %s examples", len(results))
                        checkpoint_dataset = Dataset.from_list([r.to_dict() for r in results])
                        checkpoint_dataset.push_to_hub(dataset_config.target_dataset_hub)
                        logger.info("Checkpoint upload complete")
                elif reprocess_item is not None:
                    work_queue.push(reprocess_item)

    if not results:
        logger.error("No valid examples generated")
        return

    rebalance_clean_examples(results, dataset_config, logger)

    final_clean_count = sum(1 for result in results if not result.info.gold_comments)
    logger.info(
        "Generated %s examples (%s clean), merging with existing dataset",
        len(results),
        final_clean_count,
    )

    final_dataset = Dataset.from_list([record.to_dict() for record in results])

    logger.info(
        "Uploading dataset with %s total examples to %s",
        len(final_dataset),
        dataset_config.target_dataset_hub,
    )
    final_dataset.push_to_hub(dataset_config.target_dataset_hub)
    logger.info("Upload complete")

    if context.timer:
        context.timer.log_summary("Overall timing summary")


def rebalance_clean_examples(
    results: list[ExampleRecord], dataset_config: DatasetConfig, event_logger: logging.Logger
) -> None:
    existing_clean = sum(1 for result in results if not result.info.gold_comments)
    target_clean = round(len(results) * dataset_config.no_comment_ratio)

    if existing_clean < target_clean:
        additional_needed = target_clean - existing_clean
        event_logger.info("Converting %s additional examples to clean code examples", additional_needed)

        examples_with_comments = [i for i, result in enumerate(results) if result.info.gold_comments]
        if additional_needed <= len(examples_with_comments):
            clean_indices = random.sample(examples_with_comments, additional_needed)

            for idx in clean_indices:
                record = results[idx]
                original_code = record.question.split("```rust\n")[1].split("\n```")[0]
                gold_code = record.info.gold_code

                record.question = record.question.replace(original_code, gold_code)
                record.info.cargo_outputs = []
                record.info.gold_comments = []

    elif existing_clean > target_clean:
        excess_clean = existing_clean - target_clean
        event_logger.info("Removing %s clean examples to maintain ratio", excess_clean)

        clean_indices = [i for i, result in enumerate(results) if not result.info.gold_comments]
        indices_to_remove = random.sample(clean_indices, excess_clean)

        for idx in sorted(indices_to_remove, reverse=True):
            results.pop(idx)

    else:
        event_logger.info("Already have %s clean examples (target: %s), ratio is correct", existing_clean, target_clean)


if __name__ == "__main__":
    asyncio.run(main())
