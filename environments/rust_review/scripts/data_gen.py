from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Literal, Optional

from datasets import Dataset, load_dataset
from openai import OpenAI
from rust_review import custom_parser
from tqdm import tqdm

logger = logging.getLogger(__name__)


CODE_WRITER_SYSTEM_PROMPT = """You are a junior Rust developer who makes common programming mistakes. Given the following question, write a Rust function that attempts to complete the task but contains bugs. Do NOT write perfect code - you must introduce realistic bugs that cargo tools would catch. Then write unit tests for the function you defined. Write multiple unit tests for the function. The tests should be a simple line delimited list of assert! or assert_eq! statements. When writing the unit tests you can have comments specifying what you are testing in plain english. The tests should use super::*. CRITICAL: You MUST introduce bugs into your code. Do NOT write perfect code. Your code should always contain bugs and logical issues that cargo clippy, cargo build, or cargo test would catch."""


CARGO_OUTPUT_CONVERTER_SYSTEM_PROMPT = """You are a helpful code reviewer. Given some code and a list of cargo diagnostic outputs (from cargo test/clippy/build), convert them into clear, actionable code review comments."""


CODER_SYSTEM_PROMPT = """You are a code editor that applies ONLY the specific changes mentioned in review comments."""


CODER_PROMPT = """Original Code:\n```rust\n{code}\n```\n\nReview Comments (apply ONLY these specific changes):\n{comments}\n\nIMPORTANT: Apply only the exact changes mentioned in the review comments above. Do not make any other modifications to the code. Return the minimally modified code in a ```rust block."""


@dataclass
class Config:
    code_writer_model: str = "openai/gpt-4.1-nano"
    review_comment_model: str = "openai/gpt-5-mini"
    code_fixer_model: str = "openai/gpt-4.1-nano"
    base_url: str = "https://openrouter.ai/api/v1"
    num_examples: int = 10_000
    no_comment_ratio: float = 0.1
    dataset_id: str = "ljt019/rust-17000"
    target_dataset_hub: str = "ljt019/rust-review-coral"
    outputs_base_dir: str = field(default_factory=lambda: os.path.join("outputs", "tests"))
    cargo_timeout: int = 180
    cargo_test_timeout: int = 60
    llm_retry_attempts: int = 3
    llm_retry_backoff: int = 30
    reprocess_retry_limit: int = 6
    checkpoint_every: int = 250


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
    config: Config
    client: OpenAI


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
    cargo_output_system: str = CARGO_OUTPUT_CONVERTER_SYSTEM_PROMPT
    coder_system: str = CODER_SYSTEM_PROMPT
    coder_prompt_template: str = CODER_PROMPT
    review_template: str = REVIEW_PROMPT_TEMPLATE
    cargo_comment_template: str = CARGO_COMMENT_PROMPT_TEMPLATE

    def build_review_question(self, code: str) -> str:
        return self.review_template.format(code=code)

    def build_cargo_comment_prompt(self, code: str, cargo_outputs: list[str]) -> str:
        diagnostics_text = "\n".join(f"- {diagnostic}" for diagnostic in cargo_outputs)
        return self.cargo_comment_template.format(code=code, diagnostics=diagnostics_text)

    def build_coder_prompt(self, original_code: str, comments: list[str]) -> str:
        comments_text = "\n".join(f"- {comment}" for comment in comments)
        return self.coder_prompt_template.format(code=original_code, comments=comments_text)


@dataclass
class WorkItem:
    prompt: str
    kind: Literal["dataset", "reprocess"] = "dataset"
    retry_count: int = 0
    code: Optional[str] = None

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


class ExampleProcessor:
    def __init__(self, context: PipelineContext, prompts: Prompts) -> None:
        self.context = context
        self.prompts = prompts
        self._parser = custom_parser.CustomParser()

    def process(self, item: WorkItem) -> tuple[Optional[ExampleRecord], Optional[WorkItem]]:
        if item.is_reprocess():
            rust_code_full = item.code
            if rust_code_full is None:
                logger.warning("Reprocess item missing code payload, skipping")
                return None, None
        else:
            response = get_response(
                self.context,
                self.prompts.code_writer_system,
                item.prompt,
                model=self.context.config.code_writer_model,
            )
            if response is None:
                return None, None

            rust_code_full = extract_rust_code(response)
            if rust_code_full is None:
                return None, None

        rust_code_without_tests, tests = separate_tests_from_code(rust_code_full)

        if tests:
            test_validation_code = f"""
fn dummy_main() {{}}

{tests}
"""
            test_cargo_outputs = get_cargo_outputs(self.context, test_validation_code)
            if test_cargo_outputs is None:
                logger.warning("Skipping example due to cargo failure during test validation")
                return None, None
            if test_cargo_outputs:
                logger.warning(
                    "Skipping example due to test compilation errors: %s",
                    test_cargo_outputs.messages[:1],
                )
                return None, None

        diagnostics = get_cargo_outputs(self.context, rust_code_full)
        if diagnostics is None:
            return None, None
        cargo_outputs = diagnostics.messages

        gold_comments = self._generate_gold_comments(rust_code_without_tests, cargo_outputs)
        if not gold_comments and cargo_outputs:
            return None, None

        if gold_comments:
            gold_code, reprocess_code = self._generate_gold_code(rust_code_without_tests, gold_comments)
            if reprocess_code:
                retry_count = item.retry_count + 1
                if retry_count >= self.context.config.reprocess_retry_limit:
                    logger.warning("Skipping example after %s failed attempts", retry_count)
                    return None, None
                reprocess_item = WorkItem(
                    prompt=self.prompts.build_review_question(reprocess_code),
                    kind="reprocess",
                    retry_count=retry_count,
                    code=reprocess_code,
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
        return ExampleRecord(question=review_prompt, info=info), None

    def _generate_gold_comments(self, code: str, cargo_outputs: list[str]) -> list[str]:
        if not cargo_outputs:
            return []

        prompt = self.prompts.build_cargo_comment_prompt(code, cargo_outputs)
        response = get_response(
            self.context,
            self.prompts.cargo_output_system,
            prompt,
        )

        if not response:
            return []

        return self._parser.parse_answer(response)

    def _generate_gold_code(
        self,
        original_code: str,
        gold_comments: list[str],
    ) -> tuple[Optional[str], Optional[str]]:
        if not gold_comments:
            return None, None

        prompt = self.prompts.build_coder_prompt(original_code, gold_comments)
        response = get_response(
            self.context,
            self.prompts.coder_system,
            prompt,
            model=self.context.config.code_fixer_model,
        )

        if not response:
            return None, None

        refined_code = extract_rust_code(response)

        if not refined_code:
            return None, None

        logger.info("Validating gold code compiles")
        try:
            project_dir = setup_project(self.context, refined_code)
            try:
                result = subprocess.run(
                    ["cargo", "build", "--quiet"],
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.context.config.cargo_timeout,
                )
                compiles = result.returncode == 0

                if not compiles:
                    logger.warning("Gold code failed to compile, requeueing for reprocessing")
                    return None, refined_code
                return refined_code, None

            finally:
                shutil.rmtree(project_dir)

        except Exception as exc:
            logger.error("Error validating gold code: %s", exc)
            return None, None


logger = logging.getLogger(__name__)


CODE_WRITER_SYSTEM_PROMPT = """
You are a junior Rust developer who makes common programming mistakes. Given the following question, write a Rust function that attempts to complete the task but contains bugs. Do NOT write perfect code - you must introduce realistic bugs that cargo tools would catch.

Then write unit tests for the function you defined. Write multiple unit tests for the function. The tests should be a simple line delimited list of assert! or assert_eq! statements. When writing the unit tests you can have comments specifying what you are testing in plain english. The tests should use super::*.

An example output should look like the following:

```rust
/// Reasoning goes here
/// and can be multi-line
fn add_nums(x: i32, y: i32) -> i32 {
    x + y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_nums() {
        // Test adding positive numbers
        assert_eq!(add_nums(4, 2), 6);
        // Test adding a positive and negative number
        assert_eq!(add_nums(4, -2), 2);
        // Test adding two negative numbers
        assert_eq!(add_nums(-12, -1), -13);
    }
}
```

CRITICAL: You MUST introduce bugs into your code. Do NOT write perfect code. Your code should always contain bugs and logical issues that cargo clippy, cargo build, or cargo test would catch.

Examples of bugs to include:
- Wrong operators (+ instead of -, < instead of <=)
- Off-by-one errors in loops or indexing
- Incorrect method calls (using wrong methods on types)
- Logic errors in conditions
- Performance issues (unnecessary cloning, inefficient algorithms)

The tests should be correct and would pass if the function were implemented properly.

Make sure to only respond with a single  ```rust``` block. The unit tests must be defined inside the mod tests {} module. Make sure to import any standard library modules that you need. Do not add a main function.
"""

CARGO_OUTPUT_CONVERTER_SYSTEM_PROMPT = """
You are a helpful code reviewer. Given some code and a list of cargo diagnostic outputs (from cargo test/clippy/build), convert them into clear, actionable code review comments.

RULES:
1. Generate exactly one review comment for each unique cargo diagnostic (skip true duplicates)
2. If you spot obvious logic bugs, incorrect algorithms, or significant maintainability issues that cargo cannot detect, you MAY add additional comments - but be selective and avoid nitpicking
3. Focus on issues that would genuinely help the developer - avoid minor style preferences or overly pedantic suggestions
4. If there are no cargo outputs and no obvious issues, it's perfectly fine to leave the review empty

For each diagnostic, provide:
- A clear explanation of the issue
- Why it matters (performance, safety, readability, etc.)  
- A specific suggestion for how to fix it
- If relevant, a brief code example of the fix

Format your response as follows:
{format_str}

Example input:
```rust
fn sum_vec(v: Vec<i32>) -> i32 {
    v.iter().collect::<Vec<_>>().iter().sum()
}
```

Cargo diagnostics:
- [clippy::needless_collect] warning src/main.rs:2:5 avoid collecting into a vector unnecessarily
- [dead_code] warning src/main.rs:1:4 function `sum_vec` is never used

Example output:
<think>
Two diagnostics to address:
1. Clippy warning about unnecessary collection - this is a performance issue
2. Dead code warning - this is about code cleanliness
</think>

<review>
<comment>**Unnecessary collection (line 2)**: You're collecting into a vector when you could iterate directly. This creates unnecessary memory allocation and hurts performance. Consider chaining iterators instead of `.collect().iter()` - change `v.iter().collect::<Vec<_>>().iter().sum()` to `v.iter().sum()`.</comment>
<comment>**Unused function (line 1)**: The function `sum_vec` is defined but never called. Consider removing it if it's not needed, or add `#[allow(dead_code)]` if it's intentionally unused for now.</comment>
</review>
"""

CODER_SYSTEM_PROMPT = """
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

CODER_PROMPT = """
Original Code:
```rust
{code}
```

Review Comments (apply ONLY these specific changes):
{comments}

IMPORTANT: Apply only the exact changes mentioned in the review comments above. Do not make any other modifications to the code. Return the minimally modified code in a ```rust block.
"""


def build_context(config: Config) -> PipelineContext:
    client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=config.base_url)
    return PipelineContext(config=config, client=client)


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


def setup_project(context: PipelineContext, code: str) -> str:
    base_dir = context.config.outputs_base_dir
    os.makedirs(base_dir, exist_ok=True)

    project_dir = os.path.join(base_dir, f"temp_rust_project_{uuid.uuid4()}")
    src_dir = os.path.join(project_dir, "src")
    os.makedirs(src_dir, exist_ok=True)

    cargo_toml = """
    [package]
    name = "rust-project"
    version = "0.1.0"
    edition = "2021"
    
    [dependencies]
    """
    with open(os.path.join(project_dir, "Cargo.toml"), "w", encoding="utf-8") as f:
        f.write(cargo_toml)

    main_rs = f"""
    #![allow(dead_code)]
    {code}
    
    fn main() {{
        println!("Hello World");
    }}
    """
    with open(os.path.join(src_dir, "main.rs"), "w", encoding="utf-8") as f:
        f.write(main_rs)

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


def run_cargo_json(context: PipelineContext, project_dir: str, args: list[str], timeout: int) -> Optional[list[str]]:
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
    if result.stdout is None:
        return None
    return parse_cargo_json_messages(result.stdout)


def run_cargo_build(context: PipelineContext, project_dir: str) -> Optional[list[str]]:
    return run_cargo_json(context, project_dir, ["cargo", "build"], context.config.cargo_timeout)


def run_cargo_clippy(context: PipelineContext, project_dir: str) -> Optional[list[str]]:
    return run_cargo_json(context, project_dir, ["cargo", "clippy"], context.config.cargo_timeout)


def run_cargo_tests(context: PipelineContext, project_dir: str) -> Optional[list[str]]:
    try:
        result = subprocess.run(
            ["cargo", "test"],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=context.config.cargo_test_timeout,
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
        tests_dir = context.config.outputs_base_dir
        try:
            if os.path.exists(tests_dir) and not os.listdir(tests_dir):
                os.rmdir(tests_dir)
        except OSError:
            pass
    return CargoDiagnostics(messages=diagnostics)


def get_response(
    context: PipelineContext,
    system_prompt: str,
    prompt: str,
    model: Optional[str] = None,
) -> Optional[str]:
    target_model = model or context.config.review_comment_model
    for attempt in range(context.config.llm_retry_attempts):
        try:
            response = context.client.chat.completions.create(
                model=target_model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=0.0 if target_model == context.config.code_fixer_model else 0.1,
                max_tokens=4000,
            )
            time.sleep(0.5)
            return response.choices[0].message.content
        except Exception as e:
            if attempt == context.config.llm_retry_attempts - 1:
                logger.error("API call failed after %s attempts: %s", context.config.llm_retry_attempts, e)
                return None
            wait_time = context.config.llm_retry_backoff * (attempt + 1)
            logger.warning(
                "API call failed (attempt %s/%s), retrying in %ss...",
                attempt + 1,
                context.config.llm_retry_attempts,
                wait_time,
            )
            time.sleep(wait_time)
    return None


def main() -> None:
    import random

    config = Config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    context = build_context(config)

    logger.info("Loading dataset %s", config.dataset_id)
    dataset = load_dataset(config.dataset_id, split="train", streaming=False)
    assert isinstance(dataset, Dataset)

    total_available = len(dataset)
    num_examples = min(config.num_examples, total_available)

    all_indices = list(range(total_available))
    random.shuffle(all_indices)
    selected_indices = all_indices[:num_examples]

    pipeline = ExampleProcessor(context, Prompts())
    initial_questions = [dataset[i]["question"] for i in selected_indices]

    logger.info("Processing %s examples", len(initial_questions))

    work_queue = ExampleQueue([WorkItem(prompt=question) for question in initial_questions])
    results: list[ExampleRecord] = []

    with tqdm(total=len(initial_questions), desc="Processing examples") as pbar:
        processed_originals = 0

        while work_queue and len(results) < num_examples:
            item = work_queue.pop()
            if item is None:
                break

            if item.kind == "dataset":
                processed_originals += 1
                pbar.update(1)

            result, reprocess_item = pipeline.process(item)

            if result is not None:
                results.append(result)

                if len(results) % config.checkpoint_every == 0:
                    logger.info("Uploading checkpoint at %s examples", len(results))
                    checkpoint_dataset = Dataset.from_list([r.to_dict() for r in results])
                    checkpoint_dataset.push_to_hub(config.target_dataset_hub)
                    logger.info("Checkpoint uploaded successfully")
            elif reprocess_item is not None:
                work_queue.push(reprocess_item)

    if not results:
        logger.error("No valid examples generated")
        return

    existing_clean = sum(1 for result in results if not result.info.gold_comments)
    target_clean = round(len(results) * config.no_comment_ratio)

    if existing_clean < target_clean:
        additional_needed = target_clean - existing_clean
        logger.info("Converting %s additional examples to clean code examples", additional_needed)

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
        logger.info("Removing %s clean examples to maintain ratio", excess_clean)

        clean_indices = [i for i, result in enumerate(results) if not result.info.gold_comments]
        indices_to_remove = random.sample(clean_indices, excess_clean)

        for idx in sorted(indices_to_remove, reverse=True):
            results.pop(idx)

    else:
        logger.info("Already have %s clean examples (target: %s), ratio is correct", existing_clean, target_clean)

    final_clean_count = sum(1 for result in results if not result.info.gold_comments)
    logger.info("Generated %s examples (%s clean), uploading to hub", len(results), final_clean_count)
    final_dataset = Dataset.from_list([record.to_dict() for record in results])
    final_dataset.push_to_hub(config.target_dataset_hub)
    logger.info("Upload complete")


if __name__ == "__main__":
    main()
