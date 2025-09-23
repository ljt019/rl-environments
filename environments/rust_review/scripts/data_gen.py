import json
import os
import re
import shutil
import subprocess
import time
import uuid
from typing import Optional

from datasets import Dataset, load_dataset
from openai import OpenAI
from rust_review import custom_parser
from tqdm import tqdm

CODE_WRITER_MODEL = "openai/gpt-4.1-nano"  # Generates initial Rust code from problem descriptions
REVIEW_COMMENT_MODEL = "openai/gpt-5"  # Generates review comments from cargo outputs
CODE_FIXER_MODEL = "openai/gpt-4.1-nano"  # Applies review comments to fix code
BASE_URL = "https://openrouter.ai/api/v1"

NUM_EXAMPLES = 10
NO_COMMENT_RATIO = 0.1

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

client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=BASE_URL)


def extract_rust_code(response) -> Optional[str]:
    """Extract Rust code from response text"""
    import re

    if isinstance(response, list):
        text = "\n".join([msg.get("content", "") for msg in response if msg.get("role") == "assistant"])
    else:
        text = response

    pattern = r"```rust\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1)
    else:
        return None


def separate_tests_from_code(rust_code: str) -> tuple[str, str]:
    """Separate tests from main code. Returns (code_without_tests, tests)"""
    # Extract the test module
    test_pattern = r"(#\[cfg\(test\)\]\s*mod\s+tests\s*\{.*?\n\})"
    test_match = re.search(test_pattern, rust_code, re.DOTALL)

    if test_match:
        tests = test_match.group(1)
        code_without_tests = rust_code[: test_match.start()] + rust_code[test_match.end() :]
        return code_without_tests.strip(), tests.strip()
    else:
        return rust_code.strip(), ""


def combine_code_with_tests(code: str, tests: str) -> str:
    """Combine code with tests for cargo operations"""
    if not tests:
        return code
    return f"{code}\n\n{tests}"


def setup_project(code: str) -> str:
    """Creates a temporary Rust project with the given code"""
    base_dir = os.path.join("outputs", "tests")
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
    """Parse cargo --message-format json output into formatted diagnostic strings."""
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


def run_cargo_json(project_dir: str, args: list[str], timeout: int = 180) -> list[str]:
    """Run a cargo subcommand with --message-format json and return formatted diagnostic strings."""
    result = subprocess.run(
        args + ["--message-format", "json"],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return parse_cargo_json_messages(result.stdout)


def run_cargo_build(project_dir) -> list[str]:
    """Run cargo build and return a list of formatted diagnostic strings."""
    return run_cargo_json(project_dir, ["cargo", "build"])


def run_cargo_clippy(project_dir) -> list[str]:
    """Run cargo clippy and return a list of formatted diagnostic strings."""
    return run_cargo_json(project_dir, ["cargo", "clippy"])


def run_cargo_tests(project_dir) -> list[str]:
    """Run cargo test and return a list of formatted diagnostic strings."""
    result = subprocess.run(
        ["cargo", "test"],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
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


def get_cargo_outputs(code: str) -> list[str]:
    """Run cargo clippy/build/test and return formatted diagnostic lines"""
    project_dir = setup_project(code)
    diagnostics: list[str] = []
    try:
        diagnostics.extend(run_cargo_clippy(project_dir))
        diagnostics.extend(run_cargo_build(project_dir))

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
        shutil.rmtree(project_dir)
        tests_dir = os.path.join("outputs", "tests")
        try:
            if os.path.exists(tests_dir) and not os.listdir(tests_dir):
                os.rmdir(tests_dir)
        except OSError:
            pass
    return diagnostics


def get_response(system_prompt: str, prompt: str, model: str = REVIEW_COMMENT_MODEL) -> Optional[str]:
    """Get response from LLM with rate limiting"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        temperature=0.0 if model == CODE_FIXER_MODEL else 0.1,
        max_tokens=4000,
    )
    time.sleep(0.5)
    return response.choices[0].message.content


def generate_gold_comments(code: str, cargo_outputs: list[str]) -> list[str]:
    """Convert cargo diagnostic outputs into human-readable review comments"""
    if not cargo_outputs:
        return []

    diagnostics_text = "\n".join(f"- {diagnostic}" for diagnostic in cargo_outputs)

    prompt = f"""```rust
{code}
```

Cargo diagnostics:
{diagnostics_text}"""

    response = get_response(CARGO_OUTPUT_CONVERTER_SYSTEM_PROMPT, prompt)

    if not response:
        return []

    # Use the CustomParser to extract comments
    parser = custom_parser.CustomParser()
    comments = parser.parse_answer(response)
    return comments


def generate_gold_code(original_code: str, gold_comments: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Apply gold comments to original code to generate gold_code
    Returns (gold_code, reprocess_question) where reprocess_question is set if gold_code fails to compile
    """
    if not gold_comments:
        return None, None

    comments_text = "\n".join([f"- {comment}" for comment in gold_comments])
    prompt = CODER_PROMPT.format(code=original_code, comments=comments_text)
    response = get_response(CODER_SYSTEM_PROMPT, prompt, model=CODE_FIXER_MODEL)

    if response:
        refined_code = extract_rust_code(response)

        if refined_code:
            print("Validating gold code compiles...")
            try:
                project_dir = setup_project(refined_code)
                try:
                    result = subprocess.run(
                        ["cargo", "build", "--quiet"],
                        cwd=project_dir,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    compiles = result.returncode == 0

                    if not compiles:
                        print("Gold code failed to compile, requeueing for reprocessing...")
                        return None, refined_code
                    else:
                        return refined_code, None

                finally:
                    shutil.rmtree(project_dir)

            except Exception as e:
                print(f"Error validating gold code: {e}")
                return None, None

    return None, None


def process_example(question: str, is_reprocess: bool = False) -> tuple[Optional[dict], Optional[str]]:
    """Process a single example through the complete pipeline
    Returns (result, reprocess_code) where reprocess_code is set if we need to reprocess
    """
    if is_reprocess:
        # Extract code from review-formatted question
        rust_code_full = question.split("```rust\n")[1].split("\n```")[0]
    else:
        # Generate code from problem description
        response = get_response(CODE_WRITER_SYSTEM_PROMPT, question, model=CODE_WRITER_MODEL)
        if response is None:
            return None, None

        rust_code_full = extract_rust_code(response)
        if rust_code_full is None:
            return None, None

    # Separate tests from code
    rust_code_without_tests, tests = separate_tests_from_code(rust_code_full)

    # Validate that tests themselves compile cleanly (no syntax/import errors in tests)
    if tests:
        # Create a minimal main function + tests to check test compilation
        test_validation_code = f"""
fn dummy_main() {{}}

{tests}
"""
        test_cargo_outputs = get_cargo_outputs(test_validation_code)
        if test_cargo_outputs:
            # Tests have compilation errors - skip this example
            print(f"Skipping example due to test compilation errors: {test_cargo_outputs[:1]}")
            return None, None

    # Use full code (with tests) for cargo operations
    cargo_outputs = get_cargo_outputs(rust_code_full)

    # Use code without tests for review generation
    gold_comments = generate_gold_comments(rust_code_without_tests, cargo_outputs)
    if not gold_comments and cargo_outputs:
        return None, None

    if gold_comments:
        gold_code, reprocess_code = generate_gold_code(rust_code_without_tests, gold_comments)
        if reprocess_code:
            # When reprocessing, we need to combine the failed code with tests
            reprocess_code_with_tests = combine_code_with_tests(reprocess_code, tests)
            return None, reprocess_code_with_tests
        if not gold_code:
            return None, None
    else:
        gold_code = rust_code_without_tests

    # Review prompt uses code without tests
    review_prompt = f"""Please review the following code and provide feedback on any issues you find.

Here's the code for you to review:

```rust
{rust_code_without_tests}
```"""

    return {
        "question": review_prompt,
        "info": {
            "cargo_outputs": cargo_outputs,
            "gold_comments": gold_comments,
            "gold_code": gold_code,
            "tests": tests,
        },
    }, None


def main():
    """Main data generation pipeline"""
    import random
    from collections import deque

    print("Loading dataset...")
    dataset = load_dataset("ljt019/rust-17000", split="train", streaming=False)
    assert isinstance(dataset, Dataset)

    # Shuffle the dataset indices to get random examples each run
    total_available = len(dataset)
    num_examples = min(NUM_EXAMPLES, total_available)

    # Create random indices and select that subset
    all_indices = list(range(total_available))
    random.shuffle(all_indices)
    selected_indices = all_indices[:num_examples]

    # Get the questions from the randomly selected indices
    initial_questions = [dataset[i]["question"] for i in selected_indices]

    print(f"Processing {len(initial_questions)} examples...")

    # Use a queue to handle reprocessing failed examples
    questions_queue = deque(initial_questions)
    results = []

    with tqdm(total=len(initial_questions), desc="Processing examples") as pbar:
        processed_count = 0

        while questions_queue and len(results) < num_examples:
            question = questions_queue.popleft()
            is_reprocess = isinstance(question, tuple) and question[0] == "reprocess"

            if is_reprocess:
                _, failed_code = question
                reprocess_question = f"""Please review the following code and provide feedback on any issues you find.

Here's the code for you to review:

```rust
{failed_code}
```"""
                result, reprocess_code = process_example(reprocess_question, is_reprocess=True)
            else:
                result, reprocess_code = process_example(question, is_reprocess=False)

            if result is not None:
                results.append(result)
                processed_count += 1
                if processed_count <= len(initial_questions):
                    pbar.update(1)
            elif reprocess_code is not None:
                # Add the failed code back to queue for reprocessing
                questions_queue.append(("reprocess", reprocess_code))
                # Don't update progress bar for reprocessed items
            else:
                # Complete failure, skip and update progress
                processed_count += 1
                if processed_count <= len(initial_questions):
                    pbar.update(1)

    if not results:
        print("No valid examples generated")
        return

    # Count existing clean examples (no gold_comments)
    existing_clean = sum(1 for result in results if not result["info"]["gold_comments"])
    target_clean = int(len(results) * NO_COMMENT_RATIO)

    if existing_clean < target_clean:
        additional_needed = target_clean - existing_clean
        print(f"Converting {additional_needed} additional examples to clean code examples...")

        # Only convert examples that currently have comments
        examples_with_comments = [i for i, result in enumerate(results) if result["info"]["gold_comments"]]
        if additional_needed <= len(examples_with_comments):
            clean_indices = random.sample(examples_with_comments, additional_needed)

            for idx in clean_indices:
                result = results[idx]
                original_code = result["question"].split("```rust\n")[1].split("\n```")[0]
                gold_code = result["info"]["gold_code"]

                result["question"] = result["question"].replace(original_code, gold_code)
                result["info"]["cargo_outputs"] = []
                result["info"]["gold_comments"] = []
                # Keep the tests as they are still needed for validation

    elif existing_clean > target_clean:
        excess_clean = existing_clean - target_clean
        print(f"Removing {excess_clean} clean examples to maintain ratio...")

        # Find all clean examples and randomly remove excess ones
        clean_indices = [i for i, result in enumerate(results) if not result["info"]["gold_comments"]]
        indices_to_remove = random.sample(clean_indices, excess_clean)

        # Remove in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            results.pop(idx)

    else:
        print(f"Already have {existing_clean} clean examples (target: {target_clean}), ratio is correct")

    final_clean_count = sum(1 for result in results if not result["info"]["gold_comments"])
    print(f"Generated {len(results)} examples ({final_clean_count} clean), uploading to hub...")
    final_dataset = Dataset.from_list(results)
    final_dataset.push_to_hub("ljt019/rust-review-coral")
    print("Upload complete")


if __name__ == "__main__":
    main()
