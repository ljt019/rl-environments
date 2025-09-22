import json
import os
import shutil
import subprocess
import time
import uuid
from typing import Optional

from datasets import Dataset, load_dataset
from openai import OpenAI
from tqdm import tqdm

# Configuration
MODEL = "qwen/qwen3-coder-30b-a3b-instruct"
CODER_MODEL = "qwen/qwen3-coder-30b-a3b-instruct"
BASE_URL = "https://openrouter.ai/api/v1"

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


def setup_project(code: str) -> str:
    """Creates a temporary Rust project with the given code"""
    base_dir = os.path.join("outputs", "tests")
    os.makedirs(base_dir, exist_ok=True)

    # Create temporary project directory
    project_dir = os.path.join(base_dir, f"temp_rust_project_{uuid.uuid4()}")
    src_dir = os.path.join(project_dir, "src")
    os.makedirs(src_dir, exist_ok=True)

    # Write Cargo.toml
    cargo_toml = """
    [package]
    name = "rust-project"
    version = "0.1.0"
    edition = "2021"
    
    [dependencies]
    """
    with open(os.path.join(project_dir, "Cargo.toml"), "w") as f:
        f.write(cargo_toml)

    # Write main.rs with the code
    main_rs = f"""
    #![allow(dead_code)]
    {code}
    
    // Need basic main function for the code to compile
    fn main() {{
        println!("Hello World");
    }}
    """
    with open(os.path.join(src_dir, "main.rs"), "w") as f:
        f.write(main_rs)

    return project_dir


def get_cargo_outputs(code: str) -> list[str]:
    """Run cargo clippy/build and return structured diagnostic lines"""
    project_dir = setup_project(code)

    diagnostics: list[str] = []
    try:
        commands = [
            ["cargo", "clippy", "--message-format", "json"],
            ["cargo", "build", "--message-format", "json"],
        ]

        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=project_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=180,
                )

                if not result.stdout:
                    continue

                for line in result.stdout.splitlines():
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
                    code_str = code_obj.get("code")
                    if not code_str and level == "error":
                        code_str = "error"
                    elif not code_str and level == "warning":
                        code_str = "warning"

                    primary_span = None
                    for s in msg.get("spans", []) or []:
                        if s.get("is_primary"):
                            primary_span = s
                            break

                    loc = None
                    if primary_span is not None:
                        file_name = primary_span.get("file_name")
                        line_start = primary_span.get("line_start")
                        column_start = primary_span.get("column_start")
                        if file_name is not None and line_start is not None and column_start is not None:
                            loc = f"{file_name}:{line_start}:{column_start}"

                    message_text = msg.get("message") or ""

                    if loc:
                        diagnostics.append(f"[{code_str}] {level} {loc} {message_text}".strip())
                    else:
                        diagnostics.append(f"[{code_str}] {level} {message_text}".strip())
            except Exception as e:
                diagnostics.append(f"Error running {' '.join(cmd)}: {e}")
    finally:
        shutil.rmtree(project_dir)

        tests_dir = os.path.join("outputs", "tests")
        try:
            if os.path.exists(tests_dir) and not os.listdir(tests_dir):
                os.rmdir(tests_dir)
        except OSError:
            pass

    return diagnostics


# System prompts
CODE_WRITER_SYSTEM_PROMPT = """
You are a pragmatic Rust programmer who enjoys test driven development. Given the following question, write a Rust function to complete the task. Make the code simple and easy to understand. The code should pass `cargo build` and `cargo clippy`. Try to limit library usage to the standard library std. Be careful with your types, and try to limit yourself to the basic built in types and standard library functions. When writing the function you can think through how to solve the problem and perform reasoning in the comments above the function.

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

You should act as a junior Rust developer. Your code should usually contain bugs and logical issues, but not always.
The tests should always be correct even if the code contains bugs.

Make sure to only respond with a single  ```rust``` block. The unit tests must be defined inside the mod tests {} module. Make sure to import any standard library modules that you need. Do not add a main function.
"""

CARGO_OUTPUT_CONVERTER_SYSTEM_PROMPT = """
You are a helpful Rust code reviewer. Given a single cargo clippy or build diagnostic, convert it into a clear, actionable code review comment that a human reviewer would write.

Provide:
- A clear explanation of the issue
- Why it matters (performance, safety, readability, etc.)  
- A specific suggestion for how to fix it
- If relevant, a brief code example of the fix

Keep the comment concise but helpful. Write as a single paragraph or short bulleted section.

Example input:
[clippy::needless_collect] warning src/main.rs:8:9 avoid collecting into a vector unnecessarily

Example output:
**Unnecessary collection (line 8)**: You're collecting into a vector when you could iterate directly. This creates unnecessary memory allocation and hurts performance. Consider chaining iterators instead of `.collect().iter()` - for example, change `items.collect::<Vec<_>>().iter().map(|x| x * 2)` to `items.iter().map(|x| x * 2)`.
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


def get_response(system_prompt: str, prompt: str, model: str = MODEL) -> Optional[str]:
    """Get response from LLM with rate limiting"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=0.0 if model == CODER_MODEL else 0.1,
            max_tokens=4000,
        )
        # Brief sleep to avoid rate limits
        time.sleep(0.5)
        return response.choices[0].message.content
    except Exception as e:
        print(f"API error: {e}")
        return None


def generate_gold_comments(cargo_outputs: list[str]) -> list[str]:
    """Convert cargo diagnostic outputs into human-readable review comments"""
    if not cargo_outputs:
        return []

    gold_comments = []

    for diagnostic in cargo_outputs:
        # Get a review comment for each individual diagnostic
        comment = get_response(CARGO_OUTPUT_CONVERTER_SYSTEM_PROMPT, diagnostic)
        if comment:
            gold_comments.append(comment.strip())

    return gold_comments


def generate_gold_code(original_code: str, gold_comments: list[str]) -> Optional[str]:
    """Apply gold comments to original code to generate gold_code"""
    if not gold_comments:
        return None

    # Format comments as a bullet list
    comments_text = "\n".join([f"- {comment}" for comment in gold_comments])

    # Create the prompt for the coder model
    prompt = CODER_PROMPT.format(code=original_code, comments=comments_text)

    # Get the refined code from the coder model
    response = get_response(CODER_SYSTEM_PROMPT, prompt, model=CODER_MODEL)

    if response:
        # Extract the refined code
        refined_code = extract_rust_code(response)

        if refined_code:
            # CRITICAL: Verify the gold code actually compiles!
            print("    Validating gold code compiles...")
            try:
                # Check if refined code compiles
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
                        print(f"    ❌ Gold code failed to compile! Stderr: {result.stderr[:200]}...")
                        return None
                    else:
                        print("    ✅ Gold code compiles successfully")
                        return refined_code

                finally:
                    shutil.rmtree(project_dir)

            except Exception as e:
                print(f"    ❌ Error validating gold code: {e}")
                return None

    return None


def process_example(question: str) -> Optional[dict]:
    """Process a single example through the complete pipeline"""
    try:
        print("[STEP 1] Generating initial code for question...")

        # Step 1: Generate initial code
        response = get_response(CODE_WRITER_SYSTEM_PROMPT, question)
        if response is None:
            print("Failed to generate initial code")
            return None

        rust_code = extract_rust_code(response)
        if rust_code is None:
            print("Failed to extract Rust code from response")
            return None

        print("[STEP 2] Running cargo/clippy on generated code...")

        # Step 2: Get cargo outputs
        cargo_outputs = get_cargo_outputs(rust_code)

        # Handle clean code (no issues) - we want some of these for training!
        if len(cargo_outputs) == 0:
            print("No cargo outputs found - creating 'clean code' example")

            # Create the review prompt for clean code
            review_prompt = f"""Please review the following code and provide feedback on any issues you find.

Here's the code for you to review:

```rust
{rust_code}
```"""

            # Return clean code example with empty gold_comments and original code as gold_code
            result = {
                "question": review_prompt,
                "info": {
                    "cargo_outputs": [],  # No issues
                    "gold_comments": [],  # No comments needed - this is the gold standard!
                    "gold_code": rust_code,  # Original code is already good
                },
            }

            print("✅ Successfully processed clean code example (no issues - model should output empty review)")
            return result

        print(f"[STEP 3] Converting {len(cargo_outputs)} cargo outputs to gold comments...")

        # Step 3: Generate gold comments from cargo outputs
        gold_comments = generate_gold_comments(cargo_outputs)
        if not gold_comments:
            print("Failed to generate gold comments")
            return None

        print(f"[STEP 4] Applying {len(gold_comments)} gold comments to generate gold code...")

        # Step 4: Generate gold code by applying gold comments
        gold_code = generate_gold_code(rust_code, gold_comments)
        if not gold_code:
            print("Failed to generate gold code")
            return None

        print("[STEP 5] Creating final dataset entry...")

        # Step 5: Create the review prompt (what the model will see)
        review_prompt = f"""Please review the following code and provide feedback on any issues you find.

Here's the code for you to review:

```rust
{rust_code}
```"""

        # Step 6: Return the complete dataset entry
        result = {
            "question": review_prompt,
            "info": {
                "cargo_outputs": cargo_outputs,
                "gold_comments": gold_comments,
                "gold_code": gold_code,
            },
        }

        print(
            f"✅ Successfully processed example with {len(cargo_outputs)} issues -> {len(gold_comments)} comments -> gold code"
        )
        return result

    except Exception as e:
        print(f"Error processing example: {e}")
        return None


def main():
    """Main data generation pipeline"""
    print("🚀 Starting final data generation pipeline...")

    # Load the source dataset
    print("📚 Loading source dataset...")
    dataset = load_dataset("ljt019/rust-17000", split="train", streaming=False)
    assert isinstance(dataset, Dataset)

    # Process a batch of examples
    num_examples = 1000  # Full dataset generation
    first_batch = dataset[:num_examples]
    questions = first_batch.get("question", [])

    print(f"🔄 Processing {len(questions)} examples through complete pipeline...")

    results = []
    for i, question in enumerate(tqdm(questions, desc="Processing examples", unit="ex")):
        print(f"\n--- Processing example {i + 1}/{len(questions)} ---")
        result = process_example(question)
        if result is not None:
            results.append(result)
        else:
            print(f"❌ Skipped example {i + 1}")

    if not results:
        print("❌ No valid examples generated!")
        return

    print(f"\n✅ Successfully processed {len(results)}/{len(questions)} examples")

    # Create the final dataset
    print("📦 Creating final dataset...")
    final_dataset = Dataset.from_list(results)

    print("📊 Dataset statistics:")
    print(f"  - Total examples: {len(results)}")
    print(
        f"  - Average cargo outputs per example: {sum(len(r['info']['cargo_outputs']) for r in results) / len(results):.1f}"
    )
    print(
        f"  - Average gold comments per example: {sum(len(r['info']['gold_comments']) for r in results) / len(results):.1f}"
    )

    # Show a sample
    if results:
        sample = results[0]
        print("\n📋 Sample entry:")
        print(f"  - Question length: {len(sample['question'])} chars")
        print(f"  - Cargo outputs: {len(sample['info']['cargo_outputs'])}")
        print(f"  - Gold comments: {len(sample['info']['gold_comments'])}")
        print(f"  - Gold code length: {len(sample['info']['gold_code'])} chars")

        # Only show first cargo output/comment if they exist (avoid IndexError)
        if sample["info"]["cargo_outputs"]:
            print(f"  - First cargo output: {sample['info']['cargo_outputs'][0][:100]}...")
        else:
            print("  - First cargo output: (none - clean code example)")

        if sample["info"]["gold_comments"]:
            print(f"  - First gold comment: {sample['info']['gold_comments'][0][:100]}...")
        else:
            print("  - First gold comment: (none - clean code example)")

    # Upload to Hugging Face
    hub_name = "ljt019/rust-review-coral"
    print(f"\n🚀 Uploading to Hugging Face as {hub_name}...")
    try:
        final_dataset.push_to_hub(hub_name)
        print(f"✅ Successfully uploaded to {hub_name}!")
        print(f"🔗 View at: https://huggingface.co/datasets/{hub_name}")
    except Exception as e:
        print(f"❌ Error uploading: {e}")
        print("Make sure you're logged in with: huggingface-cli login")

    print("\n🎉 Final data generation pipeline completed!")


if __name__ == "__main__":
    main()
