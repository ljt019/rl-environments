import json
import os
import shutil
import subprocess
from shutil import which

import httpx
from openai import AsyncOpenAI

from .custom_parser import CustomParser

REVIEW_APPLICATOR_MODEL_SYSTEM_PROMPT = """
You are a code editor that applies ONLY the specific changes mentioned in review comments. You must:

CRITICAL RULES:
1. Apply ONLY the exact changes specified in the review comments
2. Do NOT make any additional improvements, optimizations, or fixes beyond what's explicitly mentioned
3. Do NOT add new functionality, change variable names, or restructure code unless specifically requested
4. Do NOT fix other issues you might notice - only address the exact feedback given
5. If a comment is unclear or impossible to implement, leave that part of the code unchanged and return the original code.

Your job is to be a precise code editor, not a code improver. Apply the minimum changes necessary to address the specific feedback.

Preserve existing formatting unless changes are explicitly requested. Return ONLY the modified Rust code in a single ```rust code block. Do not include explanations.
"""

REVIEW_APPLICATOR_MODEL_PROMPT = """
Original Code:
```rust
{code}
```

Review Comments (apply ONLY these specific changes):
{comments}

IMPORTANT: Apply only the exact changes mentioned in the review comments above. Do not make any other modifications to the code. Return the minimally modified code in a ```rust block.
"""


async def get_code_from_applied_comments(model, client, completion, state):
    """
    Gets the original code from state, extracts review comments from completion,
    and uses the coder model to apply the comments to generate improved code.
    """
    # Check if we already generated the refined code (and it's not a cached None)
    if "refined_code" in state and state["refined_code"] is not None:
        return state["refined_code"]

    original_code = extract_rust_code_from_state(state)

    parser = CustomParser()
    comments = parser.parse_answer(completion)

    if not comments:
        state["refined_code"] = original_code
        state["original_code"] = original_code
        state["comments_applied"] = []
        return original_code

    # format comments as a bullet list
    comments_text = "\n".join([f"- {comment}" for comment in comments])

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REVIEW_APPLICATOR_MODEL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": REVIEW_APPLICATOR_MODEL_PROMPT.format(code=original_code, comments=comments_text),
                },
            ],
            temperature=0.0,
            max_tokens=4000,
        )
        refined_code_response = response.choices[0].message.content

        refined_code = extract_rust_code(refined_code_response)

        if refined_code:
            state["refined_code"] = refined_code
            state["original_code"] = original_code
            state["comments_applied"] = comments
            return refined_code
        else:
            print("[get_code_from_applied_comments] returning None (no refined code)")
            state["refined_code"] = None
            return None

    except Exception as e:
        print("[get_code_from_applied_comments] error: %s", e)
        state["refined_code"] = None
        return None


def extract_rust_code_from_state(state):
    """Extract the original Rust code from the user prompt in state"""
    prompt = state.get("prompt", [])

    for message in prompt:
        if message.get("role") == "user":
            content = message.get("content", "")
            code = extract_rust_code(content)
            if code:
                return code

    return None


def extract_rust_code(response):
    """Extract Rust code from response text (copied from rust_cargo)"""
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
        return ""


def setup_client(
    api_base_url, api_key, timeout=600.0, max_connections=100, max_keepalive_connections=50, max_retries=3
):
    timeout_obj = httpx.Timeout(timeout, connect=5.0)
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive_connections)
    http_client = httpx.AsyncClient(limits=limits, timeout=timeout_obj)
    client = AsyncOpenAI(
        base_url=api_base_url,
        api_key=api_key,
        max_retries=max_retries,
        http_client=http_client,
    )
    return client


def _setup_rust_project(code: str) -> str:
    """Creates a temporary Rust project with the given code."""
    import os
    import uuid

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
    with open(os.path.join(project_dir, "Cargo.toml"), "w") as f:
        f.write(cargo_toml)

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


def run_cargo_command(command: str, code: str) -> tuple[bool, list[str]]:
    """Runs a cargo command and returns success along with normalized diagnostics."""

    project_dir = _setup_rust_project(code)

    try:
        env = os.environ.copy()

        if which("cargo", path=env.get("PATH")) is None:
            raise FileNotFoundError("cargo not found on PATH. Ensure Rust is installed and PATH is set.")

        diagnostics: list[str] = []

        if command == "test":
            result = subprocess.run(
                ["cargo", "test", "--quiet"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            diagnostics = _parse_cargo_test_output(stdout, stderr)
        else:
            result = subprocess.run(
                ["cargo", command, "--message-format", "json"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            stdout = result.stdout or ""
            diagnostics = _parse_cargo_json_messages(stdout)

        success = result.returncode == 0
    except Exception as e:
        print(f"[run_cargo_command] error running cargo {command}: {e}")
        success = False
        diagnostics = [f"[internal] error {e}"]
    finally:
        # Clean up outputs directory
        shutil.rmtree(project_dir, ignore_errors=True)

        tests_dir = os.path.join("outputs", "tests")
        try:
            if os.path.exists(tests_dir) and not os.listdir(tests_dir):
                os.rmdir(tests_dir)
        except OSError:
            pass

    return success, diagnostics


def _parse_cargo_json_messages(stdout: str) -> list[str]:
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

        message_text = msg.get("message", "").strip()

        if location:
            formatted_diag = f"[{code}] {level} {location} {message_text}".strip()
        else:
            formatted_diag = f"[{code}] {level} {message_text}".strip()

        diagnostics.append(formatted_diag)

    deduped: list[str] = []
    seen: set[str] = set()
    for diag in diagnostics:
        key = diag.strip()
        if key not in seen:
            deduped.append(diag)
            seen.add(key)

    return deduped


def _parse_cargo_test_output(stdout: str, stderr: str) -> list[str]:
    import re

    diagnostics: list[str] = []
    combined = "\n".join([stdout or "", stderr or ""]).splitlines()
    for line in combined:
        line = line.strip()
        if not line:
            continue
        if "panicked at" in line:
            location_match = re.search(r"panicked at ([^:]+:\d+:\d+)", line)
            location = location_match.group(1) if location_match else ""
            diagnostics.append(f"[test_failure] error {location} {line}")
    return diagnostics
