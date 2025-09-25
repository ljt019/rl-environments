import httpx
from openai import OpenAI

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


def get_code_from_applied_comments(model, client, completion, state):
    print("[RUST_REVIEW] get_code_from_applied_comments: start")
    """
    Lazy generation - only runs once per rollout.
    Gets the original code from state, extracts review comments from completion,
    and uses the coder model to apply the comments to generate improved code.
    """
    # Check if we already generated the refined code (and it's not a cached None)
    if "refined_code" in state and state["refined_code"] is not None:
        print("[RUST_REVIEW] get_code_from_applied_comments: returning cached refined code")
        return state["refined_code"]

    # Extract original code from the user prompt
    original_code = extract_rust_code_from_state(state)
    if not original_code:
        print("[DEBUG] No original code found in state")
        print("[RUST_REVIEW] get_code_from_applied_comments: no original code")
        state["refined_code"] = None
        return None

    # Extract review comments from completion
    from .custom_parser import CustomParser

    parser = CustomParser()
    comments = parser.parse_answer(completion)
    print(f"[RUST_REVIEW] get_code_from_applied_comments: parsed {len(comments) if comments else 0} comments")

    if not comments:
        print("[DEBUG] No review comments found; using original code as refined_code")
        print("[RUST_REVIEW] get_code_from_applied_comments: no comments -> returning original")
        state["refined_code"] = original_code
        state["original_code"] = original_code
        state["comments_applied"] = []
        return original_code

    # Format comments as a bullet list
    comments_text = "\n".join([f"- {comment}" for comment in comments])

    try:
        # Call the coder model to apply the comments
        print(f"[DEBUG] Applying {len(comments)} review comments to code...")
        print(f"[RUST_REVIEW] get_code_from_applied_comments: requesting model={model} comments={len(comments)}")

        response = client.chat.completions.create(
            model=model,  # You can make this configurable
            messages=[
                {"role": "system", "content": CODER_SYSTEM_PROMPT},
                {"role": "user", "content": CODER_PROMPT.format(code=original_code, comments=comments_text)},
            ],
            temperature=0.0,  # Use temperature=0.0 for maximum consistency
            max_tokens=4000,
            # Add additional constraints
            stop=[
                "\n\n```",
                "Additional improvements:",
                "Note:",
                "Also:",
            ],  # Stop if model tries to add extra content
        )
        print("[RUST_REVIEW] get_code_from_applied_comments: received response")
        refined_code_response = response.choices[0].message.content
        print(
            f"[RUST_REVIEW] get_code_from_applied_comments: response length={len(refined_code_response) if refined_code_response else 0}"
        )

        # Extract the refined code from the response
        refined_code = extract_rust_code(refined_code_response)
        print(
            "[RUST_REVIEW] get_code_from_applied_comments: extracted refined code"
            if refined_code
            else "[RUST_REVIEW] get_code_from_applied_comments: failed to extract refined code"
        )

        if refined_code:
            # Validate that changes seem reasonable (basic sanity check)
            original_lines = len(original_code.split("\n"))
            refined_lines = len(refined_code.split("\n"))
            line_diff = abs(refined_lines - original_lines)
            print(f"[RUST_REVIEW] get_code_from_applied_comments: line_diff={line_diff}")

            # If the code changed dramatically, it might have made unauthorized changes
            if line_diff > len(comments) * 3:  # Allow up to 3 lines per comment
                print(
                    f"[DEBUG] Warning: Large code change detected ({line_diff} line difference). "
                    f"Model may have made unauthorized changes beyond the {len(comments)} comments."
                )

            print(f"[DEBUG] Successfully generated refined code ({len(refined_code)} chars, {line_diff} line diff)")
            state["refined_code"] = refined_code
            state["original_code"] = original_code
            state["comments_applied"] = comments  # Store the comments that were applied
            return refined_code
        else:
            print("[DEBUG] Failed to extract refined code from model response")
            print("[RUST_REVIEW] get_code_from_applied_comments: returning None (no refined code)")
            state["refined_code"] = None
            return None

    except Exception as e:
        print(f"[DEBUG] Error applying comments: {e}")
        print("[RUST_REVIEW] get_code_from_applied_comments: exception raised")
        state["refined_code"] = None
        return None


def extract_rust_code_from_state(state):
    """Extract the original Rust code from the user prompt in state"""
    prompt = state.get("prompt", [])

    # Find the user message
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
    http_client = httpx.Client(limits=limits, timeout=timeout_obj)
    return OpenAI(base_url=api_base_url, api_key=api_key, max_retries=max_retries, http_client=http_client)


def _setup_rust_project(code: str) -> str:
    """Creates a temporary Rust project with the given code."""
    import os
    import uuid

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


def _run_cargo_command(command: str, code: str) -> bool:
    """Runs a cargo command and returns success status."""
    import os
    import shutil
    import subprocess
    from shutil import which

    project_dir = _setup_rust_project(code)

    try:
        # Ensure cargo is on PATH for non-login shells
        env = os.environ.copy()
        candidate_paths = []
        cargo_home = env.get("CARGO_HOME")
        if cargo_home:
            candidate_paths.append(os.path.join(cargo_home, "bin"))
        # Default user install location
        candidate_paths.append(os.path.expanduser("~/.cargo/bin"))
        # Common alternate
        candidate_paths.append("/usr/local/cargo/bin")

        path_parts = env.get("PATH", "").split(os.pathsep)
        for p in candidate_paths:
            if p and p not in path_parts and os.path.isdir(p):
                path_parts.insert(0, p)
        env["PATH"] = os.pathsep.join(path_parts)

        # Optional: surface a clearer error if cargo still missing
        if which("cargo", path=env["PATH"]) is None:
            raise FileNotFoundError("cargo not found on PATH. Ensure Rust is installed and ~/.cargo/bin is on PATH.")

        result = subprocess.run(
            ["cargo", command, "--quiet"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        success = result.returncode == 0
    except Exception as e:
        print(f"Error running cargo {command}: {e}")
        success = False
    finally:
        # Clean up outputs directory
        shutil.rmtree(project_dir, ignore_errors=True)

        tests_dir = os.path.join("outputs", "tests")
        try:
            if os.path.exists(tests_dir) and not os.listdir(tests_dir):
                os.rmdir(tests_dir)
        except OSError:
            pass

    return success


def run_cargo_tests(code: str) -> bool:
    """Run cargo test on the provided code."""
    return _run_cargo_command("test", code)


def run_cargo_build(code: str) -> bool:
    """Run cargo build on the provided code."""
    return _run_cargo_command("build", code)


def run_cargo_clippy(code: str) -> bool:
    """Run cargo clippy on the provided code."""
    return _run_cargo_command("clippy", code)
