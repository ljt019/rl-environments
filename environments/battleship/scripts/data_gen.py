# /// script
# dependencies = [
#   "datasets",
#   "huggingface-hub",
#   "dotenv",
#   "openai",
#   "verifiers",
# ]
# ///

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import battleship
sys.path.append(str(Path(__file__).parent.parent))

from battleship import load_environment
from dotenv import load_dotenv  # type: ignore
from openai import OpenAI

################ Data Gen Config #################

NUM_EXAMPLES = 1
ROLLOUTS_PER_EXAMPLE = 1

MAX_TURNS = 3
NUM_GAMES = NUM_EXAMPLES

OPENROUTER_MODEL_NAME = "qwen/qwen3-30b-a3b"
MAX_CONCURRENT = 10

MIN_SCORE_THRESHOLD = 0.0
DATASET_HUB_NAME = "ljt019/battleship-testy"

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

##################################################

load_dotenv()

# Get required environment variables
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL

if not api_key:
    raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY must be set")
if not base_url:
    raise ValueError("OPENAI_BASE_URL must be set")

env = load_environment(max_turns=MAX_TURNS, num_games=NUM_GAMES)

client = OpenAI(api_key=api_key, base_url=base_url)

results = env.evaluate(
    client,
    OPENROUTER_MODEL_NAME,
    num_examples=NUM_EXAMPLES,
    rollouts_per_example=ROLLOUTS_PER_EXAMPLE,
    max_concurrent=MAX_CONCURRENT,
)

# for each example, get the last assistant message in completion, get env.env_response(messages, game_state)
# then add the env response message to the results
for i in range(len(results.completion)):
    completion = results.completion[i]
    game_state = results.state[i]
    # Get environment's response to the last state
    env_messages, updated_state = env.env_response(completion, game_state)

    # Add environment messages to completion
    if env_messages:  # Only add if there are messages
        results.completion[i].extend(env_messages)
        results.state[i] = updated_state


# Create custom dataset with only prompt, completion, reward, and info columns
from datasets import Dataset

dataset_rows = []
base_dataset = env.make_dataset(results)

for i, row in enumerate(base_dataset):
    if row["reward"] >= MIN_SCORE_THRESHOLD:
        # Extract game state information for info column
        state = results.state[i] if i < len(results.state) else {}
        emulator_state = state.get("emulator_state", {})

        # Build info dictionary with actual underlying data
        info = {
            "turn": state.get("turn", 0),
            "victory": state.get("victory", False),
            "emulator_state": emulator_state,  # Full emulator state for reward functions
        }

        dataset_row = {"prompt": row["prompt"], "completion": row["completion"], "reward": row["reward"], "info": info}
        dataset_rows.append(dataset_row)

# Create and push dataset
filtered_dataset = Dataset.from_list(dataset_rows)
filtered_dataset.push_to_hub(DATASET_HUB_NAME)
