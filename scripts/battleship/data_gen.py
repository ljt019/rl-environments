import os

from dotenv import load_dotenv
from openai import OpenAI

from environments.battleship.battleship import load_environment

################ Data Gen Config #################

NUM_EXAMPLES = 1
ROLLOUTS_PER_EXAMPLE = 1

MAX_TURNS = 3
NUM_GAMES = NUM_EXAMPLES

OPENROUTER_MODEL_NAME = "qwen/qwen3-30b-a3b"
MAX_CONCURRENT = 10

MIN_SCORE_THRESHOLD = 0.0
DATASET_HUB_NAME = "ljt019/battleship-testy"

##################################################

load_dotenv()

# Get required environment variables
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

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


# filter to just prompt, completion, answer, reward >= MIN_SCORE_THRESHOLD, ignoring the other columns
env.make_dataset(results).select_columns(["prompt", "completion", "answer", "reward", "task"]).filter(
    lambda x: x["reward"] >= MIN_SCORE_THRESHOLD
).push_to_hub(DATASET_HUB_NAME)
