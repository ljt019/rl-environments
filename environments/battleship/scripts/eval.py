import os

from dotenv import load_dotenv
from openai import OpenAI

from environments.battleship.battleship import load_environment

################ Eval Config ################

MAX_TURNS = 10
NUM_GAMES = 100

NUM_EXAMPLES = 2
NUM_ROLLOUTS_PER_EXAMPLE = 1

OPENROUTER_MODEL_NAME = "qwen/qwen3-235b-a22b-2507"

MAX_CONCURRENT = 8

#############################################

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
    rollouts_per_example=NUM_ROLLOUTS_PER_EXAMPLE,
    max_concurrent=MAX_CONCURRENT,
)

rewards = [result.reward for result in results]
top_3_results = sorted(results, key=lambda x: x.reward, reverse=True)[:3]

# print top 3 results
print("\nTop 3 Rewards:")
print("─" * 30)
for i, result in enumerate(top_3_results, 1):
    print(f"{i}. Reward: {result.reward:.3f}")
print()
