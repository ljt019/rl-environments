import os

from dotenv import load_dotenv
from openai import OpenAI

from environments.rust_cargo.rust_cargo import load_environment

################ Eval Config ################

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


env = load_environment()

client = OpenAI(api_key=api_key, base_url=base_url)

results = env.evaluate(
    client,
    OPENROUTER_MODEL_NAME,
    num_examples=NUM_EXAMPLES,
    rollouts_per_example=NUM_ROLLOUTS_PER_EXAMPLE,
    max_concurrent=MAX_CONCURRENT,
)

rewards = results.reward  # This is a list[float]

# Get statistics
print(f"Total rollouts: {len(rewards)}")
print(f"Average reward: {sum(rewards) / len(rewards):.3f}")
print(f"Min reward: {min(rewards):.3f}")
print(f"Max reward: {max(rewards):.3f}")

# Get top 3
top_3_results = sorted(rewards, reverse=True)[:3]
print(f"Top 3 rewards: {top_3_results}")

# Also check metrics if you want
if results.metrics:
    print("\nAdditional metrics:")
    for metric_name, metric_values in results.metrics.items():
        avg_value = sum(metric_values) / len(metric_values)
        print(f"  {metric_name}: {avg_value:.3f}")
