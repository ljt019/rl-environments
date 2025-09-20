import os
import subprocess

import verifiers as vf
from prime_cli.api.client import APIClient
from prime_cli.api.pods import PodsClient

from environments.twenty_questions.twenty_questions import load_environment

############## Training Config ##############

MODEL_NAME = "willcb/Qwen3-4B"

HUB_MODEL_ID = "ljt019/Qwen3-4B-20Q"

RUN_NAME = "twenty-questions-grpo-200925"

#############################################


def kill_current_pod():
    """Kill the current pod by matching hostname"""
    try:
        # Get hostname
        hostname = subprocess.check_output(["hostname"], text=True).strip()

        # Find and kill matching pod
        client = APIClient()
        pods_client = PodsClient(client)
        pods = pods_client.list()

        for pod in pods.data:
            if pod.status == "ACTIVE" and (hostname in pod.id or (pod.name and hostname in pod.name)):
                pods_client.delete(pod_id=pod.id)
                print(f"Successfully killed pod '{pod.id}' (matched hostname: {hostname})")
                return True

        print(f"No matching pod found for hostname: {hostname}")
        return False

    except Exception as e:
        print(f"Failed to kill pod: {e}")
        return False


env = load_environment(
    answerer_model="gpt-4.1-nano",
    answerer_base_url="https://openrouter.ai/api/v1",
    answerer_api_key=os.getenv("OPENROUTER_API_KEY"),
)

model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)

args = vf.grpo_defaults(run_name=RUN_NAME)

args.max_tokens = 512
args.max_seq_len = 2048

args.save_steps = 25

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
)

try:
    trainer.train()
except Exception as e:
    print(f"Training failed: {e}")
    print("Attempting to kill current pod...")
    kill_current_pod()
