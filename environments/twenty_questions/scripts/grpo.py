import os

import verifiers as vf
from environments.twenty_questions.twenty_questions import load_environment

############## Training Config ##############

MODEL_NAME = "willcb/Qwen3-4B"

RUN_NAME = "twenty-questions-grpo-190925"

#############################################

env = load_environment(
    answerer_model="gpt-4.1-nano",
    answerer_base_url="https://openrouter.ai/api/v1",
    answerer_api_key=os.getenv("OPENROUTER_API_KEY"),
)

model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)

args = vf.grpo_defaults(run_name=RUN_NAME)
args.async_generation_timeout = 1800.0  # 30 minutes

args.max_tokens = 512

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)

trainer.train()
