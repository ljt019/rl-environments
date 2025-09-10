import verifiers as vf
from environments.battleship.battleship import load_environment

############## Training Config ##############

MODEL_NAME = "willcb/Qwen3-1.7B"

RUN_NAME = "battleship-grpo-090925"

#############################################

env = load_environment(max_turns=40)

model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)

args = vf.grpo_defaults(run_name=RUN_NAME)

# Battleship games are longer than average, increase max prompt length and timeout
args.max_prompt_length = 2048
args.async_generation_timeout = 1200.0

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)

trainer.train()
