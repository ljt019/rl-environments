import verifiers as vf

from environments.battleship.battleship import load_environment

############## Training Config ##############

MODEL_NAME = "Qwen/Qwen3-1.7B"

RUN_NAME = "battleship-grpo-090925"

#############################################

env = load_environment()

model, tokenizer = vf.get_model_and_tokenizer(MODEL_NAME)

args = vf.grpo_defaults(run_name=RUN_NAME)

# Battleship prompts are longer due to game instructions
args.max_prompt_length = 2048

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)

trainer.train()
