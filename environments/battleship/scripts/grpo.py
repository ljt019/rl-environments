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
args.max_tokens = 1024  # MUST be small to fit in 40960 context limit
args.max_seq_len = 43_008  # Large enough for long battleship games (prompt + completion)

print(f"DEBUG: args.max_tokens = {args.max_tokens}")
print(f"DEBUG: args.max_seq_len = {args.max_seq_len}")

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=args,
)

print(f"DEBUG: trainer.max_tokens = {trainer.max_tokens}")
print(f"DEBUG: trainer.max_seq_len = {trainer.max_seq_len}")

trainer.train()
